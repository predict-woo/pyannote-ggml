# Integrating Whisper Transcription with Pyannote Streaming Diarization

All pipelines operate on the **filtered timeline** — the true timeline (pre-silence-filter) is not used after the filter stage, except for the 5-second silence flush which is measured in true-timeline time by the VAD filter.

## Pipeline Overview

```
Raw Audio → VAD Silence Filter → Filtered Audio
                  │                      │
                  │         ┌────────────┬┘
                  │         ▼            ▼
                  │  Audio Buffer    Pyannote Streaming
                  │         │            │
                  │         │      VAD prediction frames
                  │         │            │
                  │         │     Segment End Detector ←──┐
                  │         │            │                 │
                  │         ├────────────┘                 │
                  │         ▼                              │
                  │   Whisper Transcriber ←────────────────┘
                  │         │                    (5s silence flush)
                  │   Word-level tokens
                  │         │
                  │         ▼
                  └→ Alignment + Recluster
                            │
                            ▼
                      Callback (labeled segments + words)
```

---

## 1. VAD Silence Filter

Limits silence to at most 2 seconds anywhere in the audio while preserving natural speech boundaries (no abrupt starts/ends).

### Input

Audio frames arrive in variable-sized chunks (e.g., 512 one time, 2000 the next). The VAD processes in 512-frame units, so incoming frames are buffered internally until ≥512 frames are available. Each 512-frame batch is processed and the audio + VAD result is passed to the next stage.

### Silence Buffer

| Buffer | Type | Size | Purpose |
|--------|------|------|---------|
| `endSilence` | Circular/FIFO | 1 second | Rolling window of recent silence — prepended to next speech onset for smooth transitions |

Buffer starts empty on initialization.

### Logic

**When silent frames arrive:**
- If `silence_started` does not exist → set it to current frame number, pass frames through
- If `silence_started` exists:
  - If elapsed since `silence_started` ≤ 1 second → pass frames through (allow natural trailing silence)
  - If elapsed > 1 second → write frames into `endSilence` buffer (FIFO, 1s capacity — oldest frames are overwritten)
    - Track `consecutive_discarded_frames` (incremented for every frame written to the circular buffer that overwrites an existing frame, i.e., once the buffer is full)
    - If `consecutive_discarded_frames` reaches 5 seconds worth of frames (80,000 at 16kHz) → call the **flush function** on the Segment End Detector (see §3), then reset the counter

**When speech frames arrive:**
- If `silence_started` exists → set it to null, reset `consecutive_discarded_frames` to 0
  - If `endSilence` has frames → concat `endSilence` contents + speech frames, clear buffer, pass concatenated frames through
  - If `endSilence` is empty → pass speech frames through
- If `silence_started` does not exist → pass speech frames through

**Effect:** Up to 1 second of silence passes through after speech ends. Any silence beyond that is absorbed. When speech resumes, the most recent 1 second of silence (from `endSilence`) is prepended so the transition sounds natural. Silence gaps > 2 seconds are compressed to ~2 seconds (1s trailing + 1s leading). If silence lasts long enough (5 seconds of discarded frames ≈ 6 seconds of total silence), the downstream pipeline is flushed to avoid holding audio indefinitely.

---

## 2. Audio Buffer

Holds filtered audio until it can be chunked and sent to Whisper.

### State

| Field | Type | Purpose |
|-------|------|---------|
| `buffer` | FIFO | Accumulates filtered audio frames |
| `dequeued_frames` | int64 | Total frames ever dequeued — offset for timestamp calculation |

### Timestamp Calculation

For any frame at position `i` in the buffer:
```
absolute_frame = dequeued_frames + i
time = absolute_frame / sample_rate
```

### Operations

- **Enqueue**: Append incoming frames from the silence filter
- **Dequeue up to timestamp**: Convert a filtered-timeline timestamp to a sample position (`round(time × sample_rate)`), remove frames from front up to that position, increment `dequeued_frames`
- **Read range**: Extract audio between two absolute frame positions (for sending to Whisper)

---

## 3. Segmentation (Segment End Detection)

Uses pyannote's streaming VAD output to find speech→silence transitions for chunking audio into Whisper.

### Pyannote Integration

- Each time filtered audio arrives, push it into pyannote via `streaming_push` (zero-latency mode)
- Pyannote internally buffers until it has 16,000 samples (1 second), then processes a chunk
- A single push of N seconds of audio may produce N VADChunks — process them all in a loop
- Each chunk has `num_frames` prediction frames (~59 or 60, variable)
- Frames tile perfectly with no overlap or gap
- Frame timing: `time = frame_number × 0.016875`

### Frame Counting

Maintain a `current_prediction_frame` counter (starts at 0). Each time a VADChunk arrives, add its `num_frames` to the counter. This gives exact frame→time correspondence:

```
segment_end_time = frame_number × 0.016875
```

The 589:10s ratio is the underlying truth, but we never need to compute it — cumulative frame counting handles the alternating 59/60 pattern automatically.

### Segment End Detection Logic

Maintain `last_frame_was_speech` (bool, initially false).

When a VADChunk arrives:
1. **Check cross-boundary transition**: If `last_frame_was_speech == true` and `vad[0] == 0.0`:
   - Segment end detected at `(current_prediction_frame) × 0.016875` (the first frame of the new chunk)
   - Send this timestamp to the Transcription stage
2. **Scan within chunk**: Search for the first `speech → non-speech` transition within the chunk's `vad` array:
   - Find first index `i` where `vad[i] == 1.0` followed by `vad[i+1] == 0.0` (or the first `0.0` after any `1.0`)
   - If found: segment end at `(current_prediction_frame + i + 1) × 0.016875`
   - Send this timestamp to the Transcription stage
   - Note: only the first transition per chunk is detected. Later transitions within the same ~1s window are caught in subsequent chunks. This is intentional — one cut point per second is sufficient for Whisper chunking.
3. **Update state**: Set `last_frame_was_speech = (vad[num_frames - 1] != 0.0)`
4. **Increment counter**: `current_prediction_frame += num_frames`

### Flush Function (called by VAD filter)

Exposed to the VAD silence filter for the 5-second silence timeout. When called:
- Sends a **force-flush signal** to the Transcription stage (§4), bypassing the 20-second minimum threshold
- No segment-end timestamp is involved — the Transcription stage simply sends whatever is in the audio buffer
- If the audio buffer is empty, this is a no-op

---

## 4. Transcription (Whisper)

Receives segment-end timestamps from the Segmentation stage and decides when to send audio to Whisper.

### State

| Field | Type | Purpose |
|-------|------|---------|
| `buffer_start_time` | double | Filtered-timeline timestamp of first frame currently in audio buffer |

### Trigger Conditions

Audio is sent to Whisper when **any** of these conditions is met:

1. **Segment-end + minimum length**: A segment end is detected AND `segment_end_time - buffer_start_time ≥ 20 seconds`
   - Convert segment-end time to a sample position: `cut_sample = round(segment_end_time × sample_rate)`
   - Extract audio from buffer start to `cut_sample`
   - Dequeue buffer up to `cut_sample`
   - Send audio to Whisper

2. **Silence flush** (from VAD filter via Segment End Detector): Force-flush signal received
   - Extract all audio currently in the buffer
   - Dequeue entire buffer
   - Send audio to Whisper
   - If buffer is empty, no-op

3. **Finalize** (stream end): Flush remaining audio regardless of length (see §6)

### Output

Whisper returns word-level timestamps **relative to the start of the audio chunk sent**. Convert to filtered-timeline absolute timestamps before passing to the Alignment stage:

```
absolute_start = buffer_start_time + whisper_relative_start
absolute_end   = buffer_start_time + whisper_relative_end
```

Tokens passed to Alignment:

```
struct Token {
    std::string text;
    double start;    // filtered timeline (absolute)
    double end;      // filtered timeline (absolute)
};
```

---

## 5. Alignment (Diarization + Token Matching)

Combines pyannote speaker labels with Whisper word timestamps, following WhisperX's approach (`assign_word_speakers` from [m-bain/whisperX](https://github.com/m-bain/whisperX)).

### Core Algorithm: Maximum Intersection Assignment

For each word, find the speaker whose diarization segment has the most time overlap with the word's `[start, end]` range:

```
for each word with [start, end]:
    find all diarization segments overlapping [start, end]
    for each overlapping segment:
        intersection = min(seg_end, word_end) - max(seg_start, word_start)
        accumulate intersection per speaker
    word.speaker = speaker with maximum total intersection
```

- **Word straddling two speakers**: Whichever speaker has more overlap time wins
- **Word in a gap** (no diarization segment overlaps): Assign to the nearest diarization segment by midpoint distance (`(word_start + word_end) / 2` vs segment midpoints)
- **No pre-chunking needed**: Words are matched individually by time overlap. Whisper's natural segment boundaries (roughly sentence-level) are not used for speaker assignment — each word gets its own lookup

For efficiency with long audio, use an interval tree (sorted array + binary search) over diarization segments for O(log n) overlap queries per word.

### Data Stores

#### Token Storage (working set)

All tokens produced by Whisper that have not yet been assigned to a finished segment. Tokens are added when Whisper returns results and removed when matched to finished segments.

#### Finished Segment Storage (permanent)

Stores `(segment_boundaries, list_of_tokens)` tuples for diarization segments whose **end points are more than 10 seconds in the past**. These segments' binary activity boundaries (start/end times) are stable and will not change in future reclusters.

**Speaker labels in finished segments CAN change** — only the boundaries are stable. Labels are re-mapped on each recluster.

### Alignment Flow

When Whisper returns word-level tokens:

1. **Store tokens**: Add all new tokens to Token Storage

2. **Recluster**: Call `streaming_recluster()` on the pyannote state to get current `DiarizationResult` (segments with global speaker labels)

3. **Rebuild full mapping from scratch**: Build an interval tree from the new diarization segments. For every token (in both Token Storage and Finished Segment Storage):
   - Query the interval tree for overlapping diarization segments
   - Assign the token to the speaker with maximum intersection duration
   - If no overlap exists, assign to the nearest segment by midpoint distance
   - No diffing against previous recluster output — just overwrite all assignments

4. **Promote to finished**: For matched diarization segments whose end time is > 10 seconds before the current audio position:
   - Move the `(segment_boundaries, token_list)` into Finished Segment Storage
   - Remove those tokens from Token Storage

5. **Emit output**: Collect all diarization segments (both finished and in-progress) with their tokens and current speaker labels. Pass to callback.

---

## 6. Finalize (Stream End)

When the audio stream ends, flush the entire pipeline in order:

1. **Silence filter**: Flush `endSilence` buffer — pass any remaining frames through to both the audio buffer AND pyannote via `streaming_push`
2. **Pyannote**: Call `streaming_finalize()` — processes remaining partial chunks (zero-padded), runs final recluster. Returns complete `DiarizationResult`
3. **Whisper**: Send all remaining audio in the buffer to Whisper (regardless of length or segment-end detection)
4. **Alignment**: Final token matching against finalize's diarization result. All segments are now finished. Emit final output via callback.

After finalize, the pipeline is done. All resources can be freed.

---

## 7. Callback

```
callback(segments: [{
    speaker: string,       // "SPEAKER_00", "SPEAKER_01", ...
    start: double,         // filtered timeline seconds
    duration: double,      // seconds
    tokens: [{
        text: string,
        start: double,     // filtered timeline seconds
        end: double,       // filtered timeline seconds
    }]
}])
```

Called after each alignment step (§5.5) with the current complete state — both finished and in-progress segments. The consumer receives incremental updates as more audio is processed.

---

## Constants Reference

| Constant | Value | Origin |
|----------|-------|--------|
| Sample rate | 16,000 Hz | Input audio |
| VAD process size | 512 frames | VAD model requirement |
| Silence limit | 2 seconds | Design choice (1s trailing + 1s leading) |
| Silence flush threshold | 5 seconds of discarded frames | True-timeline; triggers Whisper flush via VAD filter |
| Pyannote push size | variable (internally buffers to 16,000) | Streaming API handles any input size |
| Prediction frames per push | ~59-60 (variable) | SincNet architecture |
| Frame time step | 0.016875s | 270 samples / 16kHz |
| Frames per 10s chunk | 589 | SincNet 3-stage conv+pool |
| Whisper min send length | 20 seconds | Design choice |
| Segment stability horizon | 10 seconds | Pyannote overlap window |
| Whisper max input | ~30 seconds | Whisper architecture limit |
