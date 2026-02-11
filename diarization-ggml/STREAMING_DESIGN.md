# Streaming Diarization Design (Periodic Reclustering)

## Overview

Real-time diarization with periodic full reclustering. Final result identical to offline.

```
Audio Stream ─┬─► Segmentation ─► Embedding ─► Buffer
              │                                  │
              │   ┌──────────────────────────────┘
              │   │
              │   ▼
              │  [Every 60s] Full Recluster (AHC + VBx)
              │       │
              │       ▼
              │   Centroids + Global Labels
              │       │
              └───────┼─► Provisional Assignment ─► Output
                      │
              [Stream End] Final Recluster ─► Identical to Offline
```

---

## State Management

```cpp
struct StreamingState {
    // Accumulated data (grows over time)
    std::vector<float> embeddings;      // [N × 256]
    std::vector<int> chunk_idx;         // which chunk each embedding came from
    std::vector<int> local_speaker_idx; // which local speaker (0-2)
    std::vector<float> binarized;       // [num_chunks × 589 × 3]

    // Current clustering result (updated every recluster)
    std::vector<float> centroids;       // [K × 256]
    std::vector<int> global_labels;     // [N] global speaker for each embedding
    int num_speakers;

    // Bookkeeping
    int chunks_processed;
    int last_recluster_chunk;
    double audio_time_processed;
};
```

---

## API

```cpp
// Initialize
streaming_state* streaming_init(const DiarizationConfig& config);

// Feed audio chunk (1s of new audio). Returns provisional segments.
std::vector<Segment> streaming_push(streaming_state* state,
                                     const float* samples,
                                     int num_samples);

// Force recluster now (optional, normally happens automatically)
void streaming_recluster(streaming_state* state);

// Finalize and get offline-identical result
DiarizationResult streaming_finalize(streaming_state* state);

// Cleanup
void streaming_free(streaming_state* state);
```

---

## Processing Pipeline

### 1. Per-Chunk Processing (~25ms)

```
Input: 1s new audio
       ↓
┌─────────────────────────────────────┐
│ 1. Append to 10s sliding window     │
│ 2. Run segmentation (CoreML, 12ms)  │
│ 3. Powerset → multilabel            │
│ 4. Extract embeddings (CoreML, 13ms)│
│ 5. Filter & store valid embeddings  │
│ 6. Provisional assign via centroids │
│ 7. Output provisional segments      │
└─────────────────────────────────────┘
```

### 2. Periodic Recluster (~800ms every 60s)

```
Trigger: chunks_processed - last_recluster_chunk >= 60
         ↓
┌─────────────────────────────────────┐
│ 1. L2-normalize all embeddings      │
│ 2. PLDA transform                   │
│ 3. AHC clustering (fastcluster)     │
│ 4. VBx refinement                   │
│ 5. Compute new centroids            │
│ 6. Reassign all embeddings          │
│ 7. Update global_labels             │
│ 8. Emit corrected segments          │
└─────────────────────────────────────┘
```

### 3. Finalization

Same as recluster, but marks stream as complete. Result identical to `diarize()`.

---

## Provisional Assignment (Between Reclusters)

Use centroids from last recluster + Hungarian assignment per chunk.

```
For new embedding e in chunk c:
    1. Compute cosine_similarity(e, centroid[k]) for all k
    2. Build cost matrix [3 local speakers × K global speakers]
    3. Hungarian assignment (same as offline step 16)
    4. If all similarities < threshold → mark as "pending new speaker"
```

**Before first recluster**: Either wait 30s, or use simple heuristic (local speaker 0 → global 0, etc.) then correct.

---

## Output Modes

### Mode 1: Provisional Only

Emit segments immediately. Client understands labels may shift at recluster.

### Mode 2: Confirmed Window

Only emit segments older than `recluster_interval`. These won't change.

### Mode 3: Diff Updates

Emit adds/removes/changes when recluster corrects past assignments.

---

## Key Parameters

| Parameter                     | Default | Description                            |
| ----------------------------- | ------- | -------------------------------------- |
| `recluster_interval`          | 60s     | Time between full reclusters           |
| `min_audio_for_first_cluster` | 30s     | Wait before first recluster            |
| `new_speaker_threshold`       | 0.6     | Cosine distance to declare new speaker |

---

## Edge Cases

### New Speaker Appears Mid-Stream

- Provisional: Assigned to closest existing centroid (wrong)
- After recluster: VBx discovers new cluster, corrects labels

### Speaker Returns After Long Silence

- Centroids persist, so returning speaker matched correctly

### Very Short Stream (<30s)

- Skip periodic recluster, just finalize at end

---

## File Structure

```
diarization-ggml/
├── src/
│   ├── streaming.h          # Public API
│   ├── streaming.cpp        # Implementation
│   ├── streaming_state.h    # State struct
│   └── provisional.cpp      # Centroid-based assignment
├── include/
│   └── diarization_stream.h # C API for bindings
└── tests/
    └── test_streaming.cpp   # Streaming accuracy tests
```

---

## Testing Strategy

1. **Correctness**: `streaming_finalize()` output == `diarize()` output (byte-identical RTTM)
2. **Latency**: Measure `streaming_push()` time, must be < chunk duration (1s)
3. **Memory**: Track embedding buffer growth, ensure reasonable for long streams
4. **Recluster overhead**: Confirm <1s for typical meeting lengths

---

## Future Optimizations

- **Incremental PLDA**: Only transform new embeddings, cache old
- **Warm-start VBx**: Initialize from previous iteration
- **Adaptive recluster interval**: More frequent early, less frequent once stable
- **GPU batching**: Batch multiple chunks if processing falls behind
