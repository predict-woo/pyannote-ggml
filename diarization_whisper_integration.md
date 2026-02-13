# Integrating Whisper transcription, Pyannote Diarization, and VAD

This entire process consists of multiple pipelines that input and output audio and prediction frames quite dynamically.

There are 2 timelines here.
The first if the `true timeline`, which is what the VAD silence filter sees.
The second is the `filtered timeline` which is the timeline of the audio that gets filtered by the VAD silence filter. This is what most pipelines use.

# The VAD silence filter

The first pipeline is a filter that only allows up to 2 seconds of silence anywhere in the audio while keeping surrounding silence of a segment so there are no abrupt starts and ends of speech.

It works as the following.

## The input

Audio frames are put into the VAD. They can be dynamic. For example, we might put 512 frames per input in one point then switch to 2000 frames per input in the next.
The VAD can process 512-frame data, so it buffers incomming frames until it is equal to or past the 512 frame mark then processes it. Then, this 512 frame audio with the VAD results are passed onto the next step of the pipeline.

## The silence buffers

These make sure the audio not to have sudden starts or stops.

We maintain two buffers.

| endSilence | CircularAudioBuffer | 1s | Rolling window of recent silence (for "before-next-speech" context) |

When the pipeline is initialized, we empty both buffers.

When audio frames come in with VAD data we use the following logic

- When silent frames come in
  - We check if the `silence_started` value exsists.
    - If it does not exist, we set the current frame number as the value
    - If it does exist, we check if it has been more than 1s since `silence_started`
      - If it has not been more than a second, we pass the frames to the next step of the pipeline
      - If it has been more than a second, we put the frames into the endSilence buffer, which is a fifo buffer with 1s in length
- When a speech frame comes in
  - We check if the `silence_started` value exsists.
    - If it does, we set the value to null
      - Then, we check if the endSilence buffer has frames or is empty
        - If it is empty, we leave it like that, then pass the frames to the next step of the pipeline
        - If it has frames, we concat the frames with the speach frames, then empty the buffer, then pass the concated frames to the next step of the pipeline

# The Diarize & Transcribe Pipeline

All timestamps in this pipeline work on the `filtered timeline`.

## Audio buffer

The point of this buffer is to hold audio until it can be chunked and put into whisper.

- Each time audio frames are passed in from the previous pipeline, we add them to a fifo buffer.
- We keep track of the length of the audio inside the pipeline, and also the global frame count from which we can get exact timestamps for any audio frame inside the buffer even after audio is dequeued and enqueued.
  - This can be done by just keeping an offset variable that is added to everytime the buffer is dequeued from.

## Segmentation

The point of this pipeline is to find the end of segments and pass the timestamps of the segment ends to the next step so we can chunk audio.

- Each time audio frames are passed in from the previous pipeline, we also `push` them into the pyannote diarization pipeline. We use zero latency mode for this.
- It should start returning prediction frames. They are either 59 or 60 frames but don't count on their length being fixed. They are produced everytime we put in 16k audio frames (1 second) so their time per frame is wildly different. keep this in mind.
  - These are not the final full accuracy frames and thus will not be used to run the actual diarization. Thus, it will only return vad activity. These are only used to chunk up audio for input into whisper.
- However, the produced frames do not overlap or miss parts of the audio, so if we keep count of how many frames have arrived, we can use the ratio (589 frames per 10 seconds) to keep the timestamps perfectly accurate.
- So, to do this, we keep track of a `current prediction frame number`.
  - Everytime a prediction frame is generated, we add the length to the current prediction frame number.
- Then, we check for segment ends everytime a prediction frame is sent.
  - To do this, we first search for the first speech -> non speech switch inside the prediction
    - If we do find this, we calculate the timestamp it would be on on the `filtered timeline` and we send it to the next step of the pipeline
    - However, there is the case where the switch happens on the edge of the prediction frames, in which case the switch would not be inside a single segment
      - To fix this, we keep the last prediction frame of the previous segment and check if the last frame is speech and the first frame of the new segment is no speech, in which case we directly return that timestamp to the next step of the pipeline.

## Transcription

- When the end of a segment is detected
  - We check if it meets specific criterias
    - Check if the time difference between audio buffer start and the segment end time given from the Segmentation pipeline is more than 20 seconds.
    - If so, we first get the audio up to the segmentation detected point then send the audio into the whisper pipeline.
    - Then, we should dequeue the buffer upto the segmentation point.
    - after we get the word level timestamps from the whisper pipeline, we send the data to the next step.
  - If it doesnt meet the criteria
    - We do nothing

## Alignment

### Finished segment storage

As the segmentation which have end points more than 10 seconds in the past is finished and will not change (the embeddding assignments might change, but the segments itself will not change.)

We can safely store a tuple (index of segments, a list of tokens for that segment) without it ever changing.

This is a storage that stores these connections to reduce re-computing it everytime.

### Token storage

This is a storage solution for all tokens and their timestamps produced by whisper. It's not fifo or lifo, we just put tokens in and out when they are added and matched.

It gets added into whenever this step receive outputs from whisper and removed as tokens are put into finished segment storage

### Diarization

- When the pipeline receives the word level timestamps from the whisper model, we run `streaming_recluster` to get the diarization results untill the current timestamp.
  - From the dirization results, we get the segments and check for segments we have not matched any words to yet.
  - We align the un-matched segments to the tokens in the token storage
    - If a matched segment is elligable for finished segment storage (end point more than 10 sec in the past), we put it in storage and remove it from token storage.
  - Then, we get all the diarization results, either stored in finisehd segment storage or not, then return it with speaker labels.

# Callback

As this entire pipeline is streaming, as the diarization segments with speaker labels are returned we pass this through a callback so the user can actually use the output.

this is what I am planning to create. Do you think this makes sense and fits with the current implementation of pyannote cpp diarization?
