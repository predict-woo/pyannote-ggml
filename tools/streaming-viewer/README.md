# Streaming Diarization Viewer

Real-time visualization of streaming speaker diarization.

## Requirements

- Python 3.8+
- Flask, flask-socketio
- Built `streaming_test` binary (see main project README)

## Setup

```bash
pip install flask flask-socketio python-socketio
```

## Usage

```bash
cd tools/streaming-viewer
python server.py
```

Then open http://localhost:5000 in your browser.

## How It Works

1. Upload a WAV file (16kHz mono)
2. Click "Start Streaming"
3. Watch segments appear in real-time as the audio is processed
4. Visual flash when reclustering happens (every 60 seconds)
5. Final results shown after processing completes

## Architecture

```
Browser                    Flask Server              streaming_test
   |                            |                          |
   |-- Upload WAV ------------->|                          |
   |<-- file_id ----------------|                          |
   |                            |                          |
   |-- WebSocket connect ------>|                          |
   |-- start_streaming -------->|                          |
   |                            |-- spawn subprocess ----->|
   |                            |<-- JSON line (push) -----|
   |<-- emit("push", data) -----|                          |
   |   [render segment]         |<-- JSON line (push) -----|
   |<-- emit("push", data) -----|                          |
   |   [render segment]         |<-- JSON line (recluster)-|
   |<-- emit("recluster") ------|                          |
   |   [flash + update]         |<-- JSON line (finalize)--|
   |<-- emit("finalize") -------|                          |
   |   [show final results]     |                          |
```

## JSON Event Format

The `streaming_test --json-stream` outputs JSON lines:

```json
{"type": "push", "chunk": 1, "time": 1.0, "segments": [...]}
{"type": "recluster", "chunk": 60, "time": 60.0, "segments": [...]}
{"type": "finalize", "segments": [...]}
```

Each segment: `{"start": 0.5, "duration": 2.0, "speaker": "SPEAKER_00"}`

## Keyboard Shortcuts

- **Space**: Play/Pause audio
- **←**: Back 5 seconds
- **→**: Forward 5 seconds
- **Home**: Go to start
