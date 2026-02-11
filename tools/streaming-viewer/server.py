#!/usr/bin/env python3
"""
Streaming Diarization Viewer - Flask + WebSocket Server

Runs streaming_test --json-stream on uploaded WAV files and streams
diarization results to the browser in real-time.
"""

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'streaming-diarization-viewer'
socketio = SocketIO(app, cors_allowed_origins="*")

PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAMING_TEST_BIN = PROJECT_ROOT / "diarization-ggml" / "build" / "bin" / "streaming_test"
PLDA_PATH = PROJECT_ROOT / "diarization-ggml" / "plda.gguf"
EMBEDDING_COREML = PROJECT_ROOT / "models" / "embedding-ggml" / "embedding.mlpackage"
SEG_COREML = PROJECT_ROOT / "models" / "segmentation-ggml" / "segmentation.mlpackage"

uploaded_files = {}


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return {'error': 'No file provided'}, 400
    
    file = request.files['file']
    if file.filename == '':
        return {'error': 'No file selected'}, 400
    
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)
    
    file_id = str(len(uploaded_files))
    uploaded_files[file_id] = {
        'path': temp_path,
        'filename': file.filename,
        'temp_dir': temp_dir
    }
    
    return {'file_id': file_id, 'filename': file.filename}


def run_streaming_test(file_id, sid):
    """Run streaming_test --json-stream and emit events to client."""
    file_info = uploaded_files.get(file_id)
    if not file_info:
        socketio.emit('error', {'message': 'File not found'}, room=sid)
        return
    
    audio_path = file_info['path']
    
    if not STREAMING_TEST_BIN.exists():
        socketio.emit('error', {
            'message': f'streaming_test not found at {STREAMING_TEST_BIN}. Run cmake --build.'
        }, room=sid)
        return
    
    cmd = [
        str(STREAMING_TEST_BIN),
        '--json-stream',
        '--plda', str(PLDA_PATH),
        '--coreml', str(EMBEDDING_COREML),
        '--seg-coreml', str(SEG_COREML),
        audio_path
    ]
    
    socketio.emit('status', {'message': 'Starting diarization...'}, room=sid)
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT / "diarization-ggml")
        )
        
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            
            try:
                event = json.loads(line)
                event_type = event.get('type', 'unknown')
                
                if event_type == 'push':
                    socketio.emit('push', event, room=sid)
                elif event_type == 'recluster':
                    socketio.emit('recluster', event, room=sid)
                elif event_type == 'finalize':
                    socketio.emit('finalize', event, room=sid)
                
                time.sleep(0.05)
                
            except json.JSONDecodeError:
                pass
        
        process.wait()
        
        stderr_output = process.stderr.read()
        if process.returncode != 0:
            socketio.emit('error', {'message': f'Process failed: {stderr_output}'}, room=sid)
        else:
            socketio.emit('complete', {'message': 'Diarization complete'}, room=sid)
            
    except Exception as e:
        socketio.emit('error', {'message': str(e)}, room=sid)


@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to server'})


@socketio.on('start_streaming')
def handle_start_streaming(data):
    file_id = data.get('file_id')
    if not file_id:
        emit('error', {'message': 'No file_id provided'})
        return
    
    sid = request.sid
    thread = threading.Thread(target=run_streaming_test, args=(file_id, sid))
    thread.start()


if __name__ == '__main__':
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Streaming test binary: {STREAMING_TEST_BIN}")
    print(f"Binary exists: {STREAMING_TEST_BIN.exists()}")
    print("\nStarting server at http://localhost:8080")
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)
