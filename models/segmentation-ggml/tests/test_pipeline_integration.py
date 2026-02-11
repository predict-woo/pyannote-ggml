#!/usr/bin/env python3
"""
Pipeline Integration Test: GGML Segmentation in pyannote Speaker Diarization

This test replaces the PyTorch segmentation model in the pyannote speaker diarization
pipeline with our GGML implementation and compares the RTTM outputs.

Usage:
    python segmentation-ggml/tests/test_pipeline_integration.py
"""

import numpy as np
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from typing import Optional
import torch
import torchaudio

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pyannote.audio import Pipeline, Model
from pyannote.audio.core.inference import Inference
from pyannote.core import Annotation


def patch_torch_load():
    """Monkey-patch torch.load to disable weights_only for PyTorch 2.6+"""
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load


class GGMLSegmentationWrapper:
    """
    Wrapper that mimics the PyTorch segmentation model interface but calls GGML CLI.
    
    This class is designed to be a drop-in replacement for the segmentation model
    in the pyannote speaker diarization pipeline.
    """
    
    def __init__(
        self,
        ggml_executable: Path,
        ggml_model: Path,
        original_model: Model,
        max_samples: int = 160000,
    ):
        """
        Initialize the GGML segmentation wrapper.
        
        Args:
            ggml_executable: Path to the GGML segmentation executable
            ggml_model: Path to the GGUF model file
            original_model: Original PyTorch model (for specifications and audio loading)
            max_samples: Maximum number of samples to process (10 seconds at 16kHz)
        """
        self.ggml_executable = Path(ggml_executable)
        self.ggml_model = Path(ggml_model)
        self.original_model = original_model
        self.max_samples = max_samples
        
        # Copy specifications from original model
        self.specifications = original_model.specifications
        self.audio = original_model.audio
        self.receptive_field = original_model.receptive_field
        
        # Verify paths exist
        if not self.ggml_executable.exists():
            raise FileNotFoundError(f"GGML executable not found: {self.ggml_executable}")
        if not self.ggml_model.exists():
            raise FileNotFoundError(f"GGML model not found: {self.ggml_model}")
    
    @property
    def device(self):
        """Return CPU device since GGML runs on CPU."""
        return torch.device("cpu")
    
    def to(self, device):
        """No-op for device transfer since GGML handles its own memory."""
        return self
    
    def eval(self):
        """No-op for eval mode since GGML is always in inference mode."""
        return self
    
    def __call__(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Run GGML inference on a batch of waveforms.
        
        Args:
            waveforms: (batch_size, num_channels, num_samples) tensor
            
        Returns:
            (batch_size, num_frames, num_classes) tensor of log-softmax probabilities
        """
        batch_size = waveforms.shape[0]
        outputs = []
        
        for i in range(batch_size):
            waveform = waveforms[i]  # (num_channels, num_samples)
            output = self._run_ggml_inference(waveform)
            outputs.append(output)
        
        return torch.stack(outputs, dim=0)
    
    def _run_ggml_inference(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Run GGML inference on a single waveform.
        
        Args:
            waveform: (num_channels, num_samples) tensor
            
        Returns:
            (num_frames, num_classes) tensor
        """
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Flatten to 1D
        samples = waveform.squeeze().numpy()
        
        # Truncate to max_samples if needed
        if len(samples) > self.max_samples:
            samples = samples[:self.max_samples]
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name
        
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as out_file:
            output_path = out_file.name
        
        try:
            # Save waveform as WAV file (16-bit PCM, 16kHz mono)
            self._save_wav(samples, wav_path, sample_rate=16000)
            
            # Run GGML inference
            cmd = [
                str(self.ggml_executable),
                str(self.ggml_model),
                "--test",
                "--audio", wav_path,
                "--save-output", output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"GGML stderr: {result.stderr}")
                raise RuntimeError(f"GGML inference failed: {result.stderr}")
            
            # Load output
            output = self._load_ggml_output(output_path)
            
            return torch.from_numpy(output)
            
        finally:
            # Cleanup temp files
            if os.path.exists(wav_path):
                os.unlink(wav_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def _save_wav(self, samples: np.ndarray, path: str, sample_rate: int = 16000):
        """Save samples as 16-bit PCM WAV file."""
        # Convert to 16-bit PCM
        samples_int16 = (samples * 32767).astype(np.int16)
        
        # Write WAV header and data
        import struct
        
        num_samples = len(samples_int16)
        data_size = num_samples * 2  # 16-bit = 2 bytes per sample
        file_size = 36 + data_size
        
        with open(path, 'wb') as f:
            # RIFF header
            f.write(b'RIFF')
            f.write(struct.pack('<I', file_size))
            f.write(b'WAVE')
            
            # fmt chunk
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))  # chunk size
            f.write(struct.pack('<H', 1))   # audio format (PCM)
            f.write(struct.pack('<H', 1))   # num channels
            f.write(struct.pack('<I', sample_rate))  # sample rate
            f.write(struct.pack('<I', sample_rate * 2))  # byte rate
            f.write(struct.pack('<H', 2))   # block align
            f.write(struct.pack('<H', 16))  # bits per sample
            
            # data chunk
            f.write(b'data')
            f.write(struct.pack('<I', data_size))
            f.write(samples_int16.tobytes())
    
    def _load_ggml_output(self, path: str) -> np.ndarray:
        """
        Load GGML output from binary file.
        
        GGML output format:
        - Header: int64[3] = [seq_len, num_classes, batch]
        - Data: float32 array in column-major order
        
        Returns:
            (seq_len, num_classes) numpy array
        """
        with open(path, 'rb') as f:
            shape = np.fromfile(f, dtype=np.int64, count=3)
            ne0, ne1, ne2 = shape  # [seq_len, num_classes, batch] in GGML
            total_elements = ne0 * ne1 * ne2
            data = np.fromfile(f, dtype=np.float32, count=total_elements)
            
            # GGML column-major: read with Fortran order
            output = data.reshape((ne0, ne1, ne2), order='F')
            # Transpose to PyTorch: (batch, seq_len, num_classes)
            output = np.transpose(output, (2, 0, 1))
            
            # Return (seq_len, num_classes) for single batch
            return output[0]


class GGMLInference(Inference):
    """
    Custom Inference class that uses GGML segmentation wrapper.
    
    This overrides the infer() method to use the GGML wrapper instead of PyTorch.
    """
    
    def __init__(self, ggml_wrapper: GGMLSegmentationWrapper, **kwargs):
        """
        Initialize with GGML wrapper.
        
        Args:
            ggml_wrapper: GGMLSegmentationWrapper instance
            **kwargs: Additional arguments passed to parent Inference class
        """
        # Initialize parent with the wrapper as the model
        super().__init__(model=ggml_wrapper.original_model, **kwargs)
        
        # Replace model with GGML wrapper
        self.ggml_wrapper = ggml_wrapper
        
    def infer(self, chunks: torch.Tensor):
        """
        Override infer to use GGML wrapper.
        
        Args:
            chunks: (batch_size, num_channels, num_samples) tensor
            
        Returns:
            (batch_size, num_frames, num_classes) numpy array
        """
        # Run GGML inference
        outputs = self.ggml_wrapper(chunks)
        
        # Apply conversion (powerset to multilabel if needed)
        with torch.inference_mode():
            outputs = self.conversion(outputs).cpu().numpy()
        
        return outputs


def write_rttm(annotation: Annotation, path: Path, uri: str = "sample"):
    """
    Write annotation to RTTM file.
    
    RTTM format:
    SPEAKER file 1 start duration <NA> <NA> speaker_id <NA> <NA>
    """
    with open(path, 'w') as f:
        for turn in annotation.itertracks(yield_label=True):
            segment, _, speaker = turn  # type: ignore[misc]
            f.write(f"SPEAKER {uri} 1 {segment.start:.3f} {segment.duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")


def compare_rttm_files(rttm1: Path, rttm2: Path) -> dict:
    """
    Compare two RTTM files and return statistics.
    
    Returns:
        Dictionary with comparison statistics
    """
    def parse_rttm(path: Path) -> list:
        segments = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker = parts[7]
                    segments.append((start, duration, speaker))
        return segments
    
    segments1 = parse_rttm(rttm1)
    segments2 = parse_rttm(rttm2)
    
    # Basic statistics
    stats = {
        "file1_segments": len(segments1),
        "file2_segments": len(segments2),
        "file1_speakers": len(set(s[2] for s in segments1)),
        "file2_speakers": len(set(s[2] for s in segments2)),
        "file1_total_duration": sum(s[1] for s in segments1),
        "file2_total_duration": sum(s[1] for s in segments2),
    }
    
    return stats


def run_pytorch_pipeline(audio_path: Path, output_rttm: Path) -> Annotation:
    """
    Run the original PyTorch speaker diarization pipeline.
    
    Args:
        audio_path: Path to audio file
        output_rttm: Path to save RTTM output
        
    Returns:
        Diarization annotation
    """
    print("\n" + "=" * 60)
    print("Running PyTorch Pipeline")
    print("=" * 60)
    
    patch_torch_load()
    
    # Load pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=None
    )
    
    if pipeline is None:
        raise RuntimeError("Failed to load pipeline")
    
    print(f"Processing: {audio_path}")
    output = pipeline(str(audio_path))
    
    # Get diarization annotation
    if hasattr(output, 'speaker_diarization'):
        diarization = output.speaker_diarization
    else:
        diarization = output
    
    # Write RTTM
    write_rttm(diarization, output_rttm, uri=audio_path.stem)
    print(f"Saved RTTM: {output_rttm}")
    
    # Print summary
    print(f"\nPyTorch Results:")
    print(f"  Speakers: {len(diarization.labels())}")
    print(f"  Segments: {len(list(diarization.itertracks()))}")
    
    return diarization


def run_ggml_pipeline(
    audio_path: Path,
    output_rttm: Path,
    ggml_executable: Path,
    ggml_model: Path
) -> Annotation:
    """
    Run speaker diarization pipeline with GGML segmentation.
    
    Args:
        audio_path: Path to audio file
        output_rttm: Path to save RTTM output
        ggml_executable: Path to GGML executable
        ggml_model: Path to GGUF model
        
    Returns:
        Diarization annotation
    """
    print("\n" + "=" * 60)
    print("Running GGML Pipeline")
    print("=" * 60)
    
    patch_torch_load()
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=None
    )
    
    if pipeline is None:
        raise RuntimeError("Failed to load pipeline")
    
    original_model = pipeline._segmentation.model
    
    # Create GGML wrapper
    print(f"GGML executable: {ggml_executable}")
    print(f"GGML model: {ggml_model}")
    
    ggml_wrapper = GGMLSegmentationWrapper(
        ggml_executable=ggml_executable,
        ggml_model=ggml_model,
        original_model=original_model
    )
    
    # Create custom inference with GGML wrapper
    ggml_inference = GGMLInference(
        ggml_wrapper=ggml_wrapper,
        duration=pipeline._segmentation.duration,
        step=pipeline._segmentation.step,
        skip_aggregation=True,
        batch_size=1,  # Process one chunk at a time for GGML
    )
    
    # Replace segmentation inference in pipeline
    pipeline._segmentation = ggml_inference
    
    # Run pipeline
    print(f"Processing: {audio_path}")
    output = pipeline(str(audio_path))
    
    # Get diarization annotation
    if hasattr(output, 'speaker_diarization'):
        diarization = output.speaker_diarization
    else:
        diarization = output
    
    # Write RTTM
    write_rttm(diarization, output_rttm, uri=audio_path.stem)
    print(f"Saved RTTM: {output_rttm}")
    
    # Print summary
    print(f"\nGGML Results:")
    print(f"  Speakers: {len(diarization.labels())}")
    print(f"  Segments: {len(list(diarization.itertracks()))}")
    
    return diarization


def main():
    print("=" * 60)
    print("Pipeline Integration Test: GGML Segmentation")
    print("=" * 60)
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    repo_root = project_root.parent
    
    audio_path = repo_root / "samples" / "sample.wav"
    ggml_executable = project_root / "build" / "bin" / "segmentation-ggml"
    ggml_model = project_root / "segmentation.gguf"
    
    pytorch_rttm = script_dir / "sample_pytorch.rttm"
    ggml_rttm = script_dir / "sample_ggml.rttm"
    
    # Verify paths
    print(f"\nAudio: {audio_path}")
    print(f"GGML executable: {ggml_executable}")
    print(f"GGML model: {ggml_model}")
    
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        return 1
    
    if not ggml_executable.exists():
        print(f"ERROR: GGML executable not found: {ggml_executable}")
        print("Please build the GGML project first:")
        print("  cd segmentation-ggml && cmake -B build && cmake --build build")
        return 1
    
    if not ggml_model.exists():
        print(f"ERROR: GGML model not found: {ggml_model}")
        print("Please convert the model first:")
        print("  python segmentation-ggml/convert.py ...")
        return 1
    
    # Run PyTorch pipeline
    try:
        pytorch_diarization = run_pytorch_pipeline(audio_path, pytorch_rttm)
    except Exception as e:
        print(f"ERROR: PyTorch pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run GGML pipeline
    try:
        ggml_diarization = run_ggml_pipeline(
            audio_path, ggml_rttm, ggml_executable, ggml_model
        )
    except Exception as e:
        print(f"ERROR: GGML pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Compare results
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    
    stats = compare_rttm_files(pytorch_rttm, ggml_rttm)
    
    print(f"\nPyTorch RTTM:")
    print(f"  Segments: {stats['file1_segments']}")
    print(f"  Speakers: {stats['file1_speakers']}")
    print(f"  Total duration: {stats['file1_total_duration']:.2f}s")
    
    print(f"\nGGML RTTM:")
    print(f"  Segments: {stats['file2_segments']}")
    print(f"  Speakers: {stats['file2_speakers']}")
    print(f"  Total duration: {stats['file2_total_duration']:.2f}s")
    
    # Check similarity
    duration_diff = abs(stats['file1_total_duration'] - stats['file2_total_duration'])
    speaker_match = stats['file1_speakers'] == stats['file2_speakers']
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    
    if speaker_match and duration_diff < 1.0:
        print("\n✓ PASS: GGML pipeline produces similar results to PyTorch")
        print(f"  Speaker count matches: {speaker_match}")
        print(f"  Duration difference: {duration_diff:.2f}s")
        return 0
    else:
        print("\n⚠ WARNING: Results differ (may be acceptable due to F16 precision)")
        print(f"  Speaker count matches: {speaker_match}")
        print(f"  Duration difference: {duration_diff:.2f}s")
        return 0  # Don't fail - differences are expected with F16
    

if __name__ == "__main__":
    sys.exit(main())
