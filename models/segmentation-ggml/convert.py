#!/usr/bin/env python3
"""
Convert PyAnnote segmentation model from PyTorch to GGUF format.

This script converts the PyAnnote PyanNet segmentation model to GGUF format
for inference with GGML. It handles:
- SincNet filter pre-computation (parametric → static conv weights)
- Tensor name mapping (PyTorch → GGUF conventions)
- GGUF file writing with proper alignment
- Dtype conversion (weights → F16, biases → F32)

Usage:
    python convert.py --model-path <path-to-pytorch_model.bin> --output segmentation.gguf
"""

import argparse
import struct
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGML type constants
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

# GGUF metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


def register_safe_globals():
    """Register safe globals for PyTorch 2.6+ compatibility."""
    try:
        from pyannote.audio.core import task
        import inspect
        task_classes = [obj for name, obj in inspect.getmembers(task) if inspect.isclass(obj)]
        torch.serialization.add_safe_globals(task_classes)
        print(f"Registered {len(task_classes)} safe globals for PyTorch serialization")
    except ImportError:
        print("Warning: Could not import pyannote.audio.core.task")
        print("  If loading fails, install pyannote.audio or use an older PyTorch version")
    except AttributeError:
        # Older PyTorch versions don't have add_safe_globals
        pass


def compute_sincnet_filters_from_model(model_path: str) -> np.ndarray:
    """
    Extract pre-computed SincNet filters directly from PyTorch model.
    
    This uses the actual ParamSincFB.filters() method to ensure exact match
    with PyTorch inference.
    
    Args:
        model_path: Path to PyTorch model checkpoint
        
    Returns:
        Filters as numpy array, shape (80, 1, 251)
    """
    try:
        from pyannote.audio import Model
        
        # Patch torch.load for PyTorch 2.6+ compatibility
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        
        # Load the model
        model = Model.from_pretrained("pyannote/segmentation-3.0")
        
        # Get the filterbank and compute filters
        filterbank = model.sincnet.conv1d[0].filterbank
        with torch.no_grad():
            filters = filterbank.filters()
        
        # Convert to numpy: shape (80, 1, 251)
        filters_np = filters.cpu().numpy().astype(np.float32)
        print(f"  Extracted SincNet filters from PyTorch model: {filters_np.shape}")
        
        return filters_np
        
    except ImportError:
        print("Warning: pyannote.audio not available, cannot extract SincNet filters")
        return None
    except Exception as e:
        print(f"Warning: Failed to extract SincNet filters: {e}")
        return None


def write_gguf_string(f, s: str):
    """Write a GGUF string (length-prefixed UTF-8)."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))  # uint64 length
    f.write(encoded)


def write_gguf_metadata_value(f, value_type: int, value: Any):
    """Write a GGUF metadata value."""
    f.write(struct.pack('<I', value_type))  # uint32 type
    
    if value_type == GGUF_TYPE_UINT32:
        f.write(struct.pack('<I', value))
    elif value_type == GGUF_TYPE_INT32:
        f.write(struct.pack('<i', value))
    elif value_type == GGUF_TYPE_FLOAT32:
        f.write(struct.pack('<f', value))
    elif value_type == GGUF_TYPE_STRING:
        write_gguf_string(f, value)
    elif value_type == GGUF_TYPE_BOOL:
        f.write(struct.pack('B', 1 if value else 0))
    else:
        raise ValueError(f"Unsupported metadata type: {value_type}")


def write_gguf_metadata_kv(f, key: str, value_type: int, value: Any):
    """Write a GGUF metadata key-value pair."""
    write_gguf_string(f, key)
    write_gguf_metadata_value(f, value_type, value)


def align_offset(offset: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    """Calculate aligned offset."""
    return offset + (alignment - (offset % alignment)) % alignment


def convert_pytorch_to_gguf(model_path: str, output_path: str, verbose: bool = True):
    """
    Convert PyAnnote segmentation model from PyTorch to GGUF format.
    
    Args:
        model_path: Path to PyTorch model checkpoint (pytorch_model.bin)
        output_path: Path for output GGUF file
        verbose: Print detailed conversion info
    """
    register_safe_globals()
    
    # Load PyTorch checkpoint
    print(f"Loading PyTorch model from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying with weights_only=True...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
    
    if verbose:
        print(f"Found {len(state_dict)} tensors in checkpoint")
    
    # Prepare tensor mapping and conversion
    # PyTorch name → (GGUF name, should_convert_to_f16)
    tensor_map: Dict[str, tuple] = {}
    converted_tensors: Dict[str, np.ndarray] = {}
    
    # Extract SincNet parameters for filter pre-computation
    sincnet_low_hz = None
    sincnet_band_hz = None
    
    for name, tensor in state_dict.items():
        # Look for SincNet learnable parameters
        if 'sincnet.conv1d.0.filterbank.low_hz_' in name or 'conv1d.0.filterbank.low_hz_' in name:
            sincnet_low_hz = tensor.detach().cpu().numpy().flatten()
            if verbose:
                print(f"Found SincNet low_hz: shape={sincnet_low_hz.shape}, range=[{sincnet_low_hz.min():.2f}, {sincnet_low_hz.max():.2f}] Hz")
        elif 'sincnet.conv1d.0.filterbank.band_hz_' in name or 'conv1d.0.filterbank.band_hz_' in name:
            sincnet_band_hz = tensor.detach().cpu().numpy().flatten()
            if verbose:
                print(f"Found SincNet band_hz: shape={sincnet_band_hz.shape}, range=[{sincnet_band_hz.min():.2f}, {sincnet_band_hz.max():.2f}] Hz")
    
    # Extract SincNet filters directly from PyTorch model
    print("Extracting SincNet filters from PyTorch model...")
    sincnet_filters = compute_sincnet_filters_from_model(model_path)
    if sincnet_filters is not None:
        converted_tensors['sincnet.0.conv.weight'] = sincnet_filters
        if verbose:
            print(f"  SincNet filters shape: {sincnet_filters.shape}")
            print(f"  SincNet filters dtype: {sincnet_filters.dtype}")
    else:
        print("Warning: Could not extract SincNet filters from model")
    
    # Map remaining tensors
    for name, tensor in state_dict.items():
        data = tensor.detach().cpu().numpy()
        
        # Skip SincNet learnable parameters (already pre-computed)
        if 'filterbank.low_hz_' in name or 'filterbank.band_hz_' in name:
            continue
        
        # Map tensor names from PyTorch to GGUF convention
        gguf_name = None
        
        # SincNet normalization layers
        if 'sincnet.wav_norm1d.weight' in name:
            gguf_name = 'sincnet.wav_norm.weight'
        elif 'sincnet.wav_norm1d.bias' in name:
            gguf_name = 'sincnet.wav_norm.bias'
        
        # SincNet conv and norm layers (stages 1, 2, 3 → indices 0, 1, 2)
        # Note: sincnet.conv1d.0 is ParamSincFB (handled separately)
        elif 'sincnet.conv1d.1.weight' in name:
            gguf_name = 'sincnet.1.conv.weight'
        elif 'sincnet.conv1d.1.bias' in name:
            gguf_name = 'sincnet.1.conv.bias'
        elif 'sincnet.conv1d.2.weight' in name:
            gguf_name = 'sincnet.2.conv.weight'
        elif 'sincnet.conv1d.2.bias' in name:
            gguf_name = 'sincnet.2.conv.bias'
        
        # SincNet normalization layers
        elif 'sincnet.norm1d.0.weight' in name:
            gguf_name = 'sincnet.0.norm.weight'
        elif 'sincnet.norm1d.0.bias' in name:
            gguf_name = 'sincnet.0.norm.bias'
        elif 'sincnet.norm1d.1.weight' in name:
            gguf_name = 'sincnet.1.norm.weight'
        elif 'sincnet.norm1d.1.bias' in name:
            gguf_name = 'sincnet.1.norm.bias'
        elif 'sincnet.norm1d.2.weight' in name:
            gguf_name = 'sincnet.2.norm.weight'
        elif 'sincnet.norm1d.2.bias' in name:
            gguf_name = 'sincnet.2.norm.bias'
        
        # LSTM layers (bidirectional, 4 layers)
        # PyTorch LSTM names: lstm.weight_ih_l{layer}, lstm.weight_hh_l{layer}
        #                    lstm.weight_ih_l{layer}_reverse, etc.
        elif 'lstm.weight_ih_l' in name:
            # Keep the same name for LSTM weights
            gguf_name = name
        elif 'lstm.weight_hh_l' in name:
            gguf_name = name
        elif 'lstm.bias_ih_l' in name:
            gguf_name = name
        elif 'lstm.bias_hh_l' in name:
            gguf_name = name
        
        # Linear layers
        elif 'linear.0.weight' in name:
            gguf_name = 'linear.0.weight'
        elif 'linear.0.bias' in name:
            gguf_name = 'linear.0.bias'
        elif 'linear.1.weight' in name:
            gguf_name = 'linear.1.weight'
        elif 'linear.1.bias' in name:
            gguf_name = 'linear.1.bias'
        
        # Classifier
        elif 'classifier.weight' in name:
            gguf_name = 'classifier.weight'
        elif 'classifier.bias' in name:
            gguf_name = 'classifier.bias'
        
        if gguf_name:
            converted_tensors[gguf_name] = data
            if verbose:
                print(f"  {name} -> {gguf_name}: {data.shape}")
    
    # Determine dtype conversion for each tensor
    # Convention: weights → F16, biases → F32, LSTM states → F32
    tensor_ftypes = {}
    for name, data in converted_tensors.items():
        if 'bias' in name or len(data.shape) == 1:
            # Biases stay as F32 for numerical stability
            tensor_ftypes[name] = GGML_TYPE_F32
        else:
            # Weights convert to F16 for memory efficiency
            tensor_ftypes[name] = GGML_TYPE_F16
    
    # Prepare GGUF file
    print(f"\nWriting GGUF file to: {output_path}")
    
    # Metadata to write
    metadata = {
        'general.architecture': (GGUF_TYPE_STRING, 'pyannet'),
        'general.name': (GGUF_TYPE_STRING, 'pyannote-segmentation-3.0'),
        'general.alignment': (GGUF_TYPE_UINT32, GGUF_DEFAULT_ALIGNMENT),
        'pyannet.sample_rate': (GGUF_TYPE_UINT32, 16000),
        'pyannet.num_classes': (GGUF_TYPE_UINT32, 7),
        'pyannet.lstm_layers': (GGUF_TYPE_UINT32, 4),
        'pyannet.lstm_hidden': (GGUF_TYPE_UINT32, 128),
        'pyannet.sincnet_kernel_size': (GGUF_TYPE_UINT32, 251),
        'pyannet.sincnet_stride': (GGUF_TYPE_UINT32, 10),
    }
    
    with open(output_path, 'wb') as f:
        # ========== HEADER ==========
        # Magic number
        f.write(struct.pack('<I', GGUF_MAGIC))
        # Version
        f.write(struct.pack('<I', GGUF_VERSION))
        # Tensor count
        f.write(struct.pack('<Q', len(converted_tensors)))
        # Metadata KV count
        f.write(struct.pack('<Q', len(metadata)))
        
        # ========== METADATA KV ==========
        for key, (value_type, value) in metadata.items():
            write_gguf_metadata_kv(f, key, value_type, value)
        
        # ========== TENSOR INFO ==========
        tensor_infos = []
        current_offset = 0
        
        for name, data in converted_tensors.items():
            ftype = tensor_ftypes[name]
            
            # Convert data to appropriate dtype
            if ftype == GGML_TYPE_F16:
                data_converted = data.astype(np.float16)
            else:
                data_converted = data.astype(np.float32)
            
            # Store tensor info - offset should be current_offset (already aligned from previous iteration)
            tensor_infos.append({
                'name': name,
                'data': data_converted,
                'ftype': ftype,
                'offset': current_offset,
            })
            
            # Update offset for next tensor - use PADDED size (GGML expects this)
            # GGML_PAD(x, n) = (x + n - 1) & ~(n - 1)
            padded_size = align_offset(data_converted.nbytes)
            current_offset += padded_size
        
        # Write tensor info
        for info in tensor_infos:
            name = info['name']
            data = info['data']
            ftype = info['ftype']
            offset = info['offset']
            
            # Name (GGUF string)
            write_gguf_string(f, name)
            
            # Number of dimensions
            n_dims = len(data.shape)
            f.write(struct.pack('<I', n_dims))
            
            # Dimensions (in GGML order: reversed from numpy/PyTorch)
            # GGML uses row-major with dimensions in reverse order
            for i in range(n_dims):
                f.write(struct.pack('<Q', data.shape[n_dims - 1 - i]))
            
            # Type
            f.write(struct.pack('<I', ftype))
            
            # Offset (relative to tensor_data start)
            f.write(struct.pack('<Q', offset))
        
        # ========== PADDING TO ALIGNMENT ==========
        current_pos = f.tell()
        aligned_pos = align_offset(current_pos)
        padding_needed = aligned_pos - current_pos
        if padding_needed > 0:
            f.write(b'\x00' * padding_needed)
        
        # ========== TENSOR DATA ==========
        tensor_data_start = f.tell()
        
        for i, info in enumerate(tensor_infos):
            data = info['data']
            offset = info['offset']
            
            # Seek to aligned position (relative to tensor_data_start)
            target_pos = tensor_data_start + offset
            current_pos = f.tell()
            if target_pos > current_pos:
                f.write(b'\x00' * (target_pos - current_pos))
            
            # Write tensor data
            data.tofile(f)
            
            # Pad to alignment (GGML expects all tensors to be padded)
            current_pos = f.tell()
            padded_size = align_offset(data.nbytes)
            padding_needed = padded_size - data.nbytes
            if padding_needed > 0:
                f.write(b'\x00' * padding_needed)
        
        file_size = f.tell()
    
    # Print summary
    print(f"\n{'='*60}")
    print("GGUF Conversion Summary")
    print(f"{'='*60}")
    print(f"Output file: {output_path}")
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")
    print(f"Tensors: {len(converted_tensors)}")
    print(f"Metadata keys: {len(metadata)}")
    print()
    
    # Print tensor summary
    print("Tensors written:")
    f16_count = 0
    f32_count = 0
    for info in tensor_infos:
        ftype_str = "F16" if info['ftype'] == GGML_TYPE_F16 else "F32"
        if info['ftype'] == GGML_TYPE_F16:
            f16_count += 1
        else:
            f32_count += 1
        print(f"  {info['name']:40s} {str(info['data'].shape):20s} {ftype_str}")
    
    print(f"\nF16 tensors: {f16_count}")
    print(f"F32 tensors: {f32_count}")
    print(f"\nConversion complete!")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyAnnote segmentation model from PyTorch to GGUF format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert from HuggingFace cache
    python convert.py --model-path ~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots/*/segmentation/pytorch_model.bin --output segmentation.gguf
    
    # Convert from local file
    python convert.py --model-path ./pytorch_model.bin --output segmentation.gguf
"""
    )
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        required=True,
        help='Path to PyTorch model checkpoint (pytorch_model.bin)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='segmentation.gguf',
        help='Output GGUF file path (default: segmentation.gguf)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed conversion info'
    )
    
    args = parser.parse_args()
    
    # Validate input path
    model_path = Path(args.model_path)
    if not model_path.exists():
        # Try to expand glob patterns
        import glob
        matches = glob.glob(str(model_path))
        if matches:
            model_path = Path(matches[0])
            print(f"Using model path: {model_path}")
        else:
            print(f"Error: Model file not found: {args.model_path}")
            sys.exit(1)
    
    # Convert
    success = convert_pytorch_to_gguf(
        str(model_path),
        args.output,
        verbose=args.verbose or True  # Always verbose for now
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
