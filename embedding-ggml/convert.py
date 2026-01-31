#!/usr/bin/env python3
"""Convert WeSpeaker ResNet34 embedding model to GGUF format.

Downloads from pyannote/speaker-diarization-community-1 and converts
the embedding model (ResNet34 + TSTP pooling) to GGUF.

Tensor naming: keeps PyTorch names as-is (minus num_batches_tracked).
Dtype: weights -> F16, biases/BN params/running stats -> F32.
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8


def write_gguf_string(f, s: str):
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def write_gguf_metadata_kv(f, key: str, value_type: int, value):
    write_gguf_string(f, key)
    f.write(struct.pack('<I', value_type))

    if value_type == GGUF_TYPE_UINT32:
        f.write(struct.pack('<I', value))
    elif value_type == GGUF_TYPE_INT32:
        f.write(struct.pack('<i', value))
    elif value_type == GGUF_TYPE_FLOAT32:
        f.write(struct.pack('<f', value))
    elif value_type == GGUF_TYPE_STRING:
        write_gguf_string(f, value)
    else:
        raise ValueError(f"Unsupported metadata type: {value_type}")


def align_offset(offset: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    return offset + (alignment - (offset % alignment)) % alignment


def should_use_f16(name: str, data: np.ndarray) -> bool:
    """Weights (multi-dim, not BN/bias) -> F16, everything else -> F32."""
    if 'bias' in name:
        return False
    if 'running_mean' in name or 'running_var' in name:
        return False
    if 'bn' in name or 'shortcut.1' in name or 'shortcut.3' in name:
        # BN weight/bias are 1D scale/shift params -> F32
        if len(data.shape) == 1:
            return False
    if len(data.shape) < 2:
        return False
    return True


def convert_embedding_to_gguf(output_path: str, verbose: bool = True):
    from huggingface_hub import hf_hub_download

    print("Downloading embedding model from pyannote/speaker-diarization-community-1...")
    emb_path = hf_hub_download(
        'pyannote/speaker-diarization-community-1',
        'embedding/pytorch_model.bin'
    )
    print(f"  Downloaded: {emb_path}")

    print("Loading PyTorch checkpoint...")
    checkpoint = torch.load(emb_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']

    print(f"  Found {len(state_dict)} tensors in checkpoint")

    converted_tensors = {}
    skipped = []

    for name, tensor in state_dict.items():
        if 'num_batches_tracked' in name:
            skipped.append(name)
            continue

        data = tensor.detach().cpu().numpy()
        converted_tensors[name] = data

        if verbose:
            use_f16 = should_use_f16(name, data)
            dtype_str = "F16" if use_f16 else "F32"
            print(f"  {name:55s} {str(data.shape):20s} -> {dtype_str}")

    print(f"\n  Converted: {len(converted_tensors)} tensors")
    print(f"  Skipped (num_batches_tracked): {len(skipped)}")

    metadata = {
        'general.architecture': (GGUF_TYPE_STRING, 'wespeaker_resnet34'),
        'general.name': (GGUF_TYPE_STRING, 'pyannote-embedding-community-1'),
        'general.alignment': (GGUF_TYPE_UINT32, GGUF_DEFAULT_ALIGNMENT),
        'wespeaker.sample_rate': (GGUF_TYPE_UINT32, 16000),
        'wespeaker.num_mel_bins': (GGUF_TYPE_UINT32, 80),
        'wespeaker.frame_length': (GGUF_TYPE_UINT32, 25),
        'wespeaker.frame_shift': (GGUF_TYPE_UINT32, 10),
        'wespeaker.embed_dim': (GGUF_TYPE_UINT32, 256),
        'wespeaker.feat_dim': (GGUF_TYPE_UINT32, 80),
    }

    print(f"\nWriting GGUF file to: {output_path}")

    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(converted_tensors)))
        f.write(struct.pack('<Q', len(metadata)))

        # Metadata KV
        for key, (value_type, value) in metadata.items():
            write_gguf_metadata_kv(f, key, value_type, value)

        # Tensor info
        tensor_infos = []
        current_offset = 0

        for name, data in converted_tensors.items():
            use_f16 = should_use_f16(name, data)
            ftype = GGML_TYPE_F16 if use_f16 else GGML_TYPE_F32

            if ftype == GGML_TYPE_F16:
                data_converted = data.astype(np.float16)
            else:
                data_converted = data.astype(np.float32)

            tensor_infos.append({
                'name': name,
                'data': data_converted,
                'ftype': ftype,
                'offset': current_offset,
            })

            padded_size = align_offset(data_converted.nbytes)
            current_offset += padded_size

        for info in tensor_infos:
            name = info['name']
            data = info['data']
            ftype = info['ftype']
            offset = info['offset']

            write_gguf_string(f, name)

            n_dims = len(data.shape)
            f.write(struct.pack('<I', n_dims))

            # GGML dimensions are reversed from numpy/PyTorch
            for i in range(n_dims):
                f.write(struct.pack('<Q', data.shape[n_dims - 1 - i]))

            f.write(struct.pack('<I', ftype))
            f.write(struct.pack('<Q', offset))

        # Padding to alignment
        current_pos = f.tell()
        aligned_pos = align_offset(current_pos)
        padding_needed = aligned_pos - current_pos
        if padding_needed > 0:
            f.write(b'\x00' * padding_needed)

        # Tensor data
        tensor_data_start = f.tell()

        for info in tensor_infos:
            data = info['data']
            offset = info['offset']

            target_pos = tensor_data_start + offset
            current_pos = f.tell()
            if target_pos > current_pos:
                f.write(b'\x00' * (target_pos - current_pos))

            data.tofile(f)

            padded_size = align_offset(data.nbytes)
            padding_needed = padded_size - data.nbytes
            if padding_needed > 0:
                f.write(b'\x00' * padding_needed)

        file_size = f.tell()

    # Summary
    print(f"\n{'='*60}")
    print("GGUF Conversion Summary")
    print(f"{'='*60}")
    print(f"Output file: {output_path}")
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")
    print(f"Tensors: {len(converted_tensors)}")
    print(f"Metadata keys: {len(metadata)}")

    f16_count = sum(1 for i in tensor_infos if i['ftype'] == GGML_TYPE_F16)
    f32_count = sum(1 for i in tensor_infos if i['ftype'] == GGML_TYPE_F32)
    print(f"F16 tensors: {f16_count}")
    print(f"F32 tensors: {f32_count}")
    print(f"\nConversion complete!")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert WeSpeaker ResNet34 embedding model to GGUF format'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='embedding.gguf',
        help='Output GGUF file path (default: embedding.gguf)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Print detailed conversion info'
    )

    args = parser.parse_args()

    success = convert_embedding_to_gguf(args.output, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
