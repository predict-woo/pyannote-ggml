# PyAnnote Segmentation Model Architecture

## Overview

**Model Type**: PyanNet  
**Total Parameters**: 1,473,265  
**Input**: Raw audio waveform (16kHz, mono)  
**Output**: Speaker segmentation probabilities (7 classes)

The PyanNet model is a neural network for speaker segmentation that combines:
1. **SincNet** - Learnable audio feature extraction
2. **LSTM** - Temporal modeling with bidirectional layers
3. **Linear** - Feed-forward layers with LeakyReLU activation
4. **Classifier** - Final classification layer with LogSoftmax

## Architecture Pipeline

```
Input Audio (16kHz, mono)
    ↓
[SincNet Feature Extraction]
    ↓
[LSTM Temporal Modeling]
    ↓
[Linear Feed-Forward Layers]
    ↓
[Classifier + LogSoftmax]
    ↓
Output Probabilities (7 classes per frame)
```

---

## Detailed Layer-by-Layer Breakdown

### 1. SincNet Block (42,602 parameters)

SincNet is a learnable audio feature extractor that uses parameterized sinc functions as convolutional filters. It consists of 3 convolutional stages, each followed by max pooling and instance normalization.

#### 1.1 Input Normalization
- **Layer**: `wav_norm1d` (InstanceNorm1d)
- **Input Shape**: `(batch, 1, samples)`
- **Output Shape**: `(batch, 1, samples)`
- **Parameters**: 2 (affine: gamma, beta)
- **Operation**: Normalize input waveform per instance

#### 1.2 SincNet Stage 1: Parameterized Sinc Convolution

**Conv Layer**: `sincnet.conv1d.0` (ParamSincFB Encoder)
- **Input Shape**: `(batch, 1, samples)`
- **Output Shape**: `(batch, 80, frames)`
- **Parameters**: 80 (40 filters × 2 parameters: low_hz, band_hz)
- **Operation**: Learnable bandpass filters using sinc functions
- **Configuration**:
  - `n_filters`: 80 (40 filters, each producing 2 outputs: positive and negative phase)
  - `kernel_size`: 251 samples
  - `stride`: 10 samples
  - `sample_rate`: 16000 Hz
  - `min_low_hz`: 50 Hz
  - `min_band_hz`: 50 Hz

**Learned Filter Parameters**:
- `low_hz`: Center frequencies ranging from 36.94 Hz to 7434.21 Hz (40 values)
- `band_hz`: Bandwidths ranging from 38.60 Hz to 529.37 Hz (40 values)

**Special Operation**: `abs()` applied after convolution (only in stage 1)
- This creates a rectified representation of the filtered signal

**Example Tensor Flow**:
```
Input:  (1, 1, 160000)  # 10 seconds at 16kHz
   ↓ ParamSincFB conv (kernel=251, stride=10)
Conv:   (1, 80, 15975)  # 80 feature channels
   ↓ abs()
   ↓ MaxPool1d (kernel=3, stride=3)
Pool:   (1, 80, 5325)
   ↓ InstanceNorm1d + LeakyReLU
Output: (1, 80, 5325)
```

**Pool Layer**: `sincnet.pool1d.0` (MaxPool1d)
- **Kernel Size**: 3
- **Stride**: 3
- **Padding**: 0
- **Dilation**: 1

**Norm Layer**: `sincnet.norm1d.0` (InstanceNorm1d)
- **Num Features**: 80
- **Parameters**: 160 (affine: 80 gamma + 80 beta)
- **Activation**: LeakyReLU (applied after normalization)

#### 1.3 SincNet Stage 2: Standard Convolution

**Conv Layer**: `sincnet.conv1d.1` (Conv1d)
- **Input Shape**: `(batch, 80, 5325)`
- **Output Shape**: `(batch, 60, 5321)`
- **Parameters**: 24,060 (80×60×5 weights + 60 biases)
- **Configuration**:
  - `in_channels`: 80
  - `out_channels`: 60
  - `kernel_size`: 5
  - `stride`: 1
  - `padding`: 0
  - `dilation`: 1

**Example Tensor Flow**:
```
Input:  (1, 80, 5325)
   ↓ Conv1d (80→60, kernel=5, stride=1)
Conv:   (1, 60, 5321)
   ↓ MaxPool1d (kernel=3, stride=3)
Pool:   (1, 60, 1773)
   ↓ InstanceNorm1d + LeakyReLU
Output: (1, 60, 1773)
```

**Pool Layer**: `sincnet.pool1d.1` (MaxPool1d)
- **Kernel Size**: 3
- **Stride**: 3
- **Padding**: 0
- **Dilation**: 1

**Norm Layer**: `sincnet.norm1d.1` (InstanceNorm1d)
- **Num Features**: 60
- **Parameters**: 120 (affine: 60 gamma + 60 beta)
- **Activation**: LeakyReLU (applied after normalization)

#### 1.4 SincNet Stage 3: Standard Convolution

**Conv Layer**: `sincnet.conv1d.2` (Conv1d)
- **Input Shape**: `(batch, 60, 1773)`
- **Output Shape**: `(batch, 60, 1769)`
- **Parameters**: 18,060 (60×60×5 weights + 60 biases)
- **Configuration**:
  - `in_channels`: 60
  - `out_channels`: 60
  - `kernel_size`: 5
  - `stride`: 1
  - `padding`: 0
  - `dilation`: 1

**Example Tensor Flow**:
```
Input:  (1, 60, 1773)
   ↓ Conv1d (60→60, kernel=5, stride=1)
Conv:   (1, 60, 1769)
   ↓ MaxPool1d (kernel=3, stride=3)
Pool:   (1, 60, 589)
   ↓ InstanceNorm1d + LeakyReLU
Output: (1, 60, 589)
```

**Pool Layer**: `sincnet.pool1d.2` (MaxPool1d)
- **Kernel Size**: 3
- **Stride**: 3
- **Padding**: 0
- **Dilation**: 1

**Norm Layer**: `sincnet.norm1d.2` (InstanceNorm1d)
- **Num Features**: 60
- **Parameters**: 120 (affine: 60 gamma + 60 beta)
- **Activation**: LeakyReLU (applied after normalization)

**SincNet Output Shape**: `(batch, 60, 589)` for 10-second input

---

### 2. Reshape for LSTM

Before feeding into LSTM, the tensor is rearranged from `(batch, feature, frame)` to `(batch, frame, feature)` using `einops.rearrange()`.

```
Input:  (1, 60, 589)
   ↓ rearrange("batch feature frame -> batch frame feature")
Output: (1, 589, 60)
```

---

### 3. LSTM Block (1,380,352 parameters)

**Layer**: `lstm` (LSTM)
- **Input Shape**: `(batch, 589, 60)`
- **Output Shape**: `(batch, 589, 256)`
- **Parameters**: 1,380,352
- **Configuration**:
  - `input_size`: 60
  - `hidden_size`: 128
  - `num_layers`: 4
  - `bidirectional`: True (output size = 128 × 2 = 256)
  - `batch_first`: True
  - `dropout`: 0.5 (applied between layers)

**Operation**: 
- 4-layer bidirectional LSTM processes the temporal sequence
- Each layer has 128 hidden units
- Bidirectional: processes sequence forward and backward, concatenating outputs
- Dropout of 0.5 applied between layers for regularization

**Example Tensor Flow**:
```
Input:  (1, 589, 60)    # 589 frames, 60 features each
   ↓ LSTM (4 layers, hidden=128, bidirectional)
Output: (1, 589, 256)   # 589 frames, 256 features (128×2)
```

**Note**: The LSTM also outputs hidden states `(h_n, c_n)`, but these are not used in the forward pass.

---

### 4. Linear Block (49,408 parameters)

Two fully-connected layers with LeakyReLU activation between them.

#### 4.1 Linear Layer 1

**Layer**: `linear.0` (Linear)
- **Input Shape**: `(batch, 589, 256)`
- **Output Shape**: `(batch, 589, 128)`
- **Parameters**: 32,896 (256×128 weights + 128 biases)
- **Configuration**:
  - `in_features`: 256
  - `out_features`: 128
  - `bias`: True
- **Activation**: LeakyReLU

#### 4.2 Linear Layer 2

**Layer**: `linear.1` (Linear)
- **Input Shape**: `(batch, 589, 128)`
- **Output Shape**: `(batch, 589, 128)`
- **Parameters**: 16,512 (128×128 weights + 128 biases)
- **Configuration**:
  - `in_features`: 128
  - `out_features`: 128
  - `bias`: True
- **Activation**: LeakyReLU

**Example Tensor Flow**:
```
Input:  (1, 589, 256)
   ↓ Linear (256→128) + LeakyReLU
Hidden: (1, 589, 128)
   ↓ Linear (128→128) + LeakyReLU
Output: (1, 589, 128)
```

---

### 5. Classifier Block (903 parameters)

#### 5.1 Classifier Layer

**Layer**: `classifier` (Linear)
- **Input Shape**: `(batch, 589, 128)`
- **Output Shape**: `(batch, 589, 7)`
- **Parameters**: 903 (128×7 weights + 7 biases)
- **Configuration**:
  - `in_features`: 128
  - `out_features`: 7 (number of speaker classes)
  - `bias`: True

#### 5.2 Activation

**Layer**: `activation` (LogSoftmax)
- **Input Shape**: `(batch, 589, 7)`
- **Output Shape**: `(batch, 589, 7)`
- **Parameters**: 0
- **Configuration**: `dim=-1` (apply softmax over the class dimension)

**Example Tensor Flow**:
```
Input:  (1, 589, 128)
   ↓ Linear (128→7)
Logits: (1, 589, 7)
   ↓ LogSoftmax(dim=-1)
Output: (1, 589, 7)  # Log probabilities for 7 classes per frame
```

---

## Complete Forward Pass Example

For a 10-second audio input at 16kHz:

```
Input Audio:           (1, 1, 160000)    # Raw waveform

[SincNet Block]
  wav_norm1d:          (1, 1, 160000)    # Normalize
  sincnet_conv1:       (1, 80, 15975)    # ParamSincFB + abs()
  sincnet_stage1_out:  (1, 80, 5325)     # MaxPool + InstanceNorm + LeakyReLU
  sincnet_conv2:       (1, 60, 5321)     # Conv1d
  sincnet_stage2_out:  (1, 60, 1773)     # MaxPool + InstanceNorm + LeakyReLU
  sincnet_conv3:       (1, 60, 1769)     # Conv1d
  sincnet_stage3_out:  (1, 60, 589)      # MaxPool + InstanceNorm + LeakyReLU

[Reshape]
  rearrange:           (1, 589, 60)      # (batch, frame, feature)

[LSTM Block]
  lstm_out:            (1, 589, 256)     # 4-layer bidirectional LSTM

[Linear Block]
  linear_0:            (1, 589, 128)     # Linear + LeakyReLU
  linear_1:            (1, 589, 128)     # Linear + LeakyReLU

[Classifier Block]
  classifier_out:      (1, 589, 7)       # Linear
  output:              (1, 589, 7)       # LogSoftmax

Final Output:          (1, 589, 7)       # 589 frames, 7 class probabilities each
```

**Frame Rate**: 589 frames for 10 seconds = ~58.9 frames/second  
**Receptive Field**: Each output frame corresponds to a window of input audio

---

## GGML Operations Required

To implement this model in GGML, the following operations are needed:

### Core Operations
1. **ggml_norm** - Instance normalization (with affine parameters)
2. **ggml_conv_1d** - 1D convolution
3. **ggml_pool_1d** - 1D max pooling
4. **ggml_abs** - Absolute value (for SincNet stage 1)
5. **ggml_leaky_relu** - Leaky ReLU activation
6. **ggml_mul_mat** - Matrix multiplication (for Linear layers)
7. **ggml_add** - Addition (for biases)
8. **ggml_soft_max** - Softmax (for final activation)
9. **ggml_log** - Logarithm (for LogSoftmax)

### LSTM Operations
10. **ggml_lstm** - LSTM cell (or implement manually with basic ops)
    - Alternatively, implement using: sigmoid, tanh, element-wise multiply, add

### SincNet-Specific Operations
11. **Custom SincNet filter generation** - Generate bandpass filters from low_hz and band_hz parameters
    - This requires computing sinc functions: `sinc(x) = sin(πx) / (πx)`
    - Apply Hamming window
    - Normalize filters

### Tensor Manipulation
12. **ggml_reshape** - Reshape tensors
13. **ggml_permute** - Permute dimensions (for rearrange operation)
14. **ggml_cont** - Make tensor contiguous in memory

---

## Key Implementation Notes

### SincNet Filters
- The first convolutional layer uses **learnable bandpass filters** defined by `low_hz` and `band_hz` parameters
- Each filter is computed as: `filter = sinc(2π × (low_hz + band_hz) × t) - sinc(2π × low_hz × t)`
- Filters are multiplied by a Hamming window and normalized
- **Critical**: The `abs()` operation is applied ONLY after the first SincNet convolution

### LSTM State Management
- The LSTM is **bidirectional**, meaning it processes the sequence in both forward and backward directions
- With 4 layers and bidirectional=True, there are 8 LSTM passes total (4 forward + 4 backward)
- Hidden state size: 128 per direction, 256 total (concatenated)
- Dropout of 0.5 is applied between layers during training (not needed for inference)

### Instance Normalization
- Instance normalization is applied per sample (not across batch)
- Formula: `y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta`
- `gamma` and `beta` are learnable affine parameters (when `affine=True`)

### Receptive Field
- The model has a large receptive field due to the convolutional and pooling layers
- Each output frame at 589 frames for 10 seconds corresponds to ~170ms of input audio
- The stride of 10 in the first SincNet layer significantly reduces the temporal resolution

### Output Interpretation
- Output shape: `(batch, frames, 7)` where 7 is the number of speaker classes
- Each frame has a probability distribution over 7 classes (log probabilities due to LogSoftmax)
- Classes likely represent: [no speech, speaker 1, speaker 2, ..., speaker N, overlap]
- Post-processing typically involves:
  1. Exponentiating log probabilities to get actual probabilities
  2. Thresholding to detect active speakers
  3. Smoothing over time
  4. Merging adjacent segments

---

## Parameter Summary

| Component | Parameters | Percentage |
|-----------|------------|------------|
| SincNet | 42,602 | 2.9% |
| LSTM | 1,380,352 | 93.7% |
| Linear | 49,408 | 3.4% |
| Classifier | 903 | 0.1% |
| **Total** | **1,473,265** | **100%** |

The LSTM dominates the parameter count, accounting for over 93% of all parameters.

---

## Reference Activations

Reference activations for a 10-second test input are saved in:
- **File**: `segmentation-ggml/tests/reference_activations.npz`
- **Contents**: Intermediate activations at each major layer
- **Usage**: For validating GGML implementation against PyTorch

To load and inspect:
```python
import numpy as np
activations = np.load('segmentation-ggml/tests/reference_activations.npz')
print(activations.files)  # List all saved tensors
print(activations['sincnet_stage1_out'].shape)  # Access specific activation
```

---

## Additional Resources

- **Layer Info JSON**: `segmentation-ggml/docs/layer_info.json` - Complete layer parameters
- **Activation Shapes JSON**: `segmentation-ggml/docs/activation_shapes.json` - Tensor shapes at each stage
- **Source Code**: 
  - `src/pyannote/audio/models/segmentation/PyanNet.py` - Main model
  - `src/pyannote/audio/models/blocks/sincnet.py` - SincNet implementation
