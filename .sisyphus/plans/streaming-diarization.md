# Streaming Diarization Implementation

## TL;DR

> **Quick Summary**: Implement pseudo-streaming diarization API for diarization-ggml with periodic reclustering. Provides real-time provisional output (<1s latency) while guaranteeing `streaming_finalize()` produces byte-identical results to offline `diarize()`.
> 
> **Deliverables**:
> - `streaming.h` - Public API with StreamingState, StreamingConfig, and function declarations
> - `streaming.cpp` - Core implementation with state management, incremental processing, and clustering
> - `streaming_state.h` - Internal state struct definition
> - `provisional.cpp` - Incremental cosine-based speaker assignment
> - `test_streaming.cpp` - C++ test harness
> - `test_streaming.py` - Python test for streaming/offline equivalence
> 
> **Estimated Effort**: Large (3-5 days)
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 5 → Task 6 → Task 8

---

## Context

### Original Request
Build a pseudo-streaming pipeline for diarization based on the STREAMING_DESIGN.md pre-draft outline. The streaming API should provide real-time provisional output while maintaining the ability to produce offline-identical results at finalization.

### Interview Summary
**Key Discussions**:
- **Initialization**: Zero-pad at stream start (matches offline behavior)
- **Output latency**: Immediate (<1s), with provisional labels that may be corrected
- **Memory**: Unbounded accumulation (guarantees offline-identical finalization)
- **Early assignment**: Incremental cosine clustering before first recluster (best UX)
- **Return value**: Full segment list from `streaming_push()` (caller can diff)
- **Recluster timing**: Every 60s of audio time (not wall-clock)
- **Provisional mode**: Configurable (can disable for confirmed-only output)
- **Thread safety**: Single-threaded (caller's responsibility)
- **Testing**: Automated verification (bash/Python scripts)

**Research Findings**:
- Current `diarize()` is ~750 lines of monolithic code
- Models are freed mid-pipeline after use (need to keep alive for streaming)
- VBx uses soft assignments (gamma) for centroid computation
- Hungarian assignment is per-chunk (local speaker → global cluster)
- Aggregation uses overlap-add with specific frame timing

### Metis Review
**Identified Gaps** (addressed):
- **Speaker label stability**: Accept ID instability in provisional; at recluster, map new VBx clusters to provisional centroids via Hungarian to minimize churn
- **PLDA timing**: Provisional clustering uses raw 256-dim embeddings; PLDA applied only at recluster/finalize
- **Zero-padding**: Zero-pad at START of first chunk (fill with zeros until 10s accumulated)
- **Partial chunks**: Buffer internally until full step (16000 samples) available
- **Model lifecycle**: Models owned by StreamingState, freed in streaming_free()
- **Recluster trigger**: Cumulative audio samples received (including silence)

---

## Work Objectives

### Core Objective
Implement a streaming diarization API that processes audio incrementally in 1-second chunks, provides immediate provisional speaker labels, and produces offline-identical results at finalization.

### Concrete Deliverables
- `diarization-ggml/src/streaming.h` - Public C++ API
- `diarization-ggml/src/streaming.cpp` - Main implementation
- `diarization-ggml/src/streaming_state.h` - Internal state struct
- `diarization-ggml/src/provisional.cpp` - Incremental cosine clustering
- `diarization-ggml/include/diarization_stream.h` - C API header (for bindings)
- `diarization-ggml/tests/test_streaming.cpp` - C++ test
- `diarization-ggml/tests/test_streaming.py` - Python equivalence test

### Definition of Done
- [ ] `streaming_finalize()` output == `diarize()` output for same audio (byte-identical RTTM)
- [ ] `streaming_push()` latency < 100ms for 1s chunk on Apple Silicon (p99)
- [ ] Memory growth is O(audio_duration), peak < 500MB for 45min audio
- [ ] All edge cases handled (empty audio, single speaker, silence, etc.)
- [ ] Tests pass: `diff streaming.rttm offline.rttm` returns 0

### Must Have
- Streaming API: init, push, recluster, finalize, free
- Provisional speaker assignment (cosine-based) before first recluster
- Periodic recluster (60s audio time) using full AHC + VBx
- Configuration options: recluster_interval, new_speaker_threshold, provisional_output
- Automated tests for streaming/offline equivalence

### Must NOT Have (Guardrails)
- **NO modifications to existing `diarize()` function** - preserve known-good offline behavior
- **NO threading/mutexes** - single-threaded only in initial implementation
- **NO disk I/O for intermediate state** - memory-only as specified
- **NO new dependencies** - use existing libs only
- **NO true streaming (sub-chunk latency)** - 1s minimum latency is acceptable
- **NO PLDA transform for provisional clustering** - raw 256-dim embeddings only
- **NO configurable chunk/step sizes** - hardcoded to match offline exactly

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (Python test scripts exist)
- **User wants tests**: Automated verification
- **Framework**: Python scripts + C++ test binary

### Automated Verification Commands

**Test 1: Streaming/Offline Equivalence (CRITICAL)**
```bash
# Generate streaming output
./build/bin/streaming_test \
  ../segmentation-ggml/segmentation.gguf \
  ../embedding-ggml/embedding.gguf \
  ../samples/sample.wav \
  --plda plda.bin \
  --coreml ../embedding-ggml/embedding.mlpackage \
  --seg-coreml ../segmentation-ggml/segmentation.mlpackage \
  -o /tmp/streaming.rttm

# Generate offline output
./build/bin/diarization-ggml \
  ../segmentation-ggml/segmentation.gguf \
  ../embedding-ggml/embedding.gguf \
  ../samples/sample.wav \
  --plda plda.bin \
  --coreml ../embedding-ggml/embedding.mlpackage \
  --seg-coreml ../segmentation-ggml/segmentation.mlpackage \
  -o /tmp/offline.rttm

# MUST be byte-identical
diff /tmp/streaming.rttm /tmp/offline.rttm
# Assert: Exit code 0
```

**Test 2: DER Sanity Check**
```bash
cd /Users/andyye/dev/pyannote-audio
.venv/bin/python3 diarization-ggml/tests/compare_rttm.py \
  /tmp/streaming.rttm /tmp/py_reference.rttm --threshold 1.0
# Assert: DER <= 1.0%
```

**Test 3: Latency Check**
```bash
./build/bin/streaming_test --benchmark ../samples/sample.wav
# Assert: Average push() time < 100ms
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Define data structures (streaming_state.h, streaming.h)
├── Task 4: Create CMake build integration
└── Task 7: Create test infrastructure (test_streaming.cpp skeleton)

Wave 2 (After Wave 1):
├── Task 2: Implement streaming_init/free
├── Task 3: Implement streaming_push (segmentation + embedding)
└── Task 9: Implement test_streaming.py skeleton

Wave 3 (After Wave 2):
├── Task 5: Implement provisional clustering (provisional.cpp)
├── Task 6: Implement streaming_recluster
└── Task 8: Implement streaming_finalize

Wave 4 (After Wave 3):
└── Task 10: Final integration testing and edge cases

Critical Path: Task 1 → Task 2 → Task 3 → Task 5 → Task 6 → Task 8 → Task 10
Parallel Speedup: ~35% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3, 5, 6, 8 | 4, 7 |
| 2 | 1 | 3, 6, 8 | 4, 7 |
| 3 | 1, 2 | 5, 8 | 9 |
| 4 | None | 7, 10 | 1, 7 |
| 5 | 1, 3 | 6, 8 | 6 (partially) |
| 6 | 1, 2, 5 | 8 | 9 |
| 7 | 4 | 10 | 1, 2, 9 |
| 8 | 3, 6 | 10 | 9 |
| 9 | 1 | 10 | 3, 5, 6, 7 |
| 10 | 7, 8, 9 | None | None (final) |

---

## TODOs

- [x] 1. Define Core Data Structures

  **What to do**:
  - Create `streaming_state.h` with `StreamingState` struct containing:
    - Audio buffer: `std::vector<float> audio_buffer`
    - Embedding buffer: `std::vector<float> embeddings` (N × 256)
    - Chunk tracking: `std::vector<int> chunk_idx`, `std::vector<int> local_speaker_idx`
    - Binarized segmentation: `std::vector<float> binarized` (chunks × 589 × 3)
    - Provisional centroids: `std::vector<float> centroids` (K × 256)
    - Global labels: `std::vector<int> global_labels`
    - Bookkeeping: `int chunks_processed`, `int last_recluster_chunk`, `double audio_time_processed`
    - Model references: pointers to segmentation and embedding CoreML contexts
    - PLDA model: `PLDAModel plda`
    - Configuration: `StreamingConfig config`
  - Create `streaming.h` with public API:
    - `StreamingConfig` struct with `recluster_interval_sec`, `new_speaker_threshold`, `provisional_output`
    - Function declarations: `streaming_init`, `streaming_push`, `streaming_recluster`, `streaming_finalize`, `streaming_free`
  - Create `diarization_stream.h` (C API for FFI bindings)

  **Must NOT do**:
  - Do not use raw pointers for buffers (use std::vector)
  - Do not add threading constructs

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Header files with struct definitions, straightforward
  - **Skills**: []
    - No special skills needed for header-only work

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 4, 7)
  - **Blocks**: Tasks 2, 3, 5, 6, 8
  - **Blocked By**: None (can start immediately)

  **References**:
  - `diarization-ggml/include/diarization.h:5-26` - Existing DiarizationConfig and DiarizationResult structs (follow same style)
  - `diarization-ggml/src/vbx.cpp:44-49` - VBxResult struct pattern for clustering results
  - `diarization-ggml/src/diarization.cpp:39-57` - Pipeline constants to include
  - `diarization-ggml/STREAMING_DESIGN.md:27-44` - Original state struct design (adapt this)

  **Acceptance Criteria**:
  ```bash
  # Verify headers compile
  cd /Users/andyye/dev/pyannote-audio/diarization-ggml
  cmake -B build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON
  cmake --build build 2>&1 | grep -E "error:|streaming"
  # Assert: No compilation errors mentioning streaming headers
  ```

  **Commit**: YES
  - Message: `feat(streaming): add core data structures for streaming diarization`
  - Files: `src/streaming_state.h`, `src/streaming.h`, `include/diarization_stream.h`
  - Pre-commit: `cmake --build build`

---

- [ ] 2. Implement streaming_init and streaming_free

  **What to do**:
  - Implement `streaming_init()`:
    - Allocate StreamingState
    - Load segmentation CoreML model (keep alive, don't free)
    - Load embedding CoreML model (keep alive)
    - Load PLDA model
    - Initialize all vectors as empty
    - Set bookkeeping to zeros
    - Return pointer to state
  - Implement `streaming_free()`:
    - Free CoreML contexts
    - Clear all vectors
    - Delete state
  - Follow RAII-style initialization (all-or-nothing)
  - Handle partial initialization failure (cleanup already-loaded models)

  **Must NOT do**:
  - Do not free models mid-operation like offline pipeline does
  - Do not use global state

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Standard initialization/cleanup code, follows existing patterns
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 9)
  - **Blocks**: Tasks 3, 6, 8
  - **Blocked By**: Task 1

  **References**:
  - `diarization-ggml/src/diarization.cpp:309-344` - Segmentation model loading pattern
  - `diarization-ggml/src/diarization.cpp:350-384` - Embedding model loading pattern
  - `diarization-ggml/src/diarization.cpp:390-419` - PLDA model loading
  - `embedding-ggml/src/coreml/coreml_bridge.h` - embedding_coreml_init/free signatures
  - `segmentation-ggml/src/coreml/segmentation_coreml_bridge.h` - segmentation_coreml_init/free signatures

  **Acceptance Criteria**:
  ```bash
  # Verify init/free don't leak (basic check)
  cd /Users/andyye/dev/pyannote-audio/diarization-ggml
  cat > /tmp/test_init.cpp << 'EOF'
  #include "streaming.h"
  int main() {
    StreamingConfig cfg;
    cfg.base.seg_coreml_path = "../segmentation-ggml/segmentation.mlpackage";
    cfg.base.coreml_path = "../embedding-ggml/embedding.mlpackage";
    cfg.base.plda_path = "plda.bin";
    auto* state = streaming_init(cfg);
    if (!state) return 1;
    streaming_free(state);
    return 0;
  }
  EOF
  # Compile and run
  cmake --build build --target streaming_test
  ./build/bin/streaming_test --init-only
  # Assert: Exit code 0, no crash
  ```

  **Commit**: YES
  - Message: `feat(streaming): implement streaming_init and streaming_free`
  - Files: `src/streaming.cpp`
  - Pre-commit: `cmake --build build`

---

- [ ] 3. Implement streaming_push (Core Processing)

  **What to do**:
  - Implement `streaming_push()`:
    1. **Append audio**: Add incoming samples to `audio_buffer`
    2. **Check for full step**: If `audio_buffer.size() >= CHUNK_SAMPLES` for first chunk, or accumulated new samples >= STEP_SAMPLES
    3. **Extract 10s chunk**: Get last 160000 samples (zero-pad front if needed)
    4. **Run segmentation**: Call `segmentation_coreml_infer()` → 589×7 logits
    5. **Powerset decode**: Call `powerset_to_multilabel()` → 589×3 binarized
    6. **Store binarized**: Append to `binarized` buffer
    7. **Extract embeddings**: Call `extract_embeddings()` for this chunk → 3×256
    8. **Store embeddings**: Append to `embeddings` buffer (with NaN for inactive speakers)
    9. **Provisional assign**: Call provisional clustering (Task 5) to get labels
    10. **Check recluster trigger**: If chunks_processed - last_recluster >= 60, call `streaming_recluster()`
    11. **Build output segments**: Aggregate binarized + labels → RTTM-style segments
    12. **Return segments**: Full timeline from stream start

  **Must NOT do**:
  - Do not run VBx on every push (only at recluster)
  - Do not apply PLDA transform (only at recluster)
  - Do not free models after inference

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Core pipeline logic, requires careful integration of multiple components
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 9)
  - **Blocks**: Tasks 5, 8
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `diarization-ggml/src/diarization.cpp:425-509` - Segmentation loop (follow same chunk extraction pattern)
  - `diarization-ggml/src/diarization.cpp:511-529` - Powerset conversion
  - `diarization-ggml/src/diarization.cpp:571-596` - Embedding extraction call
  - `diarization-ggml/src/aggregation.cpp:14-77` - Aggregation logic (simplify for streaming)
  - `diarization-ggml/src/diarization.cpp:866-901` - RTTM segment generation

  **Acceptance Criteria**:
  ```bash
  # Test push with sample audio
  cd /Users/andyye/dev/pyannote-audio/diarization-ggml
  ./build/bin/streaming_test --push-only ../samples/sample.wav
  # Assert: Exit code 0
  # Assert: Output shows segments being generated
  # Assert: Segment count increases with each 1s push
  ```

  **Commit**: YES
  - Message: `feat(streaming): implement streaming_push core processing`
  - Files: `src/streaming.cpp`
  - Pre-commit: `cmake --build build`

---

- [x] 4. Create CMake Build Integration

  **What to do**:
  - Update `diarization-ggml/CMakeLists.txt`:
    - Add `streaming.cpp`, `provisional.cpp` to library sources
    - Add `streaming_test` executable target
    - Link streaming_test with same libraries as diarization-ggml
  - Ensure CoreML flags propagate correctly

  **Must NOT do**:
  - Do not modify existing targets
  - Do not add new dependencies

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: CMake changes are straightforward
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 7)
  - **Blocks**: Task 7, 10
  - **Blocked By**: None

  **References**:
  - `diarization-ggml/CMakeLists.txt` - Existing CMake configuration (add to it)

  **Acceptance Criteria**:
  ```bash
  cd /Users/andyye/dev/pyannote-audio/diarization-ggml
  cmake -B build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON
  cmake --build build
  ls build/bin/streaming_test
  # Assert: streaming_test binary exists
  ```

  **Commit**: YES (group with Task 1)
  - Message: `build(streaming): add streaming sources and test target to CMake`
  - Files: `CMakeLists.txt`
  - Pre-commit: `cmake --build build`

---

- [ ] 5. Implement Provisional Clustering (provisional.cpp)

  **What to do**:
  - Create `provisional.cpp` with incremental cosine clustering:
    - `provisional_init()`: Initialize empty centroid list
    - `provisional_assign()`: For each new embedding:
      1. If no centroids: create SPEAKER_00 from first valid embedding
      2. Compute cosine similarity to all existing centroids
      3. If max_sim > (1 - threshold): assign to that speaker, update centroid (running mean)
      4. Else: create new speaker (SPEAKER_01, etc.)
    - `provisional_get_centroids()`: Return current centroids for Hungarian assignment
    - `provisional_remap_labels()`: After VBx recluster, map VBx clusters to provisional centroids to minimize label churn
  - Use cosine distance threshold of 0.6 (same as AHC_THRESHOLD)
  - Use running mean for centroid updates: `centroid = (centroid * n + embedding) / (n + 1)`

  **Must NOT do**:
  - Do not use PLDA transform (raw 256-dim embeddings)
  - Do not run VBx here
  - Do not use full AHC

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Straightforward clustering algorithm, clear specification
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 6)
  - **Blocks**: Tasks 6, 8
  - **Blocked By**: Tasks 1, 3

  **References**:
  - `diarization-ggml/src/clustering.cpp:18-28` - cosine_distance function (reuse this)
  - `diarization-ggml/src/clustering.cpp:251-280` - Centroid computation pattern
  - `diarization-ggml/src/clustering.cpp:187-204` - hungarian_assign for label remapping

  **Acceptance Criteria**:
  ```bash
  # Test provisional clustering on first 30s (before recluster)
  ./build/bin/streaming_test --provisional-only ../samples/sample.wav
  # Assert: Outputs provisional speaker labels
  # Assert: Speaker count is reasonable (1-3 for test audio)
  # Assert: Labels are stable (same speaker stays same ID)
  ```

  **Commit**: YES
  - Message: `feat(streaming): implement provisional cosine clustering`
  - Files: `src/provisional.cpp`, `src/provisional.h`
  - Pre-commit: `cmake --build build`

---

- [ ] 6. Implement streaming_recluster

  **What to do**:
  - Implement `streaming_recluster()`:
    1. **Filter embeddings**: Call `filter_embeddings()` on accumulated embeddings
    2. **L2 normalize**: Normalize filtered embeddings
    3. **PLDA transform**: Apply `plda_transform()` to get 128-dim features
    4. **AHC cluster**: Call `ahc_cluster()` with threshold 0.6
    5. **VBx refine**: Call `vbx_cluster()` with existing parameters
    6. **Compute VBx centroids**: Use soft assignments (gamma) to compute weighted centroids
    7. **Map to provisional labels**: Use Hungarian assignment to map VBx clusters → provisional centroid labels (minimize churn)
    8. **Reassign all embeddings**: Call `constrained_argmax()` per chunk
    9. **Update state**: Store new global_labels, update provisional centroids from VBx
    10. **Update bookkeeping**: `last_recluster_chunk = chunks_processed`
  - If too few embeddings for clustering (< 2), skip and keep provisional labels

  **Must NOT do**:
  - Do not modify AHC/VBx parameters
  - Do not skip label remapping (causes UX churn)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Complex integration of multiple clustering steps
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: PARTIAL
  - **Parallel Group**: Wave 3 (with Task 5)
  - **Blocks**: Task 8
  - **Blocked By**: Tasks 1, 2, 5

  **References**:
  - `diarization-ggml/src/diarization.cpp:617-789` - Full clustering sequence (follow exactly)
  - `diarization-ggml/src/diarization.cpp:646-664` - L2 normalization
  - `diarization-ggml/src/diarization.cpp:666-680` - PLDA transform call
  - `diarization-ggml/src/diarization.cpp:683-708` - AHC + VBx calls
  - `diarization-ggml/src/diarization.cpp:723-767` - Centroid computation from gamma

  **Acceptance Criteria**:
  ```bash
  # Test recluster at 60s
  ./build/bin/streaming_test --recluster-test ../samples/sample.wav
  # Assert: Recluster completes without error
  # Assert: Speaker labels after recluster are reasonable
  # Assert: Label churn is minimized (compare before/after)
  ```

  **Commit**: YES
  - Message: `feat(streaming): implement streaming_recluster with VBx`
  - Files: `src/streaming.cpp`
  - Pre-commit: `cmake --build build`

---

- [ ] 7. Create Test Infrastructure (test_streaming.cpp)

  **What to do**:
  - Create `tests/test_streaming.cpp` with:
    - `--init-only`: Test init/free without processing
    - `--push-only <audio>`: Test push without finalize
    - `--provisional-only <audio>`: Test first 30s (no recluster)
    - `--recluster-test <audio>`: Test recluster behavior
    - `--benchmark <audio>`: Measure push latency
    - Default: Full streaming test with finalize, output to RTTM
  - Parse command-line arguments
  - Load audio file, push in 1s chunks
  - Report timing, segment counts, speaker counts
  - Exit codes: 0 success, 1 failure

  **Must NOT do**:
  - Do not add external test frameworks (keep it simple)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Test harness with clear structure
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1-2 (evolves as other tasks complete)
  - **Blocks**: Task 10
  - **Blocked By**: Task 4

  **References**:
  - `diarization-ggml/src/main.cpp` - Existing CLI pattern (follow similar style)
  - `diarization-ggml/src/diarization.cpp:82-141` - WAV loading (copy this)

  **Acceptance Criteria**:
  ```bash
  ./build/bin/streaming_test --help
  # Assert: Shows usage with all options
  
  ./build/bin/streaming_test --init-only
  # Assert: Exit code 0
  ```

  **Commit**: YES (group with Task 4)
  - Message: `test(streaming): add C++ test harness for streaming API`
  - Files: `tests/test_streaming.cpp`
  - Pre-commit: `cmake --build build`

---

- [ ] 8. Implement streaming_finalize

  **What to do**:
  - Implement `streaming_finalize()`:
    1. **Force final recluster**: Call `streaming_recluster()` regardless of timing
    2. **Full aggregation**: Use `aggregate_chunks()` exactly as offline does
    3. **Compute speaker count**: Call `compute_speaker_count()` 
    4. **Build clustered segmentation**: Same logic as `diarization.cpp:814-845`
    5. **to_diarization**: Call `to_diarization()` for final discrete output
    6. **Generate segments**: Same RTTM logic as `diarization.cpp:866-909`
    7. **Return DiarizationResult**: Populate result struct
  - This MUST produce byte-identical output to `diarize()` for same audio

  **Must NOT do**:
  - Do not skip any step from offline pipeline
  - Do not modify aggregation parameters

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Critical for correctness, must match offline exactly
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after Tasks 3, 6)
  - **Blocks**: Task 10
  - **Blocked By**: Tasks 3, 6

  **References**:
  - `diarization-ggml/src/diarization.cpp:792-961` - Full post-clustering to RTTM (MUST MATCH)
  - `diarization-ggml/src/aggregation.cpp:79-111` - compute_speaker_count
  - `diarization-ggml/src/aggregation.cpp:113-150` - to_diarization

  **Acceptance Criteria**:
  ```bash
  # CRITICAL TEST: Streaming/Offline equivalence
  cd /Users/andyye/dev/pyannote-audio/diarization-ggml
  
  # Generate streaming output
  ./build/bin/streaming_test \
    ../segmentation-ggml/segmentation.gguf \
    ../embedding-ggml/embedding.gguf \
    ../samples/sample.wav \
    --plda plda.bin \
    --coreml ../embedding-ggml/embedding.mlpackage \
    --seg-coreml ../segmentation-ggml/segmentation.mlpackage \
    -o /tmp/streaming.rttm

  # Generate offline output
  ./build/bin/diarization-ggml \
    ../segmentation-ggml/segmentation.gguf \
    ../embedding-ggml/embedding.gguf \
    ../samples/sample.wav \
    --plda plda.bin \
    --coreml ../embedding-ggml/embedding.mlpackage \
    --seg-coreml ../segmentation-ggml/segmentation.mlpackage \
    -o /tmp/offline.rttm

  # MUST be byte-identical
  diff /tmp/streaming.rttm /tmp/offline.rttm
  echo "Exit code: $?"
  # Assert: Exit code 0
  ```

  **Commit**: YES
  - Message: `feat(streaming): implement streaming_finalize for offline-identical output`
  - Files: `src/streaming.cpp`
  - Pre-commit: `cmake --build build && diff test`

---

- [ ] 9. Create Python Equivalence Test (test_streaming.py)

  **What to do**:
  - Create `tests/test_streaming.py`:
    - Run `streaming_test` binary on `samples/sample.wav`
    - Run `diarization-ggml` binary on same audio
    - Compare RTTM outputs (must be byte-identical)
    - Also run DER comparison against Python reference
    - Exit 0 if all pass, 1 if any fail

  **Must NOT do**:
  - Do not add new Python dependencies beyond what's in .venv

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple script, follows existing test patterns
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2-3
  - **Blocks**: Task 10
  - **Blocked By**: Task 1 (needs to know output format)

  **References**:
  - `diarization-ggml/tests/compare_rttm.py` - Existing DER comparison (call this)
  - `diarization-ggml/tests/compare_pipeline.py` - Existing test pattern

  **Acceptance Criteria**:
  ```bash
  cd /Users/andyye/dev/pyannote-audio
  .venv/bin/python3 diarization-ggml/tests/test_streaming.py
  # Assert: Exit code 0
  # Assert: Output shows "PASS: Streaming output matches offline"
  ```

  **Commit**: YES
  - Message: `test(streaming): add Python equivalence test`
  - Files: `tests/test_streaming.py`
  - Pre-commit: `.venv/bin/python3 tests/test_streaming.py`

---

- [ ] 10. Final Integration Testing and Edge Cases

  **What to do**:
  - Test edge cases:
    - Empty audio (0 samples) → empty segment list
    - Audio < 10s → processes with zero-padding
    - Single speaker → 1 cluster
    - All silence → no embeddings, empty output
    - Push after finalize → error
  - Run memory leak check with AddressSanitizer
  - Run latency benchmark
  - Update STREAMING_DESIGN.md with final implementation notes

  **Must NOT do**:
  - Do not add edge case handling that differs from offline behavior

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Testing and documentation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (final)
  - **Blocks**: None
  - **Blocked By**: Tasks 7, 8, 9

  **References**:
  - All previous task implementations
  - `AGENTS.md:Testing Protocol` - Follow existing test patterns

  **Acceptance Criteria**:
  ```bash
  # Full test suite
  cd /Users/andyye/dev/pyannote-audio/diarization-ggml
  
  # C++ tests
  ./build/bin/streaming_test ../samples/sample.wav -o /tmp/streaming.rttm
  diff /tmp/streaming.rttm /tmp/offline.rttm && echo "PASS: Equivalence"
  
  # Python tests
  cd /Users/andyye/dev/pyannote-audio
  .venv/bin/python3 diarization-ggml/tests/test_streaming.py && echo "PASS: Python"
  
  # DER check
  .venv/bin/python3 diarization-ggml/tests/compare_rttm.py \
    /tmp/streaming.rttm /tmp/py_reference.rttm --threshold 1.0
  # Assert: DER <= 1.0%
  ```

  **Commit**: YES
  - Message: `test(streaming): complete edge case testing and documentation`
  - Files: `tests/test_streaming.cpp`, `STREAMING_DESIGN.md`
  - Pre-commit: Full test suite

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 + 4 | `feat(streaming): add core data structures and CMake` | headers, CMakeLists.txt | `cmake --build build` |
| 2 | `feat(streaming): implement init/free` | streaming.cpp | `--init-only` test |
| 3 | `feat(streaming): implement push core` | streaming.cpp | `--push-only` test |
| 5 | `feat(streaming): implement provisional clustering` | provisional.cpp | `--provisional-only` test |
| 6 | `feat(streaming): implement recluster` | streaming.cpp | `--recluster-test` |
| 7 | `test(streaming): add C++ test harness` | test_streaming.cpp | `cmake --build build` |
| 8 | `feat(streaming): implement finalize` | streaming.cpp | `diff` test |
| 9 | `test(streaming): add Python equivalence test` | test_streaming.py | Python test |
| 10 | `test(streaming): complete integration testing` | tests, docs | Full suite |

---

## Success Criteria

### Verification Commands
```bash
# 1. Build succeeds
cd /Users/andyye/dev/pyannote-audio/diarization-ggml
cmake -B build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON
cmake --build build

# 2. Streaming/Offline equivalence (CRITICAL)
./build/bin/streaming_test ... -o /tmp/streaming.rttm
./build/bin/diarization-ggml ... -o /tmp/offline.rttm
diff /tmp/streaming.rttm /tmp/offline.rttm  # Exit code 0

# 3. DER check
.venv/bin/python3 diarization-ggml/tests/compare_rttm.py \
  /tmp/streaming.rttm /tmp/py_reference.rttm --threshold 1.0
# DER <= 1.0%

# 4. Python test
.venv/bin/python3 diarization-ggml/tests/test_streaming.py
# Exit code 0
```

### Final Checklist
- [ ] All headers compile without errors
- [ ] streaming_init/free work correctly
- [ ] streaming_push processes audio incrementally
- [ ] Provisional clustering assigns stable labels
- [ ] streaming_recluster runs full AHC+VBx
- [ ] streaming_finalize produces offline-identical output
- [ ] All edge cases handled
- [ ] Memory usage is O(audio_duration)
- [ ] Push latency < 100ms
