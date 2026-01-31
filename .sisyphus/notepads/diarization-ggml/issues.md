# Issues — diarization-ggml

## RESOLVED: GGML Segmentation Model Output (was "Precision Issue")
- **Status**: RESOLVED (commit db4bd30b)
- **Original diagnosis**: F16 weight precision causing different LSTM outputs — WRONG
- **Actual root cause**: Tensor layout mismatch — GGML outputs [class, frame], powerset expects [frame, class]
- **Fix**: In-place transpose after model_infer() in diarization.cpp
- **Result**: DER 0.28% fully native, no Python preprocessing needed
