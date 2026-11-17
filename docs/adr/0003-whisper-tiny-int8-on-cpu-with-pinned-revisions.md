# ADR 0003: Whisper tiny, int8 on CPU, pinned model and data revisions for M1

- **Status:** Accepted (M1)

## Context

M1 has to show that resampling, VAD, language detection and word timestamps work, on public audio in
four languages, on a laptop shared with other builds (about 4 GB of memory free). Larger Whisper
models transcribe better but are slower and use more memory (base about 145 MB, small about 480 MB
of weights, large-v3 about 3 GB). Model repositories and datasets on the Hugging Face Hub can change
under the same name.

## Decision

- M1 runs `Systran/faster-whisper-tiny` (about 75 MB, MIT licence) with int8 weights on CPU and 2
  threads. `base` and `small` are listed with pinned revisions but have not been run.
- Every model is pinned to a commit (`MODELS` in `engine.py`); downloads go to `HF_HOME`, which the
  local scripts set inside the repo's `.tmp/`.
- Sample audio is 8 FLEURS test clips (CC BY 4.0) from `FluidInference/fleurs-full` at one commit,
  each checked against the SHA-256 the Hub publishes for its LFS object.

## Consequences

- The full test suite, including the real-model tests, runs on CPU in minutes and fits in a few
  hundred MB of memory.
- Tiny's error rates are high (WER 0.05 to 0.56, CER 0.24 to 0.48 on the 8 clips), so M1 numbers show
  that the pipeline works, not how good Whisper is; the model comparison is M3.
- Thresholds in the real-model tests are set for tiny; a different model may need its own.
- Pinned revisions mean upstream fixes are taken only by changing the pin on purpose.
