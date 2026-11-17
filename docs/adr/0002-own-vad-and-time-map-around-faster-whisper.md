# ADR 0002: Own VAD step, time map and chunk loop around faster-whisper

- **Status:** Accepted (M1)

## Context

faster-whisper can do everything M1 needs in one call: `model.transcribe(path, vad_filter=True,
word_timestamps=True)` decodes the file, runs Silero VAD, concatenates speech, transcribes and maps
timestamps back. That is the shortest path, but it hides the speech spans, the chunk plan and the
time mapping, so none of them can be tested, measured or reused by the diarization step in M2.
While building M1 it also turned out that its `decode_audio` fails with the current PyAV (19), and
that joined utterances can be lost when Whisper ends a window after the first one.

## Decision

The pipeline calls faster-whisper only for two things: language detection on a sample array and
transcription of one chunk array with its own VAD off. Decoding (PyAV resampler), Silero VAD (the
model faster-whisper ships), the `TimeMap`, the chunk plan and the resume pass are this project's
code, each with unit tests and a real-model test.

## Consequences

- Each step is testable on its own: resampling against tones, the time map against exact numbers,
  chunking against random layouts, the pipeline against a scripted engine, and the whole thing
  against Whisper tiny on public clips.
- Stats (seconds removed, chunks, engine calls, real-time factor) come for free and are reported.
- M2 can align diarization turns to the same spans and original-time words.
- More code to maintain than one library call; faster-whisper's own improvements to its VAD path
  are not picked up automatically.
- The resume pass can cost an extra engine call per chunk (2 instead of 1 on the two-utterance demo).
