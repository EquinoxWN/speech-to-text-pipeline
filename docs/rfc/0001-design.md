# RFC 0001: speech-to-text-pipeline design

- **Status:** Accepted for M1 (M2 and M3 sections are plans)
- **Author:** EquinoxWN
- **Created:** 2026-11-17

## Problem

Meeting recordings are long, mostly silence or crosstalk, recorded at whatever sample rate and
channel layout the device used, and often not in English. People who need the content want a
transcript they can search and jump into by timestamp, in the language that was spoken. Off-the-shelf
use of Whisper hides three things that decide whether such a transcript can be trusted: whether the
audio was converted correctly, whether silence was removed without losing speech, and whether the
timestamps still point at the right place in the original recording after silence was cut out.

### Users and critical-to-quality measures (Define)

Users: engineers building meeting or interview transcription, and teams evaluating speech models on
their own audio (see the README, "Who it helps").

| CTQ | Measure | M1 target |
|---|---|---|
| Correct input conversion | any FFmpeg-readable file becomes 16 kHz mono float32; pitch kept; content above 8 kHz filtered, not folded | tone tests within 1 Hz; 12 kHz tone below RMS 0.01 after resampling |
| Silence removed, speech kept | seconds removed; speech spans inside added silence | added silence removed on all 8 sample clips; no span inside a 3 s gap |
| Timestamps point at the original recording | word start shift when silence is added in front | 5.0 s added gives a 5.0 s shift within 0.05 s for every word |
| Language detected | correct language on public clips (en, id, ja, zh) | 8 of 8 FLEURS clips with Whisper tiny |
| No speech lost at chunk joins | utterances joined by VAD are all transcribed | two joined utterances both transcribed (test with the real model) |
| Bounded resources | file size, length, download size caps; CPU only | 512 MB, 4 h, 2 MB per sample clip |

## Goals

- M1: audio is resampled, and voice-activity detection removes silence before transcription.
- M1: faster-whisper transcribes chunks with word timestamps and detects the language (including
  Indonesian, Japanese and Chinese).
- M2: pyannote diarization finds who spoke when, and words are aligned to speakers.
- M2: overlapping speech and long recordings are handled with careful chunk boundaries.
- M3: word error rate and diarization error rate are measured on public test sets per language.
- M3: speed and accuracy are compared across model sizes on CPU and GPU.

## Non-goals

- Running as a hosted production service, streaming or real-time transcription.
- Training or fine-tuning speech models.
- Claiming accuracy numbers from 8 clips: M1 reports them only to show the pipeline works end to end;
  the per-language evaluation on full test sets is M3.

## Design (M1)

```
file --PyAV/FFmpeg--> 16 kHz mono float32 --Silero VAD--> speech spans
     --concatenate spans--> silence-free audio + TimeMap
     --first 30 s--> Whisper language detection
     --plan_chunks (<= 30 s, cut at span joins)--> Whisper per chunk (word timestamps)
     --resume at the next join if Whisper stopped early--> TimeMap --> original-time words
```

- **Decode and resample** (`audio.py`): PyAV (bundled FFmpeg) decodes the first audio stream and its
  `AudioResampler` converts to `flt`, mono, 16 kHz in one pass (FFmpeg's filtered resampler, so
  content above 8 kHz is removed rather than aliased). Size (512 MB) and length (4 h) are capped
  while decoding, so an oversized file is refused before it is fully in memory.
- **VAD** (`vad.py`): Silero VAD v6 (ONNX, shipped inside faster-whisper, run with onnxruntime on one
  thread). Defaults: threshold 0.5, minimum speech 250 ms, minimum silence 500 ms, 200 ms padding.
  Spans are kept in samples, validated (sorted, non-overlapping, inside the audio) and concatenated.
- **TimeMap** (`vad.py`): cumulative span offsets; a compact time maps to `span.start + (t -
  compact_start)`. An end time that falls exactly on a join maps to the end of the earlier span,
  so a word ending at a join is not stretched over the removed silence.
- **Chunking** (`chunking.py`): spans are packed greedily into chunks of at most 30 s (one Whisper
  window), cutting only at span joins; a single span longer than 30 s is cut at 30 s.
- **Transcription** (`engine.py`): faster-whisper (CTranslate2), int8 weights, CPU, 2 threads, beam 5,
  `word_timestamps=True`, its own VAD off, `condition_on_previous_text=False`. Language detection uses
  the first 30 s of speech (silence removed), so a long silent intro does not decide the language.
  The model id and Hugging Face revision are pinned in `MODELS`.
- **Resume pass** (`pipeline.py`): after VAD joins utterances with no pause, Whisper sometimes ends
  the window after the first one (measured on the 2-clip demo: the second sentence was lost). When a
  pass covers less than its chunk, the rest is sent again from the next span join at or after the
  covered end (0.3 s tolerance), only if at least 1 s is left; each join is used at most once.
- **Outputs** (`transcript.py`): JSON (schema version 1, UTC creation time) and text with
  `HH:MM:SS.mmm` times in the original recording; atomic writes.
- **Samples** (`samples.py`): 8 FLEURS test clips (2 each for en, id, ja, zh), CC BY 4.0, pinned to one
  dataset commit, each checked against the SHA-256 the Hub publishes; HTTPS only, 60 s timeout, 3
  attempts with 1 s and 2 s backoff, 2 MB cap, no partial files.

## Alternatives considered

| Option | Why not (for M1) |
|---|---|
| faster-whisper's built-in `vad_filter=True` | Does the same VAD, but hides the spans, the time map and the chunk plan inside one call, so none of them can be tested or reported (seconds removed, chunk count). M2 needs the spans for diarization alignment. See [ADR 0002](../adr/0002-own-vad-and-time-map-around-faster-whisper.md). |
| openai-whisper (PyTorch) | Needs PyTorch (about 200 MB+ wheels) and is several times slower on CPU than CTranslate2 int8; faster-whisper uses the same weights converted. |
| WhisperX (faster-whisper + wav2vec2 alignment + pyannote) | Better word alignment and diarization in one package, but pulls PyTorch, alignment models per language and gated pyannote models; too heavy for M1 and it would hide the steps this repo wants to measure. Candidate comparison in M3. |
| Energy-based VAD (RMS threshold) | No model, but treats steady noise and music as speech; Silero already ships in the faster-whisper wheel at no extra cost. A steady 1 kHz tone is correctly rejected by Silero (test). |
| `faster_whisper.decode_audio` for decoding | Fails with PyAV 19 (`av.open` no longer accepts `metadata_errors`, found while building M1); decoding is 30 lines with PyAV's resampler and allows a streaming length cap. |

## Measurement plan

- M1 (done, see [docs/results/m1.md](../results/m1.md)): resampling tone tests; VAD seconds removed and
  span positions on padded FLEURS clips; time-shift test; language detection on 8 clips; WER (en, id)
  and CER (ja, zh) on the same clips as a smoke check only; real-time factor on CPU.
- M3: WER/CER per language on full FLEURS test splits, DER on a public diarization set (AMI or
  VoxConverse), model sizes tiny to large-v3 on CPU and on the GPU, each with its pinned revision.

## Milestones

- **M1:** resampling and VAD before transcription; chunked faster-whisper transcription with word
  timestamps and language detection.
- **M2:** pyannote diarization and word-to-speaker alignment; overlap and long-recording chunk
  boundaries.
- **M3:** WER and DER per language on public test sets; model-size comparison on CPU and GPU.

## Risks and open questions

- Whisper tiny is the only model run so far; larger models may change error rates and timing, not the
  pipeline logic (larger-model runs are not yet done).
- Whisper writes Chinese in Traditional script on these clips while FLEURS references are Simplified,
  which inflates CER; script conversion is not in M1.
- The resume pass only helps when the early stop happens before a span join; an utterance cut inside
  one long span is not recovered.
- The 8-clip numbers are too few to compare languages or models.
