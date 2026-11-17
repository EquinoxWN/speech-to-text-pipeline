# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub:
**Security > Report a vulnerability** on this repository
(`https://github.com/EquinoxWN/speech-to-text-pipeline/security/advisories/new`). Do not open a public issue.

Include what you found, how to reproduce it, and the impact you expect. You will get an answer
within 7 days. Fixes are released as soon as they are ready, and you are credited unless you ask
not to be.

## Supported versions

This project is pre-1.0. Only the latest commit on `main` receives fixes.

## Scope

- **In scope:** a tampered or unexpected sample clip or model being used, downloads that are not size-limited or not over HTTPS, audio files that are decoded without the size and length caps, and output files left half-written.
- **Out of scope:** transcription accuracy of Whisper models; that is a measurement, not a vulnerability. Bugs inside FFmpeg (bundled with PyAV), CTranslate2 or onnxruntime belong to those projects.

## How this repository protects itself

- Every GitHub Action is pinned to a full commit SHA, and workflows run with read-only
  permissions and without persisted credentials.
- Dependabot update pull requests are paused (`open-pull-requests-limit: 0`) so the repository keeps a single `main` branch; dependencies are updated by hand and checked by the CI audit.
- Sample clips are downloaded over HTTPS only, capped at 2 MB each and refused unless their SHA-256 matches the pinned value; Whisper models are pinned to a Hugging Face commit.
- Input audio is capped at 512 MB and 4 hours, checked while decoding.
- CI runs lint, tests and a known-vulnerability check (pip-audit) on every push and pull request.
