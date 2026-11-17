# Contributing

Thanks for helping. This is a Python project; issues and pull requests are welcome.

## Set up and check your change

Needs Python 3.11+ (faster-whisper, PyAV and onnxruntime wheels, CPU only). Caches, virtual
environments, sample clips and models stay in git-ignored folders inside the repo when `HF_HOME`
points there (the local check sets it to `.tmp/hf`).

```bash
make setup   # install dependencies
make lint    # ruff, ruff format, mypy --strict
make test    # downloads 8 sample clips (2.5 MB) and Whisper tiny (75 MB), then all tests
make audit   # known-vulnerability check (as in CI)
make run     # language, silence removed and error rate on the 8 clips
```

A pull request is ready when `make lint` and `make test` pass and CI is green.

## Guidelines

- Keep each change focused; explain the problem it solves in the pull request.
- Add or update a test for every behaviour change, and update `docs/results/` when numbers change.
- Times in transcripts are always in the original recording; any new step that cuts or shifts audio must go through `TimeMap` and get a time-shift test.
- New sample clips need a public licence, a pinned revision and their published SHA-256 in `samples.py`.
- New model sizes need a pinned revision in `MODELS`; say in the docs whether they have been run.
- Significant design changes need an ADR in `docs/adr/` (copy `0001` for the format).
- Keep function comments to one short phrase; let names and tests explain the rest.
- Report security problems privately, as described in [SECURITY.md](SECURITY.md).

By contributing you agree that your work is released under the [MIT License](LICENSE).
