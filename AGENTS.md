# Repository guide

## Project

This repository implements generative speech error correction for child-speech ASR. Whisper produces a greedy baseline and sampled transcript candidates; Google FLAN-T5 uses those candidates to generate a corrected transcript. The pipeline evaluates WER and produces child-, group-, and corpus-level analyses and figures.

## Manuscript writing

- Use the `humanizer:humanizer` skill before returning any text intended for the manuscript, whether in chat or in a file. This includes draft prose, revisions, captions, and table notes.

## Structure and workflow

- `bash_scripts/train.sh` is the single HPC entry point. It activates or creates the configured Conda environment and runs `scripts/run_pipeline.py`.
- `scripts/` contains the pipeline stages. Read the relevant stage and its callers before changing pipeline behavior.
- `configs/baseline.yaml` is the source of truth for paths, models, decoding, training, splitting, and output settings. Avoid introducing duplicate hard-coded values in scripts.
- `envs/` defines runtime dependencies. Keep environment changes aligned with the entry point and document new setup requirements in `README.md`.
- `tests/` contains focused regression tests for data alignment, splitting, Whisper samples, analysis, and figures.
- `docs/changelog.md` records dated project changes.

## Data and artifacts

- Audio and source references are prepared by the sibling `asr-dataset-pipelines` project. This repository consumes that data; do not duplicate or modify upstream data-preparation logic here unless the task explicitly requires it.
- Treat `data/`, `work_directory/`, evaluation outputs, and figures as generated artifacts. Preserve existing results and caches unless the task calls for regenerating or replacing them.
- Cache signatures and `methodology_version` protect expensive outputs from reuse after incompatible changes. Review and update invalidation when changing data construction, splits, decoding, training, or inference semantics.
- Keep children isolated across train, validation, and test splits to prevent child-identity leakage.

## Change guidance

- **Always update `docs/changelog.md` for a major feature or logic change.** Add a concise entry under the actual ISO date (`YYYY-MM-DD`) that says what changed and why. This includes changes to pipeline behavior, data handling, model training or inference, evaluation methodology, and outputs. Routine formatting, typo fixes, and internal refactors that do not change behavior do not need an entry.
- Keep `README.md` aligned with user-visible workflow, configuration, and output changes.
- Prefer focused changes that preserve the single-entry pipeline and existing artifact contracts. Avoid silently changing evaluation populations, metrics, split rules, or model settings.
- Do not launch expensive HPC or GPU jobs unless the task asks for them. For local validation, use the narrowest relevant checks available and report what was run.
