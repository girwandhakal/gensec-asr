# GenSEC Pipeline — Implementation Plan

## Goal

Reproduce the GenSEC (Generative Speech Error Correction) flow that already
exists in `child-whispr-annotation`, pointed at the audio + reference
transcripts already prepared by `asr-dataset-pipelines`, and structured/run
the way the `Ecolang/Pose` repo is structured and run on the HPC cluster.

Keep the *implementation* as close to the reference GenSEC code as possible —
same stages, same file shapes, plain functions, no new abstractions. Keep the
*repo layout, config, environment, and Slurm conventions* identical to
`Ecolang/Pose`.

## What we're borrowing from where

**Algorithm / stage logic** — `../../../drivendata/child-whispr-annotation`:
a 3-stage pipeline —
`n_best_whisper_v3.py` (audio → n-best JSON, batched + resumable) →
`gensec_preprocessing.py` (n-best + references → supervised correction dataset) →
`gensec.py` (fine-tune FLAN-T5, zero/one/few-shot inference) →
`gensec_postprocessing.py` (repetition cleanup) →
`utils/calculate_wer.py` (corpus WER + exact-match).
Its own report (`gensec_flow_report.md`) documents two path-mismatch bugs
between stages; we avoid those by deriving every path from one config.

**Data** — the already-prepared clips in `asr-dataset-pipelines/data/media`.
See [The dataset this runs on](#the-dataset-this-runs-on) below for the exact
path, layout, and measured properties. **We do not touch audio acquisition** —
that pipeline is finished; this one only reads from that directory.

**Repo structure, config, envs, Slurm** — `../../Ecolang/Pose`, specifically
`acoustic_classifier/` (the simplest, most recent implementation there):
- one implementation directory containing everything it needs
  (`configs/`, `envs/`, `scripts/`, `bash_scripts/`, `data/`,
  `evaluation_results/`, `evaluation_history/`, `README.md`)
- `configs/baseline.yaml` holds every tunable value and every HPC path
- `scripts/config.py` loads that YAML and resolves paths, preferring a local
  copy when one exists so the same code runs on a laptop
- `scripts/run_pipeline.py` is the single entry point: numbered/timed stage
  banners, archives previous `evaluation_results/` into
  `evaluation_history/<timestamp>/`, skips stages whose output already exists
- `bash_scripts/*.sh` are Slurm wrappers: `#SBATCH` header, `set -euo pipefail`,
  `PROJECT_ROOT` env-overridable, `module load miniconda3/base`,
  `conda activate <env>`, dependency import check with pip repair, then
  `python -u .../run_pipeline.py`, logging to one truncated `*_training.txt`
- `envs/<name>_env.yml` (conda: python + system deps) + `envs/requirements.txt`
  (pinned pip wheels, CUDA index URL for torch)
- secrets live in a gitignored `.env.local` at repo root

## The dataset this runs on

This is the one and only input. Everything below is measured from it, not
assumed.

> **The counts below are a snapshot and are still moving.** During
> implementation the clip count rose from 55,262 to 58,270 and a new corpus
> (`EHS`) appeared, so the media pipeline was still downloading. The code never
> hardcodes these numbers — it globs the directory and intersects with the
> report every run — but re-read the stage-1 log for the real figures rather
> than trusting this table.

```text
C:\Users\g_dha\OneDrive - The University of Alabama\Uni\ml Research\ASR-Paper\Code\asr-dataset-pipelines\data\media
```

**Layout.** `<corpus>/<group>/<child_id>/<transcript_hash>_<start_ms>_<end_ms>_<ordinal>.mp3`,
e.g. `Braunwald/TD/51013/44e48e02_6768_8521_2.mp3`. Clip filename stem is the
utterance ID — the same `Path.stem` convention the reference repo uses. The
label file `media_download_report.csv` sits at the root of this same directory.

**Contents.** 55,262 mp3 clips, ~50.1 hours of audio, already mono 16 kHz
(the media pipeline encoded them that way, so Stage 2's resampling path is a
no-op safety net rather than real work):

| Corpus | Clips | Groups |
|---|---:|---|
| ENNI | 21,270 | LT, TD |
| MacWhinney | 12,438 | TD |
| Rescorla | 8,058 | LT, TD |
| HSLLD | 5,788 | TD |
| EllisWeismer | 4,161 | LT, TD |
| Gelman | 1,640 | TD |
| Braunwald | 1,007 | TD |
| Gleason | 596 | TD |
| POLER | 283 | TD |
| Forrester | 21 | TD |

Group split across the labelled clips is 43,560 TD / 11,302 LT. Since the
whole set is recursively globbed, this is corpus-agnostic — no per-corpus
branching anywhere in the pipeline.

**Measured properties that change the design:**

- **Only 54,862 of the 55,262 clips have a reference transcript.** The other
  400 are on disk but absent from `media_download_report.csv`, concentrated in
  HSLLD (278), Rescorla (93), EllisWeismer (8), and *all 21* Forrester clips.
  Stage 1 intersects disk against report and logs this rather than letting
  unlabelled clips reach training or scoring.
- **Clips are short: median 2.1 s, p90 6.0 s.** Good — Whisper's 30 s window
  comfortably fits the overwhelming majority, and short clips are why batching
  at 16 is worth doing.
- **91 clips exceed Whisper's 30-second window**, and the tail is extreme:
  the longest is **4,541 s (75 minutes)**, with four more over 13 minutes, all
  in Rescorla. These look like whole-session recordings that escaped
  segmentation. Whisper would silently transcribe only the first 30 s and be
  scored against the full session transcript — a guaranteed near-100% WER on
  those rows, plus a large chunk of decode time. Stage 2 skips clips longer
  than a configurable `max_clip_seconds` (30) and logs them.
- **11,262 clips are under 1 second**, 135 of them effectively zero-length
  (`0.00 s`). Sub-window clips are zero-padded rather than erroring (the
  reference implementation already does this), but the near-zero ones cannot
  contain speech; Stage 2 skips below `min_clip_seconds` (0.1).
- **Reference text is raw CHAT markup, not plain text** — 14,585 rows contain
  `[...]` codes, 9,648 contain `&`/`+` codes, 6,935 contain `<...>`, 1,936
  contain `xxx` (unintelligible). Stage 1 normalizes this once so the training
  target and the eval reference can never drift apart.

**Getting it onto the cluster.** The pipeline reads a directory, so the clips
have to exist on the HPC — ~50 hours of mp3 rsynced from this OneDrive path to
`hpc_media_dir`, with `media_download_report.csv` alongside them. `config.py`
resolves to the local path when it exists so the same code runs unchanged on
the laptop for smoke tests.

## Directory layout to create

```
gensec-asr/
  README.md                         # what the project is, how to run it on the cluster
  PLAN.md                           # this file
  .gitignore                        # Ecolang-style: ignore data/, checkpoints, logs; keep evaluation_*/
  .env.local                        # gitignored; HF_TOKEN etc.
  .env.example                      # committed; documents required keys, no values
  configs/
    baseline.yaml                   # all paths + all hyperparameters
  envs/
    gensec_env.yml                  # conda: python=3.10, pip, ffmpeg
    requirements.txt                # pinned: torch+cu121, transformers, datasets, soundfile, etc.
  bash_scripts/
    train.sh                        # THE entry point: sbatch this, it runs all six stages
  scripts/
    config.py                       # loads baseline.yaml, resolves local-vs-HPC paths
    build_reference_map.py          # Stage 1: media_download_report.csv -> {utt_id: reference text}
    generate_nbest.py               # Stage 2: audio -> n-best JSON  (port of n_best_whisper_v3.py)
    build_gensec_dataset.py         # Stage 3: n-best + references -> processed/dropped JSON
    train.py                        # Stage 4: fine-tune FLAN-T5 + ICL inference -> prediction CSVs
    postprocess.py                  # Stage 5: repetition/artifact cleanup
    evaluate.py                     # Stage 6: WER for 1-best vs. GenSEC vs. cleaned
    run_pipeline.py                 # runs all six stages in order; called by train.sh
  data/                             # generated, gitignored
    utterance_id_to_reference.json
    utterance_id_to_nbest.json
    processed_gensec.json
    dropped_gensec.json
    splits/train_split.csv, test_split.csv
    predictions/test_predictions_<mode>.csv, predictions_cleaned.csv
  evaluation_results/               # committed: wer_report.txt, metrics.json, config_used.yaml
  evaluation_history/<timestamp>/   # previous run, moved aside automatically
```

Checkpoints (`work_directory/`) are gitignored, like Ecolang's
`work_directory/` and `checkpoint-*/` rules.

## Configuration (`configs/baseline.yaml`)

One file, Ecolang style — HPC paths declared explicitly with a local
fallback resolved in `config.py`:

```yaml
seed: 42

# Audio + reference transcripts produced by asr-dataset-pipelines.
# The report CSV lives at the root of the media dir, so it is derived from it
# rather than declared twice.
hpc_media_dir: /bighome/gdhakal/asr-paper/data/media
local_media_dir: ../asr-dataset-pipelines/data/media   # relative to PROJECT_ROOT
reference_report_filename: media_download_report.csv

# ASR (n-best generation)
asr_model_id: rishabhjain16/whisper_medium_to_myst55h
audio_extension: .mp3
sample_rate: 16000
max_clip_seconds: 30      # Whisper's window; 91 clips exceed it, longest 75 min
min_clip_seconds: 0.1     # 135 clips are effectively zero-length
num_return_sequences: 5
temperature: 0.6
top_p: 0.95
top_k: 50
asr_batch_size: 16

# GenSEC correction model
gensec_model_id: google/flan-t5-base
max_source_length: 512
max_target_length: 128
num_train_epochs: 5
learning_rate: 3.0e-5
train_batch_size: 8
num_beams: 8
max_hypotheses: 5
min_hypotheses: 2
test_size: 0.2
inference_modes: [zero_shot]      # one_shot / few_shot switched on later

data_dir: data
work_dir: work_directory
results_dir: evaluation_results
```

`scripts/config.py` mirrors `acoustic_classifier/scripts/config.py`: resolve
repo-relative keys against `PROJECT_ROOT`, and use `_prefer_local(local, hpc)`
so `media_dir` resolves to `local_media_dir` when that directory exists
(laptop, the OneDrive path above) and to `hpc_media_dir` otherwise (cluster).
`reference_report_csv` is then just `media_dir / reference_report_filename`, so
the audio and its labels can never be resolved to two different copies of the
dataset.

## Environment (`envs/`)

`gensec_env.yml`:
```yaml
name: gensec_env
channels: [conda-forge, defaults]
dependencies:
  - python=3.10
  - pip
  - ffmpeg          # needed to decode the media pipeline's mp3 clips
  - pip:
      - -r requirements.txt
```

`requirements.txt` (pinned, CUDA wheels like Ecolang's `deit_classifier`):
```
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.4.1+cu121
torchaudio==2.4.1+cu121
transformers==4.57.6
datasets
accelerate
sentencepiece
soundfile
librosa
pandas
scikit-learn
PyYAML==6.0.2
```

Created once on the cluster with
`conda env create -f envs/gensec_env.yml`; both Slurm scripts check the
imports and pip-repair if anything is missing, exactly like Ecolang's
`train.sh` does.

## The single entry point (`bash_scripts/train.sh`)

**One command runs everything. Nothing is ever run by hand.**

```bash
sbatch bash_scripts/train.sh
```

That job does, in order: build the reference map → generate Whisper n-best
over every clip → build the GenSEC dataset → fine-tune FLAN-T5 → run
inference → postprocess → score WER → write `evaluation_results/`.

The script follows the Ecolang `train.sh` conventions exactly:

- `#SBATCH --partition=gpu --qos=gpu --gres=gpu:1 --nodes=1 --ntasks=1
  --cpus-per-task=8 --mem=64G`, `--open-mode=truncate`, both streams to
  `gensec_training.txt`
- `set -euo pipefail`; `PROJECT_ROOT="${GENSEC_PROJECT_ROOT:-/home/gdhakal/gensec-asr}"`
- `tee` fallback so a plain `bash bash_scripts/train.sh` logs identically
- pre-flight checks that fail fast with a readable message: config file
  present, media dir non-empty, `media_download_report.csv` present
- `module load miniconda3/base`, `set +u` around `conda activate gensec_env`,
  `set -u`, `export PYTHONPATH="$PROJECT_ROOT"`, `cd "$PROJECT_ROOT"`
- dependency check (`python -c "import torch, transformers, datasets,
  soundfile, pandas, yaml"`) with a pinned-wheel pip repair on failure —
  so the first submission also *creates* a working environment rather than
  needing a manual install step
- then the one line that does the work:
  `python -u "$PROJECT_ROOT/scripts/run_pipeline.py"`

**Walltime and resumability.** The whole thing in one job means the n-best
stage (54,862 labelled clips, ~50 h of audio, through Whisper-medium at 5
sampled hypotheses each) and the fine-tune share a walltime;
`--time=48:00:00` is the starting request. Every stage is resumable and
skipped when its output already exists — n-best generation resumes from the
existing JSON and checkpoints it every `print_every` clips, and stages 3–6
short-circuit on an existing output file. So if the job hits the walltime,
the fix is to `sbatch` the same script again and it picks up where it stopped.
Still one command, still no manual steps.

Individual `scripts/*.py` remain runnable standalone (each has a
`__main__` block reading the same `baseline.yaml`) purely for debugging —
that is not the normal path.

## Stage-by-stage plan

### Stage 1 — `build_reference_map.py`
Read `media_download_report.csv`; keep rows with a non-empty `output` whose
file exists on disk (87,472 of the 142,334 rows are `status=skipped` with no
clip, and 400 clips on disk have no row — both directions are counted and
logged); key by `Path(output).stem`; normalize the `utterance` CHAT markup to
plain lowercase text; drop the ~33 entries left with no alphabetic content.
Write `data/utterance_id_to_reference.json` (flat `{id: text}`) and print
kept / no-clip / orphan-clip / empty-after-cleaning counts.

This is the supervision signal for everything downstream: the `output` target
in Stage 3's training examples and the reference for every WER number in
Stage 6. It replaces the CHILDES-specific transcript plumbing
(`train_word_transcripts.jsonl`) in the reference repo. Note that
`data/processed/all_utterances_clean.csv` has cleaner text but **no timestamp
columns**, so it cannot be joined back to clip filenames — the report CSV is
the only artifact carrying the clip↔text link, which is why the cleaning is
redone here.

### Stage 2 — `generate_nbest.py`
Direct port of `n_best_whisper_v3.py`: batched, resumable, per-file
skip-on-failure, mono/16 kHz normalization before batching, whisper control
tokens stripped, per-utterance dedup with a fallback to non-unique candidates
when fewer than 5 unique ones come back. Changes only:
- `rglob("*" + config["audio_extension"])` — `.mp3`, not the reference repo's
  hardcoded `*.flac`. Recursive, so it picks up every
  `<corpus>/<group>/<child_id>/` clip with no per-corpus logic.
- model ID, decode settings, and batch size read from `baseline.yaml`
- input dir = resolved `media_dir`, output = `data/utterance_id_to_nbest.json`
- **duration guard**: skip clips outside
  `[min_clip_seconds, max_clip_seconds]`, counted and logged. This is new
  relative to the reference implementation and specific to this dataset — the
  91 over-30 s clips would otherwise be transcribed from only their first 30 s
  and scored against a full-session transcript, and the 75-minute Rescorla clip
  alone would burn a large share of the decode budget.
- clips shorter than the FFT window are zero-padded rather than erroring, as
  in the reference implementation

Output shape unchanged: `{utt_id: {"nbest": [{rank, text, score?}], "1best_text": "..."}}`.

### Stage 3 — `build_gensec_dataset.py`
Same logic as `gensec_preprocessing.py`: `1best_text` first, then deduped
stripped `nbest` texts, truncated to `max_hypotheses` (5). Drop when the id is
missing from the reference map, the reference is empty, or fewer than
`min_hypotheses` (2) unique hypotheses remain — every drop logged with its
reason to `data/dropped_gensec.json`. Output `data/processed_gensec.json` rows:
`{id, input: [...], output: "..."}` — identical schema to the reference repo.

### Stage 4 — `train.py`
Port of `gensec.py`: normalize (lowercase, collapse whitespace), re-drop
invalid rows, `build_nbest_prompt(...)` with the same instruction template,
80/20 split at `seed`, save `data/splits/{train,test}_split.csv`, fine-tune
`google/flan-t5-base` with `Seq2SeqTrainer` at the config hyperparameters,
checkpoints under `work_directory/`. Then `build_icl_prompt(...)` for
each mode in `inference_modes`, with the same 512-token trimming order (drop
demonstrations from the end first, then trim hypotheses down to 2). Writes
`data/predictions/test_predictions_<mode>.csv` with `id, input,
input_token_count, retained_icl_examples, retained_input_hypotheses, truth,
prediction`.

### Stage 5 — `postprocess.py`
Same cleanup, same order: whitespace normalize → small manual replacement
dict → collapse repeated 3–8 word phrases → collapse single-word repetition
spam → whitespace normalize. Writes `data/predictions/predictions_cleaned.csv`.

### Stage 6 — `evaluate.py`
Port of `utils/calculate_wer.py` (same normalization + edit-distance WER with
S/D/I breakdown and exact-match rate), but scoring all three systems in one
pass against the reference map:
1. raw Whisper 1-best
2. GenSEC prediction (per inference mode)
3. postprocessed GenSEC
Writes `evaluation_results/wer_report.txt` (human-readable table + worst
utterances) and `evaluation_results/metrics.json`, plus a copy of the config
used — the Ecolang `evaluation_results/` convention, with the previous run
archived into `evaluation_history/<timestamp>/` by `run_pipeline.py`.

### `run_pipeline.py`
Ecolang's `run_pipeline.py` pattern verbatim, covering **all six stages**:
`load_config()`, validate inputs, archive previous results into
`evaluation_history/<timestamp>/`, then numbered/timed `stage(n, "NAME")`
context managers (`===== STAGE n/6: ... =====` banners with per-stage
elapsed time) over stages 1→6. Each stage skips its work and prints
"using existing ..." when its output file is already there, which is what
makes a resubmitted job resume rather than restart. This is the only script
`train.sh` calls.

## What we deliberately skip for v1

- No new prompting modes, reranking, or multi-ASR ensembling.
- No ablation registry / config sweep — add once there's a baseline WER.
- Only `zero_shot` inference enabled by default (matches the reference
  default); `one_shot` / `few_shot` are a one-line config change afterwards.
- No audio acquisition — media root is a config path.
- WER + exact-match only; no CER/BLEU/semantic metrics yet.

## Order of implementation

1. Repo skeleton: `.gitignore`, `.env.example`, `configs/baseline.yaml`,
   `envs/`, `scripts/config.py`. Verify `config.py` resolves the local media
   paths correctly off-cluster.
2. `build_reference_map.py` — expect ~54,829 kept (54,862 labelled clips minus
   the ~33 empty-after-cleaning); spot-check a few id→transcript pairs and
   confirm the CHAT markup is gone.
3. `generate_nbest.py` — smoke-test on a small subset locally (`limit` in the
   config) to confirm the model loads and the JSON shape matches.
4. `build_gensec_dataset.py` — inspect kept/dropped counts and a few rows by
   hand.
5. `train.py` — short run on a subset first, then the full fine-tune.
6. `postprocess.py` + `evaluate.py` — confirm WER moves in the right direction
   across the three systems.
7. `run_pipeline.py` + `train.sh` wired last, once each stage works; from then
   on `sbatch bash_scripts/train.sh` is the only command used.
8. `README.md` written up Ecolang-style once numbers exist.

## Open items to confirm

- **HPC paths and getting the data there.** `hpc_media_dir` is written as
  `/bighome/gdhakal/asr-paper/data/media` and `PROJECT_ROOT` as
  `/home/gdhakal/gensec-asr` by analogy with the Ecolang setup
  (`/home/gdhakal/<repo>` + `/bighome/gdhakal/<project>/data`) — both need
  confirming against the actual cluster layout. The 55,262 clips (~50 h of
  mp3) plus `media_download_report.csv` need rsyncing from the OneDrive path
  to `hpc_media_dir` before the first submission; worth confirming the
  `/bighome` quota has room.
- **The 400 orphan clips.** On disk with no row in the report — HSLLD 278,
  Rescorla 93, EllisWeismer 8, and all 21 Forrester clips. Forrester being
  *entirely* absent from the report suggests the report was written before
  that corpus was downloaded, rather than random loss. Worth a look: if the
  report is simply stale, regenerating it recovers 400 labelled utterances;
  otherwise they're ignored.
- **The 91 over-length clips.** Plan is to skip them. If those Rescorla
  session-length files are actually a segmentation bug worth fixing upstream,
  that's a media-pipeline change, not a change here.
- **Slurm partition/qos/time.** Copied from Ecolang (`gpu`/`gpu`); confirm the
  same are right for this account, and what the max walltime is — the plan
  requests 48 h for the combined n-best + training job. If the cap is lower,
  nothing changes structurally: resubmit the same script and it resumes.
- **`.env.local` contents.** Does anything here need a secret? Likely just
  `HF_TOKEN` if the Whisper checkpoint or any model becomes gated; the
  `.env.example` will document that and nothing else.
