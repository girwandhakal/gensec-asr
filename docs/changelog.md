# Changelog

## 2026-11-02

- **Bound FLAN-T5 generation length by input hypotheses.**
  - **Empirically derived hypothesis length cap:** Implemented `hypothesis_length_cap` bounding output generation to `1.5 * longest_hyp_tokens + 4` (capped at `max_target_length: 128`).
  - **Justification for dual-parameter bound ($1.5 \times |H_{\text{max}}| + 4$):**
    - **$1.5\times$ Multiplier (Slack):** Scales proportionally with sequence length to accommodate subword tokenization splits, morphological expansion, and legitimate syntactic rephrasings on longer utterances.
    - **$+4$ Tokens Headroom (Additive Floor):** A pure multiplier alone ($1.5 \times |H|$) mathematically starves short utterances: a 1-token hypothesis ($|H|=1$) yields only $1 \times 1.5 \approx 1$ token, making it impossible for the corrector to recover words lost to Whisper deletion errors (e.g., child said *"want more juice"*, but Whisper only caught *"juice"*). The $+4$ token additive headroom provides an indispensable floor for short utterances.
    - **Strict Training-Set Derivation:** Both parameters were derived strictly on `train_split.csv` ($N = 59,508$), where the 99th percentile of $(L_{\text{ref}} - L_{\text{hyp\_max}})$ is $+3$ words / $+6$ tokens without slack, and $1.5 \times |H_{\text{max}}| + 4$ covers 99.3% of all reference length expansions without audio duration leakage.
  - **Text-conditioned sequence batching:** Replaced audio-duration (`clip_seconds`) sorting in `run_inference()` with hypothesis sequence length sorting (`max(len(h.split()) for h in e["input"])`). This eliminates the last vestige of acoustic duration metadata during inference, minimizes attention padding, and ensures short utterances are grouped together without their batch token cap being lifted by longer neighbors.
  - **Why:** Eliminates degenerate beam-search repetition loops on brief, noisy, or ambiguous speech (which caused a catastrophic insertion spike) while strictly adhering to text-only modality (avoiding audio duration leakage) and preserving authentic child speech repetitions without hard n-gram bans.
  - **Updated methodology version:** Advanced to `child_split_v8_hypothesis_length_cap` to invalidate predictions generated under the unconstrained ceiling.

- **Resume legacy Whisper caches after batch-size signature tracking.** Upgrade
  signatures missing `asr_batch_size` using the historical recorded size of 4
  only when the current size is 4 and all other decoding settings match. Preserve
  cached transcripts and raw samples while continuing to reject actual decoding
  changes. Legacy runs using unrecorded batch-size overrides require regeneration.

## 2026-11-01

- **Standardize FLAN-T5 generation to pure Seq2Seq defaults.**
  - **Removed audio-duration output token clamping (`generation_cap`):** Eliminated duration-based token limiting (`clip_seconds * 8`) and hypothesis slack multipliers (`generation_length_slack: 1.5`, `generation_min_tokens: 6`) during inference. FLAN-T5 now generates up to standard `max_target_length: 128` using natural end-of-sequence (`</s>`) termination.
  - **Disabled decoding repetition penalties:** Set `no_repeat_ngram_size: 0` and `repetition_penalty: 1.0` by default. This eliminates artificial suppression of genuine child speech disfluencies (e.g., emphatic and developmental repetitions like "no no no" or "and and and").
  - **Why:** Eliminates inadmissible audio-duration modality leakage into the text-only corrector, prevents an unfair evaluation double standard against Whisper, and aligns inference with standard, reproducible Hugging Face Seq2Seq generation practices.
  - **Updated methodology version:** Advanced to `child_split_v6_standard_seq2seq` to invalidate previous predictions generated under the ad-hoc token cap.

- **Exclude target-child utterances with overlapping adult speech.** Exclude target-child utterances with at least 100 ms of
  timestamp overlap with a non-target CHAT speaker labeled as an adult before
  MLU, dataset capping, balancing, and GenSec train/validation/test splitting.
  This prevents overlapping adult speech from contaminating child-speech
  references and ASR training/evaluation. Record the overlap speaker and
  duration in the upstream utterance-exclusion report.

- **Retain long clips with brief references.** Removed the over-five-second,
  at-most-one-word exclusion from upstream CHAT/manifest screening and GenSEC
  reference/dataset screening because these clips may contain valid speech
  surrounded by silence. Removed its configuration setting and advanced the
  methodology version to invalidate downstream training and prediction caches.
  Existing generated datasets and results are unchanged; upstream preprocessing
  and downstream evaluation must be rerun before reporting revised counts.

- **Invalidate decoding caches when batch sizes change.** Batch size affects duration-based token limits, so changed Whisper or FLAN decoding batches now regenerate their outputs.
- **Refreshed evaluation results and figures.** Archived the October 31 run and updated current WER and child- and corpus-level analyses.

## 2026-10-31

- **Restored the Flan-only pipeline.** Removed the temporary Qwen comparison and returned HPC runs to the single-entry workflow, adding reference-alignment screening and validation-selected checkpoints to address data quality and model-selection concerns.
- **Improved evaluation resilience.** Kept figures usable with partial Whisper sample caches and clarified prompts for short or single-hypothesis inputs.

## 2026-10-29–30

- **Added a temporary Qwen3 comparison.** Evaluated Flan and Qwen against shared Whisper outputs and added candidate-diversity figures to compare correction models on the same split.
- **Improved recovery and resource use.** Reused compatible Whisper outputs, safely restarted incompatible Qwen partials, bounded checkpoint storage, reduced provisional Qwen training to five epochs, and improved stage logging and environment setup.

## 2026-10-18

- **Held out children across data splits** to prevent child identity leakage and evaluate performance on unseen children.
- **Removed optional participant-audit outputs** that were not needed by the main analysis workflow.

## 2026-10-05–06

- **Standardized evaluation inputs.** Used finalized upstream references and saved complete Whisper sample pools before deduplication for consistent scoring and diversity analysis.
- **Removed explicit consensus evidence from prompts** and pinned Slurm training to H100 to restore BF16 throughput.

## 2026-09-28–29

- **Expanded training to 40 epochs** and stopped retaining every Trainer checkpoint to limit work-directory growth.

## 2026-09-22–23

- **Expanded child- and group-level analysis.** Added per-child WER, mixed-model comparisons, and a greedy ASR baseline to better characterize performance variation and correction gains.
- **Removed selective correction gating** so evaluation measured the correction model consistently.

## 2026-09-14

- **Added clip-duration WER breakdowns** to show how correction performance varies with utterance length.
- **Reverted experimental decoding and checkpoint changes** that were not retained in the working configuration.

## 2026-09-06–08

- **Added consensus-aware selective evaluation** to measure when correction helps, harms, or should be skipped; the gate was later removed as the analysis evolved.
- **Fixed train/test leakage and stale-cache reuse.** Held out whole transcripts, checked for overlap, and signed decoding and prediction artifacts so changed inputs or settings could not silently reuse stale outputs.
- **Deleted repetition-cleaning postprocessing.** Generation-time controls addressed loops, while the cleaner also removed legitimate child repetitions and increased net errors.
- **Fixed a stage-four crash** caused by name shadowing after inference.

## 2026-09-02–05

- **Improved text and decoding cleanup.** Addressed hallucination loops and mojibake and corrected identity mapping used by WER optimization.
- **Made HPC runs more reliable** by moving the project off an unreliable autofs path and fixing training paths and Slurm GPU selection.

## 2026-09-01

- **Invalidated stale training outputs** when the training split changed, ensuring checkpoints and predictions matched their data.

## 2026-08-28–31

- **Stabilized Whisper candidate generation.** Fixed beam-search collapse, enabled SDPA attention, repaired stale resume entries, and returned to sampling after full-scale beam search repeatedly collapsed.
- **Bound decoding cost and memory** with duration and decode-budget limits and by dropping beam scores that caused out-of-memory failures.

## 2026-08-27

- **Expanded evaluation diagnostics** with oracle bounds, LT/TD and reference-length breakdowns, and bootstrap intervals; corrected the interval calculation to use paired differences.
- **Made evaluations recoverable** by archiving outputs by run and preserving them on crashes.

## 2026-08-26

- **Hardened training and inference on HPC.** Matched fine-tuning and zero-shot prompts, enabled guarded BF16 training, made inference recoverable after crashes, and corrected the cluster GPU resource name.

## 2026-08-25

- **Created the initial end-to-end pipeline.** Added reference mapping, Whisper n-best generation, FLAN-T5 training, evaluation, postprocessing, configuration, and an HPC entry point; fixed media and report paths for cluster runs.
