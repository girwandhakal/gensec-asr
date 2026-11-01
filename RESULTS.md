# Results — Flan-T5 child speech correction

Generated from the current pipeline outputs. Counts and significance refer to this run only.

Test utterances: 17,959. Retained children for child-level analysis: 127.
Whisper 1-best WER: 44.12%. Whisper + Flan-T5 WER: 39.20%.

## Child-level results

| Group | Children | Whisper WER | Flan WER | Mean improvement |
|---|---:|---:|---:|---:|
| LT | 22 | 54.75% | 48.77% | +5.98 pp |
| TD | 105 | 44.62% | 40.28% | +4.34 pp |

Across retained children, mean paired improvement was +4.62 percentage points (paired t p=3.26e-17).
The group-by-model interaction was +1.64 percentage points (mixed-model Wald p=0.1823). Within-group improvements do not establish differential benefit.

## Corpus-matched sensitivity analysis

Within ENNI, EllisWeismer, and Rescorla (79 children), the corpus-adjusted interaction was -0.69 percentage points (Wald p=0.5328); the within-corpus permutation p was 0.4157.

These observational corpus comparisons cannot isolate developmental group from recording and task differences. See `evaluation_results/analysis/anova_report.txt` for model terms, diagnostics, and corpus composition.
