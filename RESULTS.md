# Results — Flan-T5 child speech correction

Generated from the current pipeline outputs. Counts and significance refer to this run only.

Test utterances: 14,918. Retained children for child-level analysis: 100.
Whisper 1-best WER: 40.51%. Whisper + Flan-T5 WER: 35.01%.

## Child-level results

| Group | Children | Whisper WER | Flan WER | Mean improvement |
|---|---:|---:|---:|---:|
| LT | 22 | 54.36% | 47.72% | +6.64 pp |
| TD | 78 | 42.40% | 37.81% | +4.59 pp |

Across retained children, mean paired improvement was +5.04 percentage points (paired t p=9.281e-16).
The group-by-model interaction was +2.05 percentage points (mixed-model Wald p=0.0998). Within-group improvements do not establish differential benefit.

## Corpus-matched sensitivity analysis

The shared corpora did not contain both groups in every corpus.

These observational corpus comparisons cannot isolate developmental group from recording and task differences. See `evaluation_results/analysis/anova_report.txt` for model terms, diagnostics, and corpus composition.
