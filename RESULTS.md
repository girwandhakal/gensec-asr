# Results — Flan-T5 child speech correction

Generated from the current pipeline outputs. Counts and significance refer to this run only.

Test utterances: 14,918. Retained children for child-level analysis: 100.
Whisper 1-best WER: 40.52%. Whisper + Flan-T5 WER: 34.41%.

## Child-level results

| Group | Children | Whisper WER | Flan WER | Mean improvement |
|---|---:|---:|---:|---:|
| LT | 22 | 54.42% | 45.91% | +8.50 pp |
| TD | 78 | 42.40% | 37.37% | +5.03 pp |

Across retained children, mean paired improvement was +5.79 percentage points (paired t p=7.634e-13).
The group-by-model interaction was +3.48 percentage points (mixed-model Wald p=0.0356). Within-group improvements do not establish differential benefit.

## Corpus-matched sensitivity analysis

The shared corpora did not contain both groups in every corpus.

These observational corpus comparisons cannot isolate developmental group from recording and task differences. See `evaluation_results/analysis/anova_report.txt` for model terms, diagnostics, and corpus composition.
