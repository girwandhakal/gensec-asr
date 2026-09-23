# Results — per-child analysis

163 children (29 LT, 134 TD), 27,412 test utterances. WER accumulated per child
as total errors / total reference words. Source: `evaluation_results/analysis/`.

## Per-child WER

| Group | n | Whisper | Whisper+LLM | Improvement |
|---|---:|---:|---:|---:|
| LT | 29 | 68.77% | 64.46% | 4.31pp |
| TD | 134 | 57.54% | 50.34% | 7.19pp |

## Tests

One mixed-effects model (random intercept per child), fitted to all three
effects at once. `b` is the WER difference in proportion units.

| Effect | b | MixedLM p | Verdict |
|---|---:|---:|---|
| Model (Whisper → +LLM) | −0.0431 | **.023** | Correction lowers WER |
| Group (LT → TD) | −0.1123 | **.031** | LT worse under both models |
| Group × model | −0.0288 | .168 | Not significant |

The interaction coefficient (−0.0288) *is* the 2.88pp gap in improvement
between groups. Because those improvement scores are severely non-normal, it is
re-tested below without distributional assumptions; both routes agree.

Simple effects — correction helped both groups: LT p<.0001, dz=0.89;
TD p<.0001, dz=0.65 (Holm-corrected).

## The interaction

TD improves 2.88pp more than LT. Not statistically supported:

| Test | p | |
|---|---:|---|
| MixedLM (Wald) | .168 | model-based |
| permutation | .156 | ← reported, assumes nothing |
| Student t | .173 | |
| Mann-Whitney | .224 | |
| Yuen (20% trim) | .221 | |
| Welch t | **.031** | ← the only one under .05 |

MixedLM and the permutation test agree (.168, .156). Welch is the lone
dissenter.

TD improvements are severely non-normal (Shapiro p<10⁻¹⁵): per-child WER is a
small-denominator ratio with no upper bound. Welch clears .05 *because* of that
skew — its unequal-variance weighting gives the high-variance group extra
leverage. Reporting it would be choosing the test by its answer.

## Power

Hedges' g = 0.279. The current design has **27% power** — it would miss a real
effect this size three times in four.

| LT children | Power |
|---:|---:|
| 29 (current) | 0.27 |
| 60 | 0.43 |
| 100 | 0.56 |
| 150 | 0.65 |
| 200 | 0.70 |

80% power needs ~203 per group. Even all 140 LT children in the corpus falls
short.

## Conclusion

Correction significantly helps both groups. The LT/TD gap persists under both
models. TD gaining more is a **trend the data cannot confirm at n=29 LT**.

Note: `wer_report.txt` gives the LT/TD gap p<0.002, but that bootstrap resamples
*utterances* — treating 75 utterances from one child as 75 independent
observations. Per-child is the conservative unit, and at the child level the
interaction is not there.
