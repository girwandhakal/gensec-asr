# Results — child-level comparison

The held-out predictions contain 27,412 utterances from 167 children. The
analysis retains 163 children (29 in the combined LT/SLI group, labeled LT;
134 typically developing, labeled TD) with at least five test utterances each,
covering 27,400 utterances. Each
child's WER is its total word errors divided by its total reference words;
cell means give each retained child equal weight. These are absolute WER
percentage-point changes, not the relative reductions in `wer_report.txt`.

| Group | Children | Whisper WER | Whisper + LLM WER | Mean improvement |
|---|---:|---:|---:|---:|
| LT | 29 | 68.53% | 64.48% | 4.04 pp |
| TD | 134 | 55.74% | 49.72% | 6.02 pp |

## Factorial model and paired comparisons

The random-intercept mixed model includes group, model, and their interaction.
Its coefficient p-values are model-based Wald tests. With LT and Whisper as
reference levels, the group coefficient describes the **Whisper** LT–TD gap,
and the model coefficient describes the **LT** Whisper-to-LLM change. Neither
coefficient alone is a marginal effect across both levels.

| Coefficient | Estimate | Wald p | Meaning |
|---|---:|---:|---|
| TD vs LT under Whisper | −12.78 pp | .0109 | Lower mean WER for TD under Whisper |
| LLM vs Whisper within LT | −4.04 pp | .0075 | Lower mean WER after correction in LT |
| Group × model | −1.98 pp | .2354 | Additional TD improvement; uncertain |

Paired child-level tests of improvement within each group give LT
`p = 0.000982`, `dz = 0.68` and TD `p = 5.95e-13`, `dz = 0.70`
(Holm-adjusted across the two tests). These show average gains within each
group; they do not establish that the groups' gains differ.

## Interaction sensitivity

TD's observed mean improvement exceeds LT's by 1.98 pp. The current tests
give the following results:

| Test | p | Assumption or estimand |
|---|---:|---|
| Mixed-model Wald | .2354 | Random-intercept normal-error model |
| Unstudentized label permutation | .2347 | Exchangeable child improvement distributions |
| Student equal-variance t | .2400 | Equal variances |
| Welch unequal-variance t | .1410 | Difference in means, unequal variances |
| Mann–Whitney | .3387 | Rank-based distribution comparison |
| Yuen 20% trimmed | .5032 | Difference in trimmed means |
| Studentized permutation | .1438 | Welch statistic, 20,000 seeded draws |

The improvement standard deviations are 5.92 pp (LT) and 8.60 pp (TD).
TD's Shapiro p-value is `3.69e-11`. The unstudentized permutation requires
exchangeable group improvement distributions; unequal variances make that
assumption uncertain. None of the reported child-level interaction tests
crosses the .05 threshold in this run. The observed 1.98 pp difference is
therefore descriptive, without clear evidence of different gains by group.

## Scope of the comparison

All 29 LT children come from ENNI, Rescorla, or EllisWeismer. Only 42 of the
134 TD children come from those three corpora. Group differences therefore
also reflect corpus composition; this analysis cannot attribute them solely
to developmental group. Twelve retained test children also occur in the
training split through different source transcripts, even though no source
transcript or utterance ID overlaps. This limits claims about completely
unseen children.

The `wer_report.txt` group comparison is a different estimand: relative WER
reduction with utterance-level bootstrap resampling. Its small p-value does
not substitute for child-level uncertainty because utterances within a child
are correlated.

The old `anova_report.txt` power section used the observed effect size in a
two-sample equal-variance t-test calculation. That post-hoc calculation does
not establish that sample size caused the uncertain interaction and is no
longer part of the analysis report.
