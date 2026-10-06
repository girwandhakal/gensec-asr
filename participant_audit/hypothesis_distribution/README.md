# Whisper hypothesis distribution

Frequency of retained distinct, nonempty Whisper hypotheses in each utterance's nbest list, grouped by TD and CD. Each utterance contributes to exactly one bin (1–5). The separate 1best_text baseline field is not counted again. These are retained candidates, not the number of raw sampling attempts. CD pools metadata labels LT, CD, and SLI; the current data uses LT. Panels use separate y-axis scales. All generated utterances with recognized group metadata and at least one nonempty candidate are included, across dataset splits.

Hypothesis source: `C:\Users\g_dha\OneDrive - The University of Alabama\Uni\ml Research\ASR-Paper\Code\gensec-asr\data\utterance_id_to_nbest_greedy_sample10.json`
Group source: `C:\Users\g_dha\OneDrive - The University of Alabama\Uni\ml Research\ASR-Paper\Code\gensec-asr\data\utterance_metadata.json`

| Hypotheses per utterance | TD frequency | CD frequency |
|---:|---:|---:|
| 1 | 38,462 | 2,393 |
| 2 | 22,105 | 1,671 |
| 3 | 15,086 | 1,284 |
| 4 | 11,820 | 1,024 |
| 5 | 38,339 | 4,819 |
| Total utterances | 125,812 | 11,191 |

Input utterances: 139,348. Excluded: {'missing_metadata': 2345}.

Suggested caption: Distribution of the number of distinct retained Whisper hypotheses per utterance for typically developing (TD) children and children with communication disorders (CD). The x-axis indicates the number of hypotheses (1–5), and the y-axis indicates the number of utterances with that many hypotheses. Panels use separate y-axis scales.
