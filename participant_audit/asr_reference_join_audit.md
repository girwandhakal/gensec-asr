# ASR reference-join audit — 2026-10-06

Read-only audit of saved hypotheses, reference map, processed correction dataset,
drop log, current media report, timestamp manifest, and pipeline source. No
production dataset, model, split, or evaluation was regenerated.

## Verified counts

| Artifact or condition | Count |
| --- | ---: |
| Saved greedy-plus-sampling Whisper records | 139,348 |
| Saved nonempty cleaned human references | 140,268 |
| Final correction examples | 137,003 |
| Dropped Whisper records | 2,345 |
| Drops logged as `no_reference_transcript` | 2,345 |
| Dropped records with report references empty after CHAT cleaning | 1,947 |
| Dropped records with no current report row | 398 |
| Records with an empty stored greedy string | 0 |
| Records with no nonempty candidate string | 0 |
| References without a saved Whisper record | 3,265 |

The processed ID set equals the intersection of hypothesis and reference IDs.
The dropped ID set equals hypothesis IDs minus reference IDs. Every saved
reference matches normalization of its current media-report utterance; every
processed target matches the saved reference. No duplicate report clip IDs were
found. Exclusions are 1.683% of the 139,348 Whisper records.

## Actual order and explanation

`run_pipeline.py` builds the cleaned reference map first, then transcribes audio,
then joins hypotheses to references. Reference construction omits empty cleaned
references. Whisper generation independently scans all MP3s under the media
directory, rather than restricting itself to IDs with usable reference pairs.
Consequently, clips rejected during reference preparation still receive Whisper
predictions and are later dropped at the join. Empty human references are logged
as missing references because their IDs have already been omitted from the map.
None of the 2,345 exclusions was caused by all Whisper candidates being empty.

## Findings requiring attention

1. **Definite normalization bug:** `EVENT_CODE_RE = r"[&+]\S+"` also matches a
   plus sign inside a spoken compound. `snow+man` becomes `snow`, and `hot+dog`
   becomes `hot`. There are 27 affected utterances among saved Whisper records;
   all 27 are retained correction examples. Separate utterance/event marker
   handling from within-word compound normalization. Changing references later
   requires deliberate regeneration of dependent artifacts.

2. **Filler policy needs an explicit decision:** the same regex deletes whole
   `&-um`, `&-uh`, and similar filled-pause tokens. Of the 1,947 empty-reference
   exclusions, 985 would become nonempty if filler prefixes were removed while
   their content was preserved. Another 9,380 retained utterances contain fillers
   removed from their targets. This is a policy mismatch if the intended target
   is a verbatim transcript; it can be intentional for a lexical-only target,
   but must be documented and reconciled with prediction scoring. CHAT describes
   `&-` as filled-pause notation, distinct from events such as `&=laughs`.

3. **Unpaired audio scanned unnecessarily:** all 398 clips lacking report rows
   exist on disk. They span 12 source hashes, all absent from current report
   rows, and none matches an interval in the current timestamp manifest. Disk
   folders are TD: Rescorla/big60 (93), HSLLD/aprmt5 (16), HSLLD/braer2 (25),
   HSLLD/brtmt3 (22), HSLLD/doner2 (2), HSLLD/dontp2 (53), HSLLD/jermt3 (69),
   HSLLD/monbr2 (9), HSLLD/rasbr1 (26), HSLLD/rembr2 (56), Forrester/realfood
   (21), and EllisWeismer/11051 (6). Leftover audio from an earlier selection is
   a plausible explanation, not proven provenance. Do not invent references or
   reintroduce these clips without verifying their selection and source labels.

4. **Drop reasons obscure provenance:** the drop log collapses empty cleaned
   human references and truly missing report rows into `no_reference_transcript`.
   Preserve separate reference-preparation reasons and use the eligible ID set
   when scheduling Whisper generation in future runs.

5. **Misleading code documentation:** the dataset-builder module header says
   agreeing candidates are dropped, but configured `min_hypotheses: 1` retains
   single-candidate examples. The generator header also incorrectly states that
   the corrector cannot recover words absent from candidates; its output is not
   constrained to the candidate vocabulary. These comments are not evidence of
   actual processing behavior.

There are also 180 punctuation-only stored greedy strings (empty after scoring
normalization), including 175 retained examples. These were not the cause of the
2,345 drops; their scoring is separate from empty raw candidate filtering.

## Accurate manuscript wording for current artifacts

Whisper produced hypothesis records for 139,348 audio clips. Matching these
records to the cleaned human references yielded 137,003 correction examples.
Of the 2,345 excluded records, 1,947 had human transcripts that became empty
under the reference-cleaning rules, and 398 had no corresponding row in the
current media report. These exclusions were not caused by empty Whisper outputs.

Avoid describing all 1,947 as containing no speech: some contain transcribed
filled pauses that the current cleaner removes.

## Source

CHAT notation: https://talkbank.org/0info/manuals/CHAT.html

The 3,265 reference IDs without hypotheses were counted but not attributed to
decoded-duration exclusions versus ASR failures: this requires per-clip generation
logs or an additional audio audit. Timestamp durations alone do not establish the
decoded durations used by `generate_nbest.py`.
