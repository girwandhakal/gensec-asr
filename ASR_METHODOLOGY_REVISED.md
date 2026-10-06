# ASR

We used a multi-stage pipeline to transcribe the target-child audio clips and correct recognition errors. A Whisper model adapted to child speech generated an initial transcript and alternative transcriptions for each clip. We paired these hypotheses with the corresponding human CHILDES transcript to train a FLAN-T5 sequence-to-sequence correction model. On a held-out test set, we compared the uncorrected Whisper transcript and the corrected transcript against the same human reference using word error rate (WER).

## 1. Data Preparation

This stage began with the target-child audio clips and aligned CHILDES transcripts produced by the separate TalkBank data and audio-clipping procedure described earlier in the methodology. We matched each available clip to the media report using its utterance identifier. The report also carried the clip's corpus, LT or TD group, and child identifier. A report row without an output clip, a row whose output file was missing, or a clip without a matching report row could not provide a usable audio–reference pair.

The aligned transcript still contained CHAT markup, so we converted it to the plain-text reference used for training and evaluation. We first applied Unicode NFKC normalization, repaired common character-encoding errors, and converted curly apostrophes to straight apostrophes. We removed bracketed annotation codes, including nested codes; event and utterance-terminator tokens beginning with `&` or `+`; special-form tokens beginning with `@`; unspoken forms beginning with `0`; and the unintelligibility markers `xxx`, `yyy`, and `www`. Words inside angle brackets were retained while the brackets were removed. Likewise, the letters inside parentheses were retained, so a form such as `(be)cause` became `because`. Underscores were changed to spaces. Finally, we lowercased the text, replaced punctuation other than apostrophes with spaces, and collapsed repeated whitespace. We did not use an empty cleaned transcript as a correction target. The same cleaned human transcript served as the fine-tuning target and the reference for final scoring.

## 2. Whisper Transcription

We decoded each clip to mono audio at 16 kHz. Clips shorter than 0.1 seconds or longer than 30 seconds, as well as files that could not be decoded, were not transcribed. We used the child-adapted Whisper checkpoint `rishabhjain16/whisper_medium_to_myst55h` with English transcription specified during generation. For each eligible clip, we first generated one transcript with greedy decoding (`num_beams=1`, `do_sample=False`). This one-best transcript was the uncorrected ASR baseline. In a separate pass, we drew 10 sampled transcriptions of the same clip (`do_sample=True`, `num_beams=1`, `num_return_sequences=10`, `temperature=0.8`, `top_p=0.95`, `top_k=50`). The PyTorch random seed was 42. Clips were processed in batches of four; the maximum number of new Whisper tokens for each batch was `floor(8 × the longest clip duration in seconds) + 16`.

After decoding, we removed Whisper control tokens, repaired encoding errors, and collapsed extra spaces. We identified duplicate sampled outputs using a comparison form that was Unicode-normalized, lowercased, stripped of punctuation other than apostrophes, normalized for equivalent nasal-hum spellings, and reduced to single spaces. The actual retained candidate text was not stripped of ordinary punctuation at this point. The greedy transcript was kept first whenever it was nonempty. We then retained up to four distinct sampled alternatives. When there were more than four, we prioritized candidates by how often they occurred in the 10 draws; ties favored a candidate introducing at least one word absent from those already selected, followed by the candidate drawn earlier. Before model training, candidate text was lowercased and candidates that then became identical were removed again. We retained clips with only one usable hypothesis, so an example had between one and five hypotheses without duplicate padding.

Whisper produced hypothesis records for 139,348 clips. We joined these records to the cleaned references by utterance identifier. The resulting correction dataset contained 137,003 examples, each with an ordered hypothesis list and a nonempty human reference. Of the 2,345 excluded hypothesis records, 1,947 had a report transcript that contained no spoken words after CHAT codes, unintelligibility markers, and punctuation were removed; the other 398 clip identifiers had no corresponding output row in the current media report, so no human transcript could be retrieved for them. For illustration, an input might list `i want red ball`, `i want the red ball`, and `i want the red doll`, with `i want the red ball` as the target. This is an illustrative format, not a quoted dataset row.

## 3. Train / Test split

We divided the 137,003 correction examples into training and held-out test sets by source recording transcript rather than by individual utterance. The source transcript was identified from the prefix of each clip identifier. Within the LT and TD groups, transcript identifiers were sorted and shuffled using seed 42, and whole transcripts were assigned to the test set until their utterance count was closest to 20% of that group's total. We checked that no source transcript appeared on both sides. This yielded 109,591 training-side examples and 27,412 test examples. Because some children had multiple recording sessions, 12 children appeared in both partitions; the test set therefore represents unseen sessions, not entirely unseen children.

We further reserved 2,192 training-side examples, or 2%, for validation during fine-tuning. This validation selection was made at the example level with seed 42, leaving 107,399 examples for weight updates. The held-out test examples were used for final inference and evaluation.

## 4. Finetuning

We fine-tuned `google/flan-t5-base` to generate the cleaned human transcript from the Whisper hypotheses. Each input began with an instruction and then listed only the hypotheses available for that utterance, with the greedy one-best transcript first. The prompt had the following form:

```text
You are correcting ASR output.
Below are multiple recognition hypotheses for the same utterance.
Choose the single most likely correct transcript.
Do not paraphrase.
Do not add information.
Preserve the original wording as much as possible.
Output only the corrected transcript.

Hypotheses:
1. <greedy one-best transcript>
2. <sampled hypothesis, if available>
...
Consensus evidence (word support): <word> (<count>/<available hypotheses>), ...
```

The consensus line showed how many distinct hypotheses contained each lowercased, whitespace-separated word. A repeated word within one hypothesis counted only once. Words supported by at least two hypotheses were listed in descending support order, with alphabetical order breaking ties, up to a maximum of 40 words. If only one hypothesis was available, its words were listed with support of `1/1`. The denominator was always the number of hypotheses actually present. The prompt did not add blank candidate lines to make every example contain five hypotheses, and it did not include the human reference.

The model learned the mapping **Whisper hypotheses → cleaned human reference transcript**. Source prompts were truncated to 512 tokenizer tokens, and reference targets to 128 tokens. Training used 10 epochs, a learning rate of `3 × 10⁻⁵`, a per-device training batch size of 8, a per-device validation batch size of 16, and seed 42. Validation loss was measured and a checkpoint saved after each epoch. We used the model saved at the end of epoch 10 for test inference, without selecting an earlier checkpoint based on validation loss. Bfloat16 training was enabled when the CUDA device supported it.

## 5. Inference

We applied the fine-tuned FLAN-T5 model to the 27,412 held-out test utterances. Each test prompt used the same instruction, numbered hypotheses, and consensus format as the training prompts, without example corrections. If a prompt exceeded the 512-token input limit, trailing hypotheses were removed until it fit or only one remained. This was necessary for six test prompts. We sorted test utterances by clip duration and generated one correction per utterance in batches of four. Generation was deterministic, with eight-beam decoding and no sampling. We blocked repeated three-token sequences (`no_repeat_ngram_size=3`) and applied a repetition penalty of 1.2.

We also bounded generated length. For each batch, the maximum number of new tokens was at least six and otherwise the smallest of three limits: 128 tokens; `floor(1.5 × the token count of the longest input hypothesis in the batch) + 5`; and `floor(8 × the longest clip duration in seconds in the batch)`. These limits used both the available hypothesis text and the length of the underlying audio.

## 6. Postprocessing and Evaluation

After generation, we collapsed extra whitespace. We also limited each prediction to at most `max(2, ceil(4 × clip duration in seconds) + 1)` words, truncating any words beyond that bound. We did not run a separate cleanup step to delete repeated words or phrases from the output, since repetition can occur in child speech. The generation settings described above nevertheless constrained repeated three-token sequences.

We compared the greedy Whisper transcript and the FLAN-T5 correction against the same cleaned reference for each of the 27,412 test utterances. Before scoring, we applied the same normalization to reference and system output: Unicode NFKC normalization, repair of common encoding errors, removal of Whisper control tokens, lowercasing, removal of punctuation other than apostrophes, and collapse of repeated spaces. We also mapped the nasal-hum spellings `mm`, `mmm`, `mhm`, `mmhm`, `mmhmm`, `hm`, `hmm`, `hmhm`, and `mhmm` to `mm` on both sides. We did not merge `uhhuh` and `uhuh`, which may have different meanings.

Word-level alignment counted substitutions (`S`), deletions (`D`), and insertions (`I`). We calculated corpus-level word error rate as **WER = (S + D + I) / N**, where `N` was the total number of reference words across all test utterances. We also computed normalized exact-match rate. As a reference-informed benchmark, we calculated the score obtained by choosing the lowest-error Whisper candidate for each utterance; this oracle comparison was not used to generate the model's predictions.
