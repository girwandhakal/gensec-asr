# Revised ASR and correction methodology

This draft describes the inspected pipeline and saved dataset artifacts. Before submission, resolve the two verification notes below. The saved predictions identify a 40-epoch run, but their recorded processed-dataset SHA-256 differs from the current processed dataset. The timestamp-based duration summary also does not exactly describe the saved Whisper records.

## ASR transcription and candidate preparation

We loaded each audio clip as a mono waveform sampled at 16 kHz. The ASR pipeline accepted clips whose decoded waveform duration was between 0.1 and 30 seconds, inclusive. The lower threshold excluded extremely short clips, while the upper threshold kept each utterance within the single audio window used by our Whisper implementation. Clips outside these limits were excluded rather than divided into shorter segments. The 0.1-second threshold was a preprocessing choice rather than a requirement imposed by Whisper.

[VERIFY BEFORE SUBMISSION: Report actual duration exclusions from the decoding run. An audit of the 142,334 source timestamp intervals found 3,529 intervals below 0.1 seconds and 91 above 30 seconds, totaling 3,620 intervals (2.54%). These are timestamp-based counts, whereas ASR eligibility is determined from decoded waveform duration. The saved correction dataset includes 222 examples with source timestamp intervals below 0.1 seconds. Do not describe all 3,620 as observed ASR exclusions without reconciling those measurements.]

We used the saved Whisper-medium model distributed as `rishabhjain16/whisper_medium_to_myst55h` to transcribe English speech. Its public model card identifies it as a fine-tuned version of Whisper-medium, although its training-data description is incomplete. Any specific statement about child-speech adaptation or the amount of MyST training data should be supported by the model's original training documentation rather than inferred solely from its name. Whisper received audio as input; the human reference transcriptions were not provided to it during recognition.

For each eligible clip, we generated a baseline transcript and a set of alternative transcripts. We refer to a possible transcript of the same audio as a recognition hypothesis. First, we generated one transcript by choosing the most probable next token at each decoding step, conditioned on the audio and the tokens already generated. This procedure, called greedy decoding, supplied the baseline ASR output. The term “1-best” refers here to that designated baseline and does not imply a globally optimal transcript.

Second, we generated ten sampled transcripts for the same clip. Sampling draws the next token from a probability distribution over possible tokens, allowing repeated passes through the same audio to produce different transcripts. These alternatives supplied the correction model with competing interpretations of the audio. Sampling used a temperature of 0.8, a cumulative-probability threshold of 0.95, and a top-token limit of 50. We used random seed 42 and processed clips in batches of four. Each batch had a generation limit of eight tokens per second of its longest clip, rounded down, plus 16 tokens. This was a heuristic to limit unnecessarily long decoding, not an empirically established speech-rate bound.

We cleaned the generated text by removing Whisper control tokens, applying Unicode normalization, repairing common encoding artifacts, standardizing apostrophes, and collapsing whitespace. We compared normalized forms to identify duplicate transcripts. We retained the nonempty greedy transcript first and then selected distinct sampled alternatives until the list contained at most five hypotheses. Alternatives were prioritized by their frequency among the ten samples. Ties favored alternatives introducing words absent from the hypotheses already selected, followed by earlier occurrence in the sampling pool. Sample frequency was a selection heuristic, not a calibrated measure of correctness.

The five-hypothesis limit followed earlier post-ASR correction work using five hypotheses, including La Quatra et al. (2024) and Xu et al. (2025). We did not establish that five was optimal for this dataset. Examples with fewer distinct hypotheses were retained. In the saved correction dataset, 40,853 examples had one hypothesis, 23,759 had two, 16,380 had three, 12,823 had four, and 43,188 had five. Because the stored lists were capped at five, this distribution describes retained inputs rather than the number of distinct transcripts in the original ten-sample pool.

The saved Whisper file contained hypothesis records for 139,348 clip identifiers. Joining those records to the reference map by utterance identifier produced 137,003 correction examples, each containing a hypothesis list and a nonempty reference transcription. The remaining 2,345 records lacked a matching reference and were excluded. These counts describe the saved files and require reconciliation with the duration-filtering record before they are presented as a single complete exclusion flow.

## Training validation and test sets

We divided the 137,003 correction examples by source CHAT transcript, targeting approximately 20% of the examples in each group for the test set. All examples from a transcript were assigned to the same side of the split. With seed 42, this produced 109,591 training-side examples from 703 transcripts and 27,412 test examples from 170 transcripts. We verified that no source transcript occurred in both sets. The test set contained 25,177 TD and 2,235 CD examples; the training side contained 100,635 TD and 8,956 CD examples.

The split separated transcripts rather than verified speakers. Consequently, it did not establish that training and test speech came from different children. Twelve corpus-scoped export IDs occurred on both sides, but these IDs cannot be interpreted as twelve confirmed children.

Within the training side, we randomly reserved 2% of examples for monitoring validation loss, leaving 107,399 training examples and 2,192 validation examples. This second split operated at the example level, so training and validation examples could originate from the same source transcript. The transcript-disjoint test set remained separate. This was a single held-out split, not cross-validation.

## Correction-model fine-tuning

We fine-tuned FLAN-T5-base (`google/flan-t5-base`), a pretrained text-to-text language model, to generate the reference transcription from the Whisper hypotheses. Fine-tuning updated the correction model using paired hypothesis lists and reference targets; Whisper was used as a fixed upstream recognizer.

Each utterance had one prompt containing its available hypotheses, with the greedy baseline listed first. The prompt asked the model to correct recognition errors while preserving the child's wording. The requests not to paraphrase or add information were intended to discourage rewriting, expansion, and unsupported additions; they did not guarantee that the model would avoid those behaviors. The output-only instruction requested a transcript without explanations. The model could generate a corrected sequence rather than being restricted to copying one hypothesis verbatim.

The following template was used for both training and inference without example demonstrations:

```text
You are correcting ASR output.
Below are multiple recognition hypotheses for the same utterance.
Choose the single most likely correct transcript.
Do not paraphrase.
Do not add information.
Preserve the original wording as much as possible.
Output only the corrected transcript.

Hypotheses:
1. <baseline transcript>
2. <sampled alternative, if available>
...
```

The inspected prompt did not include word-support counts. Reference transcriptions served as the training targets and were not included in the input prompt.

Input prompts were truncated to 512 tokens and reference targets to 128 tokens. These were fixed sequence-length limits; their suitability should be supported by reporting the fraction of prompts and targets truncated. The training learning rate was 3 × 10⁻⁵, the per-device training batch size was 8, the validation batch size was 16, and the seed was 42. Validation loss was evaluated after each epoch, and the pipeline saved the model at the end of training rather than selecting the epoch with the lowest validation loss.

[VERIFY BEFORE SUBMISSION: The inspected configuration and saved prediction signature specify 40 epochs, whereas the supplied manuscript says 10. Confirm the completed run and its dataset version, then report that epoch count. The prediction signature's recorded processed-data hash does not match the current file; do not claim the current dataset and those predictions are one verified run until reconciled.]

## Correction inference

For each test example, the fine-tuned FLAN-T5 model received the Whisper hypotheses in the same prompt format used for training. No example corrections or human reference text were included in the test prompt. The inspected prompt contained no word-support counts. The model received text hypotheses rather than audio.

The model generated one corrected transcript using eight-beam decoding without sampling, in batches of four. Beam decoding maintained competing partial sequences during generation before selecting one output. Generation blocked repeated sequences of three model tokens and applied a repetition penalty of 1.2. These settings discouraged looping but could also suppress genuine repetitions in children's speech.

For each batch, the generated-token limit was the tighter of a hypothesis-based limit, a duration-based limit, and 128 tokens, subject to a minimum allowance of six tokens. The hypothesis-based limit was 1.5 times the longest hypothesis's token count, rounded down, plus five tokens. The duration-based limit was eight tokens per second of the longest clip, rounded down.

## Postprocessing and evaluation

After generation, we collapsed repeated whitespace. We also imposed a per-clip word-count cap of the larger of two words or four words per second of audio, rounded up, plus one word. Outputs exceeding the cap were truncated to their first permitted words. This was a heuristic intended to limit overlong outputs; it should not be described as a validated maximum child speech rate. We did not separately remove repeated words or phrases during postprocessing, although the decoding constraints already discouraged repetition. The duration-based word cap was applied to the correction output and should be disclosed as such.

We evaluated the greedy Whisper baseline and the postprocessed FLAN-T5 output on the same test examples and references. The inspected scoring code retained finalized reference strings and normalized model outputs to those reference conventions, including lowercasing, Unicode and encoding normalization, removal of control tokens and punctuation other than apostrophes, whitespace normalization, and mapping selected humming spellings to a common form. Therefore, the manuscript should not claim that the current scoring code re-cleans all three strings identically. Verify that the finalized references use the same humming convention before reporting equivalence under that normalization.

We calculated corpus-level word error rate by summing substitutions, deletions, and insertions across the evaluated examples and dividing by the total number of reference words: WER = (S + D + I) / N. This is not the unweighted mean of individual utterance WERs. If reported as a percentage, the ratio is multiplied by 100.

## Evidence consulted

- Meeting transcript: September 29, 2026, especially 19:43–19:59 (audio, decoding, parameters, hypothesis count), 20:00–20:01 (flowchart), and 20:02–20:14 (prompts, model naming, validation, parameter rationale).
- `scripts/generate_nbest.py`: audio loading, decoded-duration filtering, greedy and sampled decoding, duplicate handling, and alternative selection.
- `scripts/train.py`: prompt format, transcript split, example-level validation split, fine-tuning, inference limits, and output truncation.
- `scripts/text.py` and `scripts/evaluate.py`: prediction normalization and corpus-level WER.
- `data/processed_gensec.json`, `data/dropped_gensec.json`, and `data/splits/`: verified saved example counts and hypothesis distribution.
- `data/predictions/test_predictions_zero_shot.signature.json`: recorded training settings and dataset provenance discrepancy.
- [La Quatra et al. official FlanEC repository](https://github.com/MorenoLaQuatra/FlanEC).
- [Xu et al. 2025 paper](https://www.isca-archive.org/interspeech_2025/xu25g_interspeech.pdf).
- [Whisper model card](https://huggingface.co/rishabhjain16/whisper_medium_to_myst55h).
