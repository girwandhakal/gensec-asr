"""
What this file is for:
Stage 4. Fine-tunes FLAN-T5 to read a list of competing ASR hypotheses and
write the corrected transcript, then runs it on the held-out test split.

High-level role in the pipeline:
This is the correction model itself. It can do three things the acoustic model
cannot: prefer the reading most hypotheses agree on, splice the right words
out of different candidates, and fall back on what English usually sounds
like.
"""

from __future__ import annotations

import math
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import clip_seconds, load_config, predictions_path
from text import collapse_whitespace

INSTRUCTION = """You are correcting ASR output.
Below are multiple recognition hypotheses for the same utterance.
Choose the single most likely correct transcript.
Do not paraphrase.
Do not add information.
Preserve the original wording as much as possible.
Output only the corrected transcript."""


def normalize(text: str) -> str:
    return collapse_whitespace(str(text or "").lower())


def format_hypotheses(hypotheses: list[str]) -> str:
    return "\n".join(f"{i}. {text}" for i, text in enumerate(hypotheses, start=1))


def consensus_evidence(hypotheses: list[str], config: dict) -> str:
    """Render word support across hypotheses as explicit model evidence."""
    word_counts = Counter()
    for hypothesis in hypotheses:
        # Count a word once per hypothesis: repeated words in one transcript
        # should not masquerade as agreement between candidates.
        word_counts.update(set(normalize(hypothesis).split()))

    minimum = min(config.get("consensus_min_support", 2), len(hypotheses))
    supported = [
        (word, count)
        for word, count in word_counts.items()
        if count >= minimum
    ]
    supported.sort(key=lambda item: (-item[1], item[0]))
    supported = supported[: config.get("consensus_max_words", 40)]

    if not supported:
        return "none"
    total = max(len(hypotheses), 1)
    return ", ".join(f"{word} ({count}/{total})" for word, count in supported)


def hypotheses_block(hypotheses: list[str], config: dict) -> str:
    return (
        f"Hypotheses:\n{format_hypotheses(hypotheses)}\n"
        f"Consensus evidence (word support): {consensus_evidence(hypotheses, config)}"
    )


def build_nbest_prompt(hypotheses: list[str], config: dict) -> str:
    return f"{INSTRUCTION}\n\n{hypotheses_block(hypotheses, config)}"


def build_icl_prompt(hypotheses, demonstrations, tokenizer, config):
    """Prompt with in-context examples, trimmed to fit the token budget.

    Demonstrations go first because they are the cheapest thing to lose; only
    once they are all gone do we start dropping hypotheses.
    """
    demos = list(demonstrations)
    hyps = list(hypotheses)

    def render() -> str:
        # With no demonstrations this has to match the fine-tuning prompt
        # exactly. It used to append a trailing "Corrected:" the fine-tune
        # never saw, which put zero-shot inference off-distribution and made
        # the model echo the hypothesis list back instead of correcting it.
        if not demos:
            return build_nbest_prompt(hyps, config)

        blocks = [INSTRUCTION]
        for demo in demos:
            demo_hyps = demo["input"][:config["icl_demo_hypotheses"]]
            blocks.append(f"{hypotheses_block(demo_hyps, config)}\nCorrected: {demo['output']}")
        blocks.append(f"{hypotheses_block(hyps, config)}\nCorrected:")
        return "\n\n".join(blocks)

    def token_count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    prompt = render()
    while token_count(prompt) > config["max_source_length"] and demos:
        demos.pop()
        prompt = render()
    while token_count(prompt) > config["max_source_length"] and len(hyps) > config["min_hypotheses"]:
        hyps.pop()
        prompt = render()

    return prompt, len(demos), len(hyps), token_count(prompt)


def load_examples(config: dict) -> pd.DataFrame:
    """Read the processed dataset, normalize it, and drop anything unusable."""
    frame = pd.read_json(config["processed_path"])

    rows = []
    for example in frame.to_dict("records"):
        target = normalize(example["output"])
        hypotheses = [h for h in (normalize(h) for h in example["input"]) if h]
        # Deduplicate again: two hypotheses can collide once lowercased.
        hypotheses = list(dict.fromkeys(hypotheses))

        if target and len(hypotheses) >= config["min_hypotheses"]:
            rows.append({"id": example["id"], "input": hypotheses, "output": target})

    print(f"Examples after normalizing: {len(rows):,} of {len(frame):,}")
    return pd.DataFrame(rows)


def make_splits(examples: pd.DataFrame, config: dict):
    train_frame, test_frame = train_test_split(
        examples,
        test_size=config["test_size"],
        random_state=config["seed"],
        shuffle=True,
    )

    config["splits_dir"].mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(config["splits_dir"] / "train_split.csv", index=False)
    test_frame.to_csv(config["splits_dir"] / "test_split.csv", index=False)
    print(f"Train: {len(train_frame):,} | Test: {len(test_frame):,}")

    return train_frame, test_frame


class AbortOnDeadTraining(TrainerCallback):
    """Stop the job the moment the optimizer stops updating weights.

    A NaN gradient norm, or a loss of exactly 0.0, means the scaler is
    discarding every step and the run cannot learn anything. Left alone that
    still costs a full walltime and produces a checkpoint identical to the
    pretrained model, which is indistinguishable from a bad result until you
    read the log. Fail loudly instead.
    """

    def __init__(self, grace_steps: int = 100):
        self.grace_steps = grace_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or state.global_step < self.grace_steps:
            return

        grad_norm = logs.get("grad_norm")
        if grad_norm is not None and math.isnan(float(grad_norm)):
            raise RuntimeError(
                f"Gradient norm is NaN at step {state.global_step}. No weights are "
                "being updated - check the training precision (T5 needs bf16 or fp32, "
                "never fp16)."
            )

        loss = logs.get("loss")
        if loss is not None and float(loss) == 0.0:
            raise RuntimeError(
                f"Training loss is exactly 0.0 at step {state.global_step}. That is a "
                "dead run, not a converged one - check the training precision."
            )


def fine_tune(train_frame: pd.DataFrame, tokenizer, config):
    def tokenize(batch):
        prompts = [build_nbest_prompt(h, config) for h in batch["input"]]
        model_inputs = tokenizer(
            prompts, max_length=config["max_source_length"], truncation=True
        )
        model_inputs["labels"] = tokenizer(
            text_target=batch["output"], max_length=config["max_target_length"], truncation=True
        )["input_ids"]
        return model_inputs

    dataset = Dataset.from_pandas(train_frame[["input", "output"]], preserve_index=False)
    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    # A slice held out of training, purely so the loss curve has something to
    # be checked against per epoch. The scored test split is separate.
    split = dataset.train_test_split(test_size=config["eval_size"], seed=config["seed"])
    print(f"Fine-tune on {len(split['train']):,} | validate on {len(split['test']):,}")

    model = AutoModelForSeq2SeqLM.from_pretrained(config["gensec_model_id"])
    arguments = Seq2SeqTrainingArguments(
        output_dir=str(config["work_dir"]),
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=1,
        logging_steps=100,
        # NOT fp16. T5 was pretrained in bfloat16 and overflows fp16's range:
        # the gradient scaler then skips every optimizer step, so the loss logs
        # as exactly 0.0, the LR never decays, and five epochs of training
        # change no weights at all. That is what happened on 2026-08-25.
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to=[],
        seed=config["seed"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[AbortOnDeadTraining()],
    )
    trainer.train()

    final_checkpoint = config["work_dir"] / "final_checkpoint"
    trainer.save_model(str(final_checkpoint))
    tokenizer.save_pretrained(str(final_checkpoint))
    (final_checkpoint / "train_rows.txt").write_text(str(len(train_frame)), encoding="utf-8")
    (final_checkpoint / "training_signature.txt").write_text(
        config["methodology_version"], encoding="utf-8"
    )
    print(f"Saved model to {final_checkpoint}")

    return model


def pick_demonstrations(mode: str, utterance_id: str, pool: list[dict], config: dict) -> list[dict]:
    """Sample in-context examples, deterministically per test utterance."""
    if mode == "zero_shot":
        return []

    count = 1 if mode == "one_shot" else config["few_shot_examples"]
    rng = random.Random(f"{mode}:{utterance_id}")
    candidates = [example for example in pool if example["id"] != utterance_id]
    return rng.sample(candidates, min(count, len(candidates)))


def generation_cap(batch: list[dict], tokenizer, config) -> int:
    """The longest output worth allowing for this batch - the tighter of two bounds.

    What the clip could physically contain, from its duration, and roughly the
    length of the longest hypothesis. The duration bound is the one that
    matters: Whisper hallucinates whole sentences onto near-silent clips, and
    every catastrophic row of the 2026-08-27 run was a 100-500 ms clip scored
    against a one-word reference. A hypothesis-derived bound cannot catch that,
    because it scales off the hallucination itself.
    """
    longest = max(
        (
            len(tokenizer(text, add_special_tokens=False)["input_ids"])
            for example in batch
            for text in example["input"]
        ),
        default=0,
    )
    by_hypothesis = int(longest * config["generation_length_slack"]) + 5

    # Unparseable ids fall back to the hypothesis bound alone rather than
    # silently capping everything at the floor.
    seconds = max((clip_seconds(example["id"]) or 0.0 for example in batch), default=0.0)
    by_duration = (
        int(seconds * config["generation_tokens_per_second"])
        if seconds
        else config["max_target_length"]
    )

    cap = min(by_hypothesis, by_duration, config["max_target_length"])
    return max(config["generation_min_tokens"], cap)


def generated_sequence_confidence(model, inputs, sequences, tokenizer) -> list[float]:
    """Estimate confidence from the model's mean generated-token probability.

    This uses one teacher-forced forward pass after generation rather than
    retaining every beam's vocabulary logits during generation. That keeps the
    confidence signal practical for the same small inference batches used by
    the pipeline.
    """
    if sequences.shape[1] < 2:
        return [0.0] * sequences.shape[0]

    forward_inputs = {
        key: value
        for key, value in inputs.items()
        if key in {"input_ids", "attention_mask"}
    }
    outputs = model(
        **forward_inputs,
        decoder_input_ids=sequences[:, :-1],
        use_cache=False,
    )
    labels = sequences[:, 1:]
    log_probabilities = torch.log_softmax(outputs.logits.float(), dim=-1)
    token_log_probabilities = log_probabilities.gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    valid = labels.ne(tokenizer.pad_token_id)
    lengths = valid.sum(dim=1).clamp_min(1)
    mean_log_probability = (token_log_probabilities * valid).sum(dim=1) / lengths
    return torch.exp(mean_log_probability).detach().cpu().tolist()


def hypothesis_support(text: str, hypotheses: list[str]) -> float:
    """Return the average fraction of hypotheses supporting each output word."""
    words = normalize(text).split()
    if not words or not hypotheses:
        return 0.0

    candidate_words = [set(normalize(hypothesis).split()) for hypothesis in hypotheses]
    support = sum(
        sum(word in candidate for candidate in candidate_words) / len(candidate_words)
        for word in words
    )
    return support / len(words)


def selective_decision(
    prediction: str,
    one_best: str,
    generation_confidence: float,
    support: float,
    config: dict,
) -> tuple[str, float, bool, str]:
    """Accept a correction only when its confidence and evidence pass the gate."""
    selective_score = generation_confidence * (0.5 + 0.5 * support)

    if prediction == one_best:
        return prediction, selective_score, False, "same_as_1best"

    if (
        selective_score >= config["selective_min_score"]
        and support >= config["selective_min_support"]
    ):
        return prediction, selective_score, True, "accepted"

    return one_best, selective_score, False, "abstained"


def run_inference(model, tokenizer, train_frame, test_frame, mode, config) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    pool = train_frame.to_dict("records")

    # Batch clips of similar length together. The cap below is one number for
    # the whole batch, so mixing a 30-second clip with a 100 ms one would lift
    # the short clip's ceiling to the long clip's. Sorting also cuts padding.
    tests = sorted(test_frame.to_dict("records"), key=lambda e: clip_seconds(e["id"]) or 0.0)

    # Beam search holds num_beams sequences in flight per row, and reordering
    # the cache between steps allocates a second copy of it - so generation
    # needs a far smaller batch than the teacher-forced eval pass does. Sharing
    # one number with eval_batch_size is what put a T4 out of memory.
    batch_size = config["generation_batch_size"]

    rows = []
    for start in range(0, len(tests), batch_size):
        batch = tests[start:start + batch_size]
        built = [
            build_icl_prompt(
                example["input"],
                pick_demonstrations(mode, example["id"], pool, config),
                tokenizer,
                config,
            )
            for example in batch
        ]

        inputs = tokenizer(
            [prompt for prompt, _, _, _ in built],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["max_source_length"],
        ).to(device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=generation_cap(batch, tokenizer, config),
                num_beams=config["num_beams"],
                # Repetition control, not cleanup. The first working run emitted
                # 110,711 insertions - looping phrases until the length limit -
                # and postprocessing stripped 98,197 of them after the fact.
                # Blocking the loop here is what should carry the result.
                no_repeat_ngram_size=config["no_repeat_ngram_size"],
                repetition_penalty=config["repetition_penalty"],
                do_sample=False,
            )
            generation_confidences = generated_sequence_confidence(
                model, inputs, generated, tokenizer
            )
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for example, (prompt, demos, hyps, tokens), prediction, gen_confidence in zip(
            batch, built, predictions, generation_confidences
        ):
            cleaned_pred = collapse_whitespace(prediction)
            words = cleaned_pred.split()

            # Per-utterance physical speech limit guard: a clip cannot contain more words
            # than physically possible at maximum child speech rate (~4 words/sec).
            sec = clip_seconds(example["id"])
            if sec is not None and sec > 0:
                max_words = max(2, math.ceil(sec * 4.0) + 1)
                if len(words) > max_words:
                    words = words[:max_words]
                    cleaned_pred = " ".join(words)

            # Short-utterance hallucination gate:
            # Seq2seq LMs tend to expand short 1-2 word utterances into whole sentences.
            # If all candidate hypotheses are <= 2 words, reject any prediction that
            # inflates beyond 3 words and fall back to the Whisper 1-best.
            max_hyp_len = max((len(h.split()) for h in example["input"]), default=0)
            if max_hyp_len <= 2 and len(words) > 3:
                one_best = example["input"][0] if example["input"] else cleaned_pred
                cleaned_pred = one_best

            one_best = example["input"][0] if example["input"] else ""
            support = hypothesis_support(cleaned_pred, example["input"])
            selective_pred, selective_score, accepted, decision = selective_decision(
                cleaned_pred,
                one_best,
                float(gen_confidence),
                support,
                config,
            )

            rows.append({
                "id": example["id"],
                "input": " | ".join(example["input"]),
                "one_best": one_best,
                "input_token_count": tokens,
                "retained_icl_examples": demos,
                "retained_input_hypotheses": hyps,
                "truth": example["output"],
                "prediction": cleaned_pred,
                "selective_prediction": selective_pred,
                "generation_confidence": float(gen_confidence),
                "hypothesis_support": support,
                "selective_score": selective_score,
                "correction_accepted": accepted,
                "selective_decision": decision,
            })

        if start % (batch_size * 20) == 0:
            print(f"  [{mode}] {min(start + len(batch), len(tests)):,}/{len(tests):,}")

    output_path = predictions_path(config, mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


def main(config: dict | None = None) -> None:
    config = config or load_config()

    if not config["processed_path"].is_file():
        raise SystemExit(f"Missing dataset: {config['processed_path']}")

    examples = load_examples(config)
    train_frame, test_frame = make_splits(examples, config)

    tokenizer = AutoTokenizer.from_pretrained(config["gensec_model_id"])

    # Stage 4 is two expensive halves and only the second one is cheap to
    # repeat. Inference dying - on an OOM, or a walltime - used to throw away a
    # finished fine-tune, because the stage only skips on the predictions file.
    # That reuse used to key on the checkpoint directory merely existing, which
    # silently ran a stale fine-tune against a train split that had since grown
    # or been rebuilt (2026-08-31: stage 2/3 reworked the n-best data and the
    # train split changed from 59,937 to 67,249 rows, but inference still ran
    # the 2026-08-26 checkpoint trained on the old split - a real train/test
    # mismatch, not a decoding regression). Stamp the row count the checkpoint
    # was trained on and retrain whenever the current split doesn't match it.
    final_checkpoint = config["work_dir"] / "final_checkpoint"
    stamp_path = final_checkpoint / "train_rows.txt"
    signature_path = final_checkpoint / "training_signature.txt"
    stamp = stamp_path.read_text(encoding="utf-8").strip() if stamp_path.is_file() else None
    signature = (
        signature_path.read_text(encoding="utf-8").strip()
        if signature_path.is_file()
        else None
    )

    if (
        final_checkpoint.is_dir()
        and stamp == str(len(train_frame))
        and signature == config["methodology_version"]
    ):
        print(
            f"Using existing fine-tune: {final_checkpoint} "
            f"(trained on {stamp} rows, {signature})"
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(str(final_checkpoint))
    else:
        if final_checkpoint.is_dir():
            print(
                f"Discarding fine-tune at {final_checkpoint}: trained on "
                f"{stamp or 'an unknown number of'} rows / {signature or 'unknown methodology'}, "
                f"current split has {len(train_frame):,} / {config['methodology_version']} "
                "- retraining"
            )
        model = fine_tune(train_frame, tokenizer, config)

    for mode in config["inference_modes"]:
        print(f"===== inference: {mode} =====")
        run_inference(model, tokenizer, train_frame, test_frame, mode, config)


if __name__ == "__main__":
    main()
