"""Fine-tune Qwen3 to correct the same Whisper hypotheses used by FLAN-T5."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import clip_seconds, load_config, predictions_path
from text import collapse_whitespace
from train import build_nbest_prompt, load_examples, make_splits, training_signature


def qwen_training_signature(config: dict) -> str:
    return training_signature(config) + "|qwen_chat_no_thinking_v1"


def qwen_prompt(hypotheses: list[str], tokenizer) -> str:
    """Use Qwen's chat format while keeping the shared GenSEC prompt intact."""
    messages = [{"role": "user", "content": build_nbest_prompt(hypotheses)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def encode_training_batch(batch: dict, tokenizer, config: dict) -> dict:
    """Mask prompt tokens so the causal loss trains only on reference text."""
    input_ids, labels, attention_masks = [], [], []
    for hypotheses, target in zip(batch["input"], batch["output"]):
        prompt = qwen_prompt(hypotheses, tokenizer)
        prompt_ids = tokenizer(
            prompt, add_special_tokens=False, truncation=True,
            max_length=config["max_source_length"],
        )["input_ids"]
        target_ids = tokenizer(
            target, add_special_tokens=False, truncation=True,
            max_length=config["max_target_length"] - 1,
        )["input_ids"] + [tokenizer.eos_token_id]
        input_ids.append(prompt_ids + target_ids)
        # The prompt supplies context; only reference tokens should contribute to loss.
        labels.append([-100] * len(prompt_ids) + target_ids)
        attention_masks.append([1] * (len(prompt_ids) + len(target_ids)))
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_masks}


class CompletionCollator:
    """Pad causal batches without letting padded or prompt tokens affect loss."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        width = max(len(item["input_ids"]) for item in features)
        input_ids, labels, attention_mask = [], [], []
        for item in features:
            missing = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.tokenizer.pad_token_id] * missing)
            labels.append(item["labels"] + [-100] * missing)
            attention_mask.append(item["attention_mask"] + [0] * missing)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def prepare_data(config: dict, metadata: dict):
    examples = load_examples(config)
    training, test = make_splits(examples, config, metadata)
    fit, validation = make_splits(
        training, {**config, "test_size": config["eval_size"]}, metadata, save=False
    )
    return test, fit, validation


def train_model(fit: pd.DataFrame, validation: pd.DataFrame, tokenizer, config: dict):
    work_dir = config["qwen_work_dir"]
    work_dir.mkdir(parents=True, exist_ok=True)
    active_signature = work_dir / "active_training_signature.txt"
    signature = qwen_training_signature(config)
    if active_signature.is_file() and active_signature.read_text(encoding="utf-8") != signature:
        raise ValueError("Qwen training settings changed inside a partial run; use a fresh run directory.")
    active_signature.write_text(signature, encoding="utf-8")

    def encode(batch):
        return encode_training_batch(batch, tokenizer, config)

    datasets = {}
    for name, frame in (("train", fit), ("validation", validation)):
        source = Dataset.from_pandas(frame[["input", "output"]], preserve_index=False)
        datasets[name] = source.map(encode, batched=True, remove_columns=source.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        config["qwen_model_id"], torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
    )
    model.config.use_cache = False
    args = TrainingArguments(
        output_dir=str(work_dir),
        per_device_train_batch_size=config["qwen_train_batch_size"],
        per_device_eval_batch_size=config["qwen_eval_batch_size"],
        learning_rate=config["qwen_learning_rate"],
        num_train_epochs=config["qwen_num_train_epochs"],
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="epoch",
        logging_steps=100,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to=[],
        seed=config["seed"],
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=CompletionCollator(tokenizer),
    )
    print(f"Qwen fine-tune: {len(datasets['train']):,} train | {len(datasets['validation']):,} validation")
    checkpoints = sorted(work_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    # Resume the last epoch if the Slurm job ended before training finished.
    trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)

    checkpoint = work_dir / "final_checkpoint"
    trainer.save_model(str(checkpoint))
    tokenizer.save_pretrained(str(checkpoint))
    (checkpoint / "training_signature.txt").write_text(qwen_training_signature(config), encoding="utf-8")
    (checkpoint / "train_rows.txt").write_text(str(len(fit)), encoding="utf-8")
    # Once the final model is saved and signed, optimizer/RNG state is no longer
    # needed. Keep resumable Trainer checkpoints only while training is partial.
    for stale in work_dir.glob("checkpoint-*"):
        if stale.is_dir():
            shutil.rmtree(stale)
            print(f"Removed completed Trainer checkpoint: {stale}", flush=True)
    model.config.use_cache = True
    return model


def clean_completion(text: str) -> str:
    """Remove any residual Qwen reasoning markers and normalize whitespace."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return collapse_whitespace(text.replace("<|im_end|>", "").replace("<|endoftext|>", ""))


def generation_tokens(example: dict, tokenizer, config: dict) -> int:
    longest = max((len(tokenizer(h, add_special_tokens=False)["input_ids"])
                   for h in example["input"]), default=0)
    by_hypothesis = int(longest * config["generation_length_slack"]) + 5
    seconds = clip_seconds(example["id"])
    by_duration = int(seconds * config["generation_tokens_per_second"]) if seconds else config["max_target_length"]
    return max(config["generation_min_tokens"], min(by_hypothesis, by_duration, config["max_target_length"]))


def generate_predictions(model, tokenizer, test: pd.DataFrame, config: dict) -> None:
    """Write one Qwen correction per test clip from the shared Whisper input pool."""
    rows_path = predictions_path(config, config["inference_modes"][0])
    rows_by_id = {}
    if rows_path.is_file():
        old_rows = pd.read_csv(rows_path).fillna("").to_dict("records")
        rows_by_id = {row["id"]: row for row in old_rows}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    model.config.use_cache = True
    tests = test.to_dict("records")
    # Decoder-only generation must continue from the final prompt token.
    tokenizer.padding_side = "left"
    pending = [example for example in tests if example["id"] not in rows_by_id]
    print(f"Qwen corrections: {len(rows_by_id):,} complete | {len(pending):,} to generate")

    batch_size = config["qwen_generation_batch_size"]
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        prompts = [qwen_prompt(item["input"], tokenizer) for item in batch]
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=config["max_source_length"],
        ).to(device)
        caps = [generation_tokens(item, tokenizer, config) for item in batch]
        cap = max(caps, default=config["generation_min_tokens"])
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=cap, do_sample=False,
                num_beams=config["num_beams"],
                no_repeat_ngram_size=config["no_repeat_ngram_size"],
                repetition_penalty=config["repetition_penalty"],
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_width = encoded["input_ids"].shape[1]
        predictions = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        for index, (example, prediction) in enumerate(zip(batch, predictions)):
            uid = example["id"]
            primary = clean_completion(prediction)
            seconds = clip_seconds(uid)
            if seconds and seconds > 0:
                # Match FLAN's clip-length guard for implausibly long corrections.
                max_words = max(2, math.ceil(seconds * 4.0) + 1)
                primary = " ".join(primary.split()[:max_words])
            rows_by_id[uid] = {
                "id": uid,
                "input": " | ".join(example["input"]),
                "one_best": example["input"][0] if example["input"] else "",
                "input_token_count": len(tokenizer(prompts[index])["input_ids"]),
                "retained_icl_examples": 0,
                "retained_input_hypotheses": len(example["input"]),
                "truth": example["output"],
                "prediction": primary,
            }

        completed = min(start + len(batch), len(pending))
        if completed % config["qwen_save_every"] < batch_size or completed == len(pending):
            saved_rows = [rows_by_id[item["id"]] for item in tests if item["id"] in rows_by_id]
            temporary = rows_path.with_suffix(".tmp")
            pd.DataFrame(saved_rows).to_csv(temporary, index=False)
            temporary.replace(rows_path)
            print(f"  Qwen predictions saved: {len(rows_by_id):,}/{len(tests):,}")


def main(config: dict | None = None) -> None:
    config = config or load_config()
    metadata = json.loads(config["metadata_path"].read_text(encoding="utf-8"))
    test_frame, fit, validation = prepare_data(config, metadata)

    qwen_config = dict(config)
    qwen_config["gensec_model_id"] = config["qwen_model_id"]
    qwen_config["num_train_epochs"] = config["qwen_num_train_epochs"]
    qwen_config["learning_rate"] = config["qwen_learning_rate"]
    qwen_config["train_batch_size"] = config["qwen_train_batch_size"]

    tokenizer = AutoTokenizer.from_pretrained(config["qwen_model_id"], padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    current_signature = qwen_training_signature(qwen_config)
    work_dir = qwen_config["qwen_work_dir"]
    active_signature = work_dir / "active_training_signature.txt"
    saved_signature = (
        active_signature.read_text(encoding="utf-8")
        if active_signature.is_file()
        else None
    )

    if saved_signature is not None and saved_signature != current_signature:
        print(f"Qwen signature changed; replacing incompatible checkpoints in {work_dir}", flush=True)
        for stale in [work_dir / "final_checkpoint", *work_dir.glob("checkpoint-*")]:
            if stale.is_dir():
                shutil.rmtree(stale)
        prediction_path = predictions_path(config, config["inference_modes"][0])
        if prediction_path.is_file():
            prediction_path.unlink()
        active_signature.unlink()

    checkpoint = qwen_config["qwen_work_dir"] / "final_checkpoint"
    signature_path = checkpoint / "training_signature.txt"
    rows_path = checkpoint / "train_rows.txt"
    if (
        checkpoint.is_dir() and signature_path.is_file() and rows_path.is_file()
        and signature_path.read_text(encoding="utf-8") == current_signature
        and rows_path.read_text(encoding="utf-8").strip() == str(len(fit))
    ):
        print(f"Using existing Qwen fine-tune: {checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="right")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = "left"
    else:
        qwen_config["qwen_work_dir"].mkdir(parents=True, exist_ok=True)
        model = train_model(fit, validation, tokenizer, qwen_config)

    generate_predictions(model, tokenizer, test_frame, config)


if __name__ == "__main__":
    main()
