"""
What this file is for:
Stage 2. Runs the child-speech Whisper model over every clip and keeps the
top N candidate transcripts for each one, not just the best.

High-level role in the pipeline:
This is where the raw material for correction comes from, and it sets the
ceiling on everything downstream: the corrector cannot recover a word that
appears in none of these candidates. Decoding is beam search, so the candidates
come back ranked and rank 1 is a genuine 1-best. Sampling was tried first and
was worse on both counts - it returned candidates in no order at all, and 44%
of clips collapsed to a single distinct string.

This is the long stage. It saves as it goes and skips clips it has already
done, so a job that runs out of walltime just needs resubmitting.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import clip_seconds, load_config
from text import clean_whisper_text


def load_model(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" if device == "cuda" else "eager",
    ).to(device)
    # Legacy forced_decoder_ids from this checkpoint's generation_config
    # conflicts with the explicit task="transcribe" passed at call time;
    # transformers already prefers the explicit arg, this just silences the
    # per-batch warning about it.
    model.generation_config.forced_decoder_ids = None
    model.eval()

    print(f"Model:  {model_id}")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    return processor, model, device, dtype


def unique_hypotheses(texts: list[str], scores: list, limit: int) -> list[dict]:
    """The distinct candidates for one clip, best first.

    Under beam search the incoming order is already by sequence score, so rank 1
    is a genuine 1-best rather than an arbitrary draw. Scores are kept because
    they are what any later reranking would need.
    """
    ranked: list[tuple[str, float | None]] = []
    seen: set[str] = set()

    for text, score in zip(texts, scores):
        cleaned = clean_whisper_text(text)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ranked.append((cleaned, score))

    # Fewer distinct strings than asked for is a real signal about this clip -
    # don't pad it out with duplicates.
    return [
        {"rank": i + 1, "text": text, "score": score}
        for i, (text, score) in enumerate(ranked[:limit])
    ]


def decode_arguments(config: dict) -> dict:
    """Whichever decoding strategy the config asks for.

    Beam search is the default. Sampling gave no ranking at all - rank 1 was
    whichever draw happened to land first - so the 1-best baseline it produced
    was not a best of anything, and 44% of clips collapsed to a single distinct
    string. Sampling stays available for ablation.
    """
    if config["asr_do_sample"]:
        return {
            "do_sample": True,
            "num_beams": 1,
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config["top_k"],
        }

    # early_stopping=True was tried here (2026-08-27) to bound runtime, but it
    # stops the search the instant num_beams sequences hit EOS - with
    # num_beams == num_return_sequences that left no room for beams to diverge
    # into different word choices, and candidate diversity collapsed (98% of
    # clips down to one distinct string). Runtime is bounded by max_new_tokens
    # (decode_budget below) instead, so early stopping is left at its default
    # and the beam margin in configs/baseline.yaml does the diversity work.
    arguments = {
        "do_sample": False,
        "num_beams": config["asr_num_beams"],
    }

    # Grouped beams spread the candidates further apart. Transformers rejects a
    # diversity penalty when there is only one group, so only send it when asked.
    if config["asr_beam_groups"] > 1:
        arguments["num_beam_groups"] = config["asr_beam_groups"]
        arguments["diversity_penalty"] = config["asr_diversity_penalty"]

    return arguments


def decode_budget(audio_arrays, config) -> int:
    """How many tokens the longest clip in this batch could plausibly need.

    Whisper pads every clip to 30 seconds and will happily decode toward its
    448-token limit on a 2-second one. Sampling hid this because each draw
    stopped at its own EOS; beam search does not, and uncapped it made stage 2
    roughly 45x slower than it needs to be.
    """
    longest = max((len(audio) / config["sample_rate"] for audio in audio_arrays), default=0.0)
    budget = int(longest * config["asr_tokens_per_second"]) + config["asr_token_margin"]
    return max(config["asr_token_margin"], budget)


def transcribe_batch(audio_arrays, processor, model, device, dtype, config) -> list[list[dict]]:
    inputs = processor(
        audio_arrays,
        sampling_rate=config["sample_rate"],
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device=device, dtype=dtype)

    with torch.inference_mode():
        # Deliberately NOT output_scores/return_dict_in_generate. Getting
        # sequences_scores out of transformers requires output_scores, which
        # also retains the full per-step logits: 448 steps x (batch x beams) x
        # 51,865 vocab is several GB per batch and exhausted an 80 GB H100.
        # Beam search already returns candidates best-first, so rank carries the
        # ordering and the numeric score was only ever a nice-to-have.
        sequences = model.generate(
            input_features,
            language="english",
            task="transcribe",
            num_return_sequences=config["num_return_sequences"],
            max_new_tokens=decode_budget(audio_arrays, config),
            **decode_arguments(config),
        )

    texts = processor.batch_decode(sequences, skip_special_tokens=True)
    scores = [None] * len(texts)

    # generate() returns the sequences for each clip back to back.
    per_clip = config["num_return_sequences"]
    return [
        unique_hypotheses(
            texts[i * per_clip:(i + 1) * per_clip],
            scores[i * per_clip:(i + 1) * per_clip],
            per_clip,
        )
        for i in range(len(audio_arrays))
    ]


def generate_nbest(config: dict) -> None:
    output_path = config["nbest_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    if output_path.is_file():
        results = json.loads(output_path.read_text(encoding="utf-8"))
        print(f"Resuming from {len(results):,} clips already transcribed")

    clips = sorted(config["media_dir"].rglob("*" + config["audio_extension"]))
    if config["asr_limit"]:
        clips = clips[:config["asr_limit"]]
    # Longest-clip-wins is how the decode budget is set, so group similar
    # durations together; otherwise one 25 s clip lifts the budget for the
    # 0.1 s clips sharing its batch.
    todo = sorted((p for p in clips if p.stem not in results),
                  key=lambda p: clip_seconds(p.stem) or 0.0)
    print(f"Clips found: {len(clips):,} | to transcribe: {len(todo):,}")

    if not todo:
        return

    processor, model, device, dtype = load_model(config["asr_model_id"])

    started = time.perf_counter()
    done = 0
    skipped_duration = 0
    failed = 0
    batch_ids: list[str] = []
    batch_audio: list = []

    def save() -> None:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)

    def flush() -> None:
        nonlocal done, failed, batch_ids, batch_audio
        if not batch_audio:
            return

        def record(utterance_id: str, nbest: list[dict]) -> None:
            nonlocal done
            results[utterance_id] = {
                "nbest": nbest,
                "1best_text": nbest[0]["text"] if nbest else "",
            }
            done += 1

        failure = None
        try:
            for utterance_id, nbest in zip(
                batch_ids, transcribe_batch(batch_audio, processor, model, device, dtype, config)
            ):
                record(utterance_id, nbest)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"

        # The retry has to happen out here, not in the except block. While that
        # block is running the traceback still references the failed call's
        # frames - every activation and cache it allocated - so on an OOM there
        # is no memory to retry into and all the singles fail too. Leaving the
        # block drops those references; empty_cache then returns the blocks.
        if failure is not None:
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"[batch failed, retrying singly] {failure}")

            for utterance_id, audio in zip(batch_ids, batch_audio):
                try:
                    record(
                        utterance_id,
                        transcribe_batch([audio], processor, model, device, dtype, config)[0],
                    )
                except Exception as inner:
                    failed += 1
                    print(f"[skip] {utterance_id}: {type(inner).__name__}: {inner}")
                    if device == "cuda":
                        torch.cuda.empty_cache()

        batch_ids, batch_audio = [], []

    for clip in todo:
        try:
            audio, _ = librosa.load(clip, sr=config["sample_rate"], mono=True)
        except Exception as error:
            failed += 1
            print(f"[skip] {clip.name}: {type(error).__name__}: {error}")
            continue

        # A handful of clips are whole sessions or empty. Whisper only sees the
        # first 30 seconds, so transcribing them would score against a
        # transcript it never had a chance to produce.
        seconds = len(audio) / config["sample_rate"]
        if not config["min_clip_seconds"] <= seconds <= config["max_clip_seconds"]:
            skipped_duration += 1
            continue

        batch_ids.append(clip.stem)
        batch_audio.append(audio)

        if len(batch_audio) >= config["asr_batch_size"]:
            flush()
            if done and done % config["asr_save_every"] < config["asr_batch_size"]:
                save()
                rate = done / (time.perf_counter() - started)
                print(f"[{done:,}/{len(todo):,}] {rate:.2f} clips/sec")

    flush()
    save()

    elapsed = time.perf_counter() - started
    print(f"Transcribed:        {done:,}")
    print(f"Skipped (duration): {skipped_duration:,}")
    print(f"Failed:             {failed:,}")
    print(f"Time:               {elapsed / 60:.1f} min")
    print(f"Wrote {output_path}")


def main(config: dict | None = None) -> None:
    generate_nbest(config or load_config())


if __name__ == "__main__":
    main()
