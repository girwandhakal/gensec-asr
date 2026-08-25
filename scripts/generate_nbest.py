"""
What this file is for:
Stage 2. Runs the child-speech Whisper model over every clip and keeps the
top N candidate transcripts for each one, not just the best.

High-level role in the pipeline:
This is where the raw material for correction comes from. Decoding is sampled
rather than beam-searched on purpose: beams collapse into near-identical
strings, and the correction model only has something to work with when the
hypotheses actually disagree.

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
from config import load_config
from text import clean_whisper_text


def load_model(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    print(f"Model:  {model_id}")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    return processor, model, device, dtype


def unique_hypotheses(texts: list[str], limit: int) -> list[dict]:
    """Rank the candidates for one clip, keeping the distinct ones first."""
    ranked: list[str] = []
    for text in texts:
        cleaned = clean_whisper_text(text)
        if cleaned and cleaned not in ranked:
            ranked.append(cleaned)

    # If sampling gave us fewer distinct strings than asked for, that is a real
    # signal about this clip - don't pad it out with duplicates.
    return [{"rank": i + 1, "text": text} for i, text in enumerate(ranked[:limit])]


def transcribe_batch(audio_arrays, processor, model, device, dtype, config) -> list[list[dict]]:
    inputs = processor(
        audio_arrays,
        sampling_rate=config["sample_rate"],
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device=device, dtype=dtype)

    with torch.inference_mode():
        generated = model.generate(
            input_features,
            language="english",
            task="transcribe",
            do_sample=True,
            num_beams=1,
            num_return_sequences=config["num_return_sequences"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            top_k=config["top_k"],
        )

    texts = processor.batch_decode(generated, skip_special_tokens=True)

    # generate() returns the sequences for each clip back to back.
    per_clip = config["num_return_sequences"]
    return [
        unique_hypotheses(texts[i * per_clip:(i + 1) * per_clip], per_clip)
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
    todo = [p for p in clips if p.stem not in results]
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

        try:
            for utterance_id, nbest in zip(
                batch_ids, transcribe_batch(batch_audio, processor, model, device, dtype, config)
            ):
                results[utterance_id] = {
                    "nbest": nbest,
                    "1best_text": nbest[0]["text"] if nbest else "",
                }
                done += 1
        except Exception as error:
            # One bad clip shouldn't cost the whole batch, so retry singly.
            print(f"[batch failed, retrying singly] {type(error).__name__}: {error}")
            for utterance_id, audio in zip(batch_ids, batch_audio):
                try:
                    nbest = transcribe_batch(
                        [audio], processor, model, device, dtype, config
                    )[0]
                    results[utterance_id] = {
                        "nbest": nbest,
                        "1best_text": nbest[0]["text"] if nbest else "",
                    }
                    done += 1
                except Exception as inner:
                    failed += 1
                    print(f"[skip] {utterance_id}: {type(inner).__name__}: {inner}")

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
