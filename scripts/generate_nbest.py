"""
What this file is for:
Stage 2. Runs the child-speech Whisper model over every clip, storing a greedy
ASR baseline and distinct sampled alternatives for correction.

High-level role in the pipeline:
This is where the raw material for correction comes from, and it sets the
ceiling on everything downstream: the corrector cannot recover a word that
appears in none of these candidates.

The first decode is deterministic greedy search. A separate, larger sampling
pool supplies alternatives. The deterministic transcript is always first in
the n-best input, and `1best_text` is never taken from the sampling pool.

This is the long stage. It saves as it goes and skips clips it has already
done, so a job that runs out of walltime just needs resubmitting.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import clip_seconds, load_config
from text import clean_whisper_text, normalize_for_scoring


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


def unique_hypotheses(greedy_text: str, sampled_texts: list[str], limit: int) -> list[dict]:
    """Keep greedy first, then frequent sampled alternatives with distinct words."""
    greedy = clean_whisper_text(greedy_text)
    greedy_key = normalize_for_scoring(greedy)
    counts: Counter[str] = Counter()
    representatives: dict[str, str] = {}
    first_seen: dict[str, int] = {}
    for position, text in enumerate(sampled_texts):
        cleaned = clean_whisper_text(text)
        key = normalize_for_scoring(cleaned)
        if not key:
            continue
        counts[key] += 1
        representatives.setdefault(key, cleaned)
        first_seen.setdefault(key, position)

    chosen: list[tuple[str, str]] = [(greedy_key, greedy)] if greedy_key else []
    remaining = set(counts) - {greedy_key}
    while remaining and len(chosen) < limit:
        selected_words = set().union(*(set(key.split()) for key, _ in chosen))
        key = max(
            remaining,
            key=lambda candidate: (
                counts[candidate],
                bool(set(candidate.split()) - selected_words),
                -first_seen[candidate],
            ),
        )
        chosen.append((key, representatives[key]))
        remaining.remove(key)

    return [
        {
            "rank": rank,
            "text": text,
            "score": None,
            "source": "greedy" if rank == 1 and key == greedy_key and greedy_key else "sampled",
            "sample_count": counts[key],
        }
        for rank, (key, text) in enumerate(chosen, start=1)
    ]


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


def transcribe_batch(audio_arrays, processor, model, device, dtype, config) -> list[dict]:
    inputs = processor(
        audio_arrays,
        sampling_rate=config["sample_rate"],
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device=device, dtype=dtype)

    budget = decode_budget(audio_arrays, config)
    with torch.inference_mode():
        greedy_sequences = model.generate(
            input_features,
            language="english",
            task="transcribe",
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            max_new_tokens=budget,
        )
        sampled_sequences = model.generate(
            input_features,
            language="english",
            task="transcribe",
            do_sample=True,
            num_beams=1,
            num_return_sequences=config["asr_sample_pool_size"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            max_new_tokens=budget,
        )

    greedy_texts = processor.batch_decode(greedy_sequences, skip_special_tokens=True)
    sampled_texts = processor.batch_decode(sampled_sequences, skip_special_tokens=True)
    per_clip = config["asr_sample_pool_size"]
    if len(greedy_texts) != len(audio_arrays) or len(sampled_texts) != len(audio_arrays) * per_clip:
        raise RuntimeError("Whisper returned an unexpected number of transcripts")
    return [
        {
            "1best_text": clean_whisper_text(greedy_texts[i]),
            "nbest": unique_hypotheses(
                greedy_texts[i], sampled_texts[i * per_clip:(i + 1) * per_clip],
                config["num_return_sequences"],
            ),
        }
        for i in range(len(audio_arrays))
    ]


# The settings that determine what a cached n-best entry actually contains.
# Resuming keys on whether a clip id is already present, so without this a
# decoding change reruns nothing and the corpus stays whatever the previous
# strategy produced - the whole corpus silently disagreeing with the config
# archived next to the results.
DECODE_SIGNATURE_KEYS = (
    "asr_model_id",
    "seed",
    "asr_sample_pool_size",
    "temperature",
    "top_p",
    "top_k",
    "num_return_sequences",
    "asr_tokens_per_second",
    "asr_token_margin",
    "sample_rate",
    "min_clip_seconds",
    "max_clip_seconds",
)


def decode_signature(config: dict) -> str:
    payload = {key: config[key] for key in DECODE_SIGNATURE_KEYS if key in config}
    payload["decode_method"] = "greedy_plus_sampled_v2"
    return json.dumps(payload, sort_keys=True, default=str)


def check_decode_signature(config: dict, output_path: Path, cached: int) -> None:
    """Refuse to extend a corpus that was decoded with different settings.

    Deliberately fatal rather than self-healing: regenerating means days of GPU
    time and discarding work nobody asked to discard, so that call belongs to
    whoever is running the job. Continuing would be worse than either option -
    it would mix two decoding strategies into one corpus and label the result
    with whichever config happened to be on disk when it finished.
    """
    stamp_path = output_path.with_suffix(".signature.json")
    current = decode_signature(config)

    if not stamp_path.is_file():
        if cached:
            raise SystemExit(
                f"{output_path.name} has {cached:,} cached clips but no decode signature; "
                "its baseline cannot be verified. Move it aside before regenerating."
            )
        stamp_path.write_text(current, encoding="utf-8")
        return

    previous = stamp_path.read_text(encoding="utf-8")
    if previous != current:
        raise SystemExit(
            f"Decode settings changed since {output_path.name} was generated.\n"
            f"  cached ({cached:,} clips): {previous}\n"
            f"  current:                   {current}\n"
            "Resuming keys on clip id, so those clips would keep their old decoding "
            "and the corpus would mix two strategies. Either revert the change in "
            f"configs/baseline.yaml, or move {output_path.name} and its .signature.json "
            "aside (see data/archive/) to regenerate the corpus from scratch."
        )


def generate_nbest(config: dict) -> None:
    output_path = config["nbest_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    if output_path.is_file():
        results = json.loads(output_path.read_text(encoding="utf-8"))
        print(f"Resuming from {len(results):,} clips already transcribed")

        # Requeue entries carrying no usable candidate at all. Note this catches
        # only empty ones: min_hypotheses is 1, so a clip that collapsed to a
        # single distinct string passes here and keeps its cached result. That
        # collapse is a decoding problem, and a decoding change is what
        # check_decode_signature() handles - this is not the check that
        # rescues those 49,088 single-candidate clips.
        stale = [uid for uid, entry in results.items()
                 if len(entry.get("nbest", [])) < config["min_hypotheses"]]
        for uid in stale:
            del results[uid]
        if stale:
            print(f"Discarding {len(stale):,} stale/collapsed entries for regeneration")

    check_decode_signature(config, output_path, len(results))

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
    torch.manual_seed(config["seed"])

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

        def record(utterance_id: str, entry: dict) -> None:
            nonlocal done
            results[utterance_id] = entry
            done += 1

        failure = None
        try:
            for utterance_id, entry in zip(
                batch_ids, transcribe_batch(batch_audio, processor, model, device, dtype, config)
            ):
                record(utterance_id, entry)
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
