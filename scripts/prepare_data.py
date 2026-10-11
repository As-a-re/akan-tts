
"""Prepare a single-speaker LJSpeech-style corpus for Coqui VITS."""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from normalize import normalize_text, is_valid

SR = 22050


def to_numpy(value):
    """Convert arrays or PyTorch tensors to NumPy."""
    if value is None:
        return None

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()

    return np.asarray(value)


def decode_audio(audio_obj):
    """Decode old Hugging Face audio dictionaries or AudioDecoder objects."""
    if audio_obj is None:
        raise ValueError("Missing audio")

    if isinstance(audio_obj, dict):
        y = audio_obj.get("array")
        original_sr = audio_obj.get("sampling_rate")

        if y is None:
            raise ValueError("Audio dictionary has no array")

    elif hasattr(audio_obj, "get_all_samples"):
        decoded = audio_obj.get_all_samples()

        y = getattr(decoded, "data", None)
        if y is None:
            y = getattr(decoded, "samples", None)

        original_sr = getattr(decoded, "sample_rate", None)

        if y is None:
            raise ValueError("AudioDecoder returned no samples")

    else:
        raise TypeError(
            f"Unsupported audio object: {type(audio_obj).__name__}"
        )

    y = to_numpy(y)

    if y is None or y.size == 0:
        raise ValueError("Audio is empty")

    # AudioDecoder commonly returns [channels, samples].
    # Convert stereo/multichannel audio to mono.
    if y.ndim == 2:
        if y.shape[0] <= 8 and y.shape[1] > y.shape[0]:
            y = y.mean(axis=0)
        elif y.shape[1] <= 8:
            y = y.mean(axis=1)
        else:
            raise ValueError(f"Unexpected audio shape: {y.shape}")

    elif y.ndim != 1:
        raise ValueError(f"Unexpected audio dimensions: {y.shape}")

    y = y.astype(np.float32, copy=False)

    if not np.isfinite(y).all():
        raise ValueError("Audio contains NaN or infinite values")

    if not original_sr:
        raise ValueError("Audio sample rate is missing")

    original_sr = int(original_sr)

    if original_sr != SR:
        y = librosa.resample(
            y,
            orig_sr=original_sr,
            target_sr=SR
        )

    if y.size == 0:
        raise ValueError("Audio is empty after resampling")

    return y


def iter_local(args):
    root = Path(args.audio_root)

    with open(args.tsv, encoding="utf-8") as f:
        for index, row in enumerate(csv.reader(f, delimiter="\t")):
            if len(row) < 2:
                continue

            path = root / row[0]

            if not path.exists():
                print(f"Missing local audio: {path}")
                continue

            try:
                y, _ = librosa.load(path, sr=SR, mono=True)
                yield path.stem, y, row[1]
            except Exception as exc:
                print(f"Skipping {path}: {exc}")


def iter_hf(args):
    from datasets import load_dataset

    print(
        f"Loading {args.hf_dataset}, config={args.hf_config}, "
        f"split={args.split}"
    )

    ds = load_dataset(
        args.hf_dataset,
        args.hf_config,
        split=args.split
    )

    print("Rows loaded:", len(ds))
    print("Columns:", ds.column_names)

    required = [args.audio_col, args.text_col]
    missing = [c for c in required if c not in ds.column_names]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available: {ds.column_names}"
        )

    # Build a valid transcript index before decoding audio.
    valid_rows = []

    for i, ex in enumerate(ds):
        raw_text = ex.get(args.text_col)

        if not isinstance(raw_text, str) or not raw_text.strip():
            continue

        if args.speaker_col:
            if args.speaker_col not in ds.column_names:
                raise ValueError(
                    f"Speaker column '{args.speaker_col}' not found"
                )

            speaker = ex.get(args.speaker_col)
            if speaker is None or not str(speaker).strip():
                continue
        else:
            speaker = None

        valid_rows.append((i, str(speaker) if speaker is not None else None))

    print("Rows with non-empty transcripts:", len(valid_rows))

    if not valid_rows:
        raise ValueError(
            "No non-empty transcripts found. Check --text_col and split."
        )

    # VITS configuration here is single-speaker. If no speaker ID was
    # specified, choose the speaker with the most non-empty transcripts.
    selected_speaker = args.speaker_id

    if args.speaker_col and selected_speaker is None:
        counts = Counter(s for _, s in valid_rows if s is not None)

        if not counts:
            raise ValueError("No valid speaker IDs were found.")

        selected_speaker, count = counts.most_common(1)[0]

        print(
            f"Automatically selected speaker: {selected_speaker!r} "
            f"({count} rows with transcripts)"
        )

    if args.speaker_col and selected_speaker is not None:
        valid_rows = [
            (i, speaker)
            for i, speaker in valid_rows
            if speaker == str(selected_speaker)
        ]

        print("Rows for selected speaker:", len(valid_rows))

    if not valid_rows:
        raise ValueError(
            "No rows remain after speaker filtering. "
            "Check the selected speaker ID."
        )

    # Decode each record individually so one damaged recording does not
    # abort the entire preprocessing job.
    decode_failures = 0

    for i, _speaker in tqdm(valid_rows, desc="Decoding audio"):
        ex = ds[i]
        raw_text = ex.get(args.text_col)

        try:
            y = decode_audio(ex.get(args.audio_col))
            uid = f"utt{i:06d}"
            yield uid, y, raw_text

        except Exception as exc:
            decode_failures += 1
            print(
                f"\nSkipping row {i} due to audio error: {exc}",
                file=sys.stderr
            )

    print(f"Audio decoding failures: {decode_failures}")


def clean_audio(y):
    y = np.asarray(y, dtype=np.float32)

    if y.size == 0:
        raise ValueError("Cannot clean empty audio")

    y, _ = librosa.effects.trim(y, top_db=35)

    if y.size == 0:
        raise ValueError("Audio became empty after trimming")

    padding = np.zeros(int(0.1 * SR), dtype=np.float32)
    y = np.concatenate([padding, y, padding])

    peak = float(np.max(np.abs(y)))

    if not np.isfinite(peak) or peak < 1e-8:
        raise ValueError("Audio is silent or invalid")

    return y / peak * 0.891


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", default="data/akan_ljs")
    parser.add_argument("--tsv")
    parser.add_argument("--audio_root")

    parser.add_argument("--hf_dataset")
    parser.add_argument("--hf_config")
    parser.add_argument("--split", default="train")

    parser.add_argument("--audio_col", default="audio")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--speaker_col")
    parser.add_argument("--speaker_id")

    parser.add_argument("--min_dur", type=float, default=1.0)
    parser.add_argument("--max_dur", type=float, default=15.0)
    parser.add_argument("--val_size", type=int, default=30)
    parser.add_argument("--max_hours", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)

    # Adjustable because character-per-second limits may not fit every
    # language, speaking style, or transcript convention.
    parser.add_argument("--min_chars_sec", type=float, default=4.0)
    parser.add_argument("--max_chars_sec", type=float, default=25.0)

    args = parser.parse_args()

    if args.tsv and not args.audio_root:
        parser.error("--audio_root is required when --tsv is used")

    if not args.tsv and not args.hf_dataset:
        parser.error("Provide either --tsv or --hf_dataset")

    if args.val_size < 1:
        parser.error("--val_size must be at least 1")

    out = Path(args.out)
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # Clear old metadata so a failed run cannot be mistaken for a
    # successful preprocessing result.
    for name in ("metadata_train.csv", "metadata_val.csv"):
        path = out / name
        if path.exists():
            path.unlink()

    source = iter_local(args) if args.tsv else iter_hf(args)

    rows = []
    total_seconds = 0.0
    dropped = {
        "text": 0,
        "audio": 0,
        "duration": 0,
        "character_rate": 0,
    }

    for uid, audio, raw_text in tqdm(source, desc="Preparing corpus"):
        try:
            text = normalize_text(raw_text)

            if not isinstance(text, str) or not is_valid(text):
                dropped["text"] += 1
                continue

            y = clean_audio(audio)
            duration = len(y) / SR

            if not (args.min_dur <= duration <= args.max_dur):
                dropped["duration"] += 1
                continue

            chars_per_second = len(text) / duration

            if not (
                args.min_chars_sec
                <= chars_per_second
                <= args.max_chars_sec
            ):
                dropped["character_rate"] += 1
                continue

            wav_path = wav_dir / f"{uid}.wav"
            sf.write(wav_path, y, SR, subtype="PCM_16")

            rows.append((uid, text))
            total_seconds += duration

            if args.max_hours and total_seconds >= args.max_hours * 3600:
                break

        except Exception as exc:
            dropped["audio"] += 1
            print(f"\nSkipping {uid}: {exc}", file=sys.stderr)

    if len(rows) < 3:
        raise RuntimeError(
            f"Only {len(rows)} usable utterances were prepared. "
            f"Drop counts: {dropped}. No metadata was written. "
            "Inspect the errors and adjust preprocessing parameters."
        )

    random.Random(args.seed).shuffle(rows)

    val_size = min(args.val_size, len(rows) - 2)
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    for name, partition in (
        ("train", train_rows),
        ("val", val_rows),
    ):
        metadata_path = out / f"metadata_{name}.csv"

        with open(metadata_path, "w", encoding="utf-8", newline="") as f:
            for uid, text in partition:
                f.write(f"{uid}|{text}|{text}\n")

    # Verify every metadata row points to an existing WAV file.
    for uid, _text in rows:
        if not (wav_dir / f"{uid}.wav").exists():
            raise FileNotFoundError(f"Missing WAV for utterance: {uid}")

    charset = sorted({c for _, text in rows for c in text})

    stats = {
        "utterances": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "hours": round(total_seconds / 3600, 3),
        "sample_rate": SR,
        "speaker_id": args.speaker_id,
        "dropped": dropped,
        "charset": "".join(charset),
    }

    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\nPreprocessing completed:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("Training metadata:", out / "metadata_train.csv")
    print("Validation metadata:", out / "metadata_val.csv")


if __name__ == "__main__":
    main()
