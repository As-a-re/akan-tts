"""Build an LJSpeech-style single-speaker corpus for Coqui VITS.

Two input modes:
  A) local files:  --tsv list.tsv --audio_root /path   (TSV columns: relative_audio_path <TAB> transcript)
  B) HF datasets:  --hf_dataset NAME [--hf_config CFG] --split train --audio_col audio --text_col sentence
                   [--speaker_col speaker_id --speaker_id XYZ]   (keep ONE speaker!)

Output (--out data/akan_ljs):
  wavs/*.wav (22.05 kHz mono 16-bit), metadata_train.csv, metadata_val.csv, stats.json
"""
import argparse, csv, json, random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from normalize import normalize_text, is_valid

SR = 22050


def iter_local(args):
    root = Path(args.audio_root)
    with open(args.tsv, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            p = root / row[0]
            if p.exists():
                y, _ = librosa.load(p, sr=SR, mono=True)
                yield p.stem, y, row[1]


def iter_hf(args):
    """Iterate HF audio rows across both legacy dict and datasets v4 AudioDecoder APIs."""
    from datasets import load_dataset, Audio
    ds = load_dataset(args.hf_dataset, args.hf_config, split=args.split)
    ds = ds.cast_column(args.audio_col, Audio(sampling_rate=SR))
    if args.speaker_col and args.speaker_id is not None:
        ds = ds.filter(lambda ex: ex.get(args.speaker_col) is not None and
                       str(ex[args.speaker_col]) == str(args.speaker_id))
    for i, ex in enumerate(ds):
        raw_text = ex.get(args.text_col)
        # Public corpora can contain null or blank transcripts; do not pass these
        # into normalization or pair them with audio.
        if raw_text is None or not str(raw_text).strip():
            continue
        audio_obj = ex.get(args.audio_col)
        if audio_obj is None:
            continue
        if isinstance(audio_obj, dict):
            audio_array = audio_obj.get("array")
        elif hasattr(audio_obj, "get_all_samples"):
            decoded = audio_obj.get_all_samples()
            audio_array = getattr(decoded, "samples", None)
            if audio_array is None:
                audio_array = getattr(decoded, "data", None)
            if audio_array is None:
                raise TypeError(f"Unsupported decoded audio object: {type(decoded)!r}")
        else:
            raise TypeError(f"Unsupported audio value: {type(audio_obj)!r}")
        if audio_array is None:
            continue
        # AudioDecoder samples are commonly channel-first (channels, time).
        if hasattr(audio_array, "detach"):
            audio_array = audio_array.detach().cpu().numpy()
        audio_array = np.asarray(audio_array, dtype=np.float32)
        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=0) if audio_array.shape[0] <= 8 else audio_array.mean(axis=1)
        audio_array = np.squeeze(audio_array)
        if audio_array.ndim != 1 or audio_array.size == 0:
            continue
        yield f"utt{i:06d}", audio_array, str(raw_text)


def clean_audio(y):
    y, _ = librosa.effects.trim(y, top_db=35)
    pad = np.zeros(int(0.1 * SR), dtype=y.dtype)
    y = np.concatenate([pad, y, pad])
    peak = np.max(np.abs(y)) + 1e-9
    return y / peak * 0.891  # peak-normalise to -1 dBFS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/akan_ljs")
    ap.add_argument("--tsv"); ap.add_argument("--audio_root")
    ap.add_argument("--hf_dataset"); ap.add_argument("--hf_config"); ap.add_argument("--split", default="train")
    ap.add_argument("--audio_col", default="audio"); ap.add_argument("--text_col", default="text")
    ap.add_argument("--speaker_col"); ap.add_argument("--speaker_id")
    ap.add_argument("--min_dur", type=float, default=1.0); ap.add_argument("--max_dur", type=float, default=15.0)
    ap.add_argument("--val_size", type=int, default=30)
    ap.add_argument("--max_hours", type=float, default=None, help="optional cap on total duration")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out = Path(args.out); (out / "wavs").mkdir(parents=True, exist_ok=True)
    src = iter_local(args) if args.tsv else iter_hf(args)
    rows, total, dropped = [], 0.0, {"dur": 0, "text": 0, "rate": 0}
    for uid, y, raw in tqdm(src):
        text = normalize_text(raw)
        if not is_valid(text):
            dropped["text"] += 1; continue
        y = clean_audio(y); dur = len(y) / SR
        if not (args.min_dur <= dur <= args.max_dur):
            dropped["dur"] += 1; continue
        if not (4 <= len(text) / dur <= 25):             # chars/sec sanity check -> catches misaligned pairs
            dropped["rate"] += 1; continue
        sf.write(out / "wavs" / f"{uid}.wav", y, SR, subtype="PCM_16")
        rows.append((uid, text)); total += dur
        if args.max_hours and total >= args.max_hours * 3600:
            break

    random.Random(args.seed).shuffle(rows)
    val, train = rows[:args.val_size], rows[args.val_size:]
    for name, part in (("train", train), ("val", val)):
        with open(out / f"metadata_{name}.csv", "w", encoding="utf-8") as f:
            for uid, t in part:
                f.write(f"{uid}|{t}|{t}\n")
    chars = sorted({c for _, t in rows for c in t})
    stats = {"utterances": len(rows), "train": len(train), "val": len(val),
             "hours": round(total / 3600, 3), "dropped": dropped, "charset": "".join(chars)}
    json.dump(stats, open(out / "stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
