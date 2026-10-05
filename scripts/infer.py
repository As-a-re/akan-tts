"""Synthesise Akan speech from unseen text (no external TTS API; local checkpoint only).

  python scripts/infer.py --ckpt runs/akan_vits/<run>/best_model.pth --config runs/akan_vits/<run>/config.json \
      --text "Akwaaba, wo ho te sɛn?" --out samples/demo.wav
  python scripts/infer.py ... --file data/test_sentences.txt --out samples/
"""
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from TTS.utils.synthesizer import Synthesizer
from normalize import normalize_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--text"); ap.add_argument("--file"); ap.add_argument("--out", default="samples")
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    synth = Synthesizer(tts_checkpoint=a.ckpt, tts_config_path=a.config, use_cuda=a.cuda)

    lines = [a.text] if a.text else [l.strip() for l in open(a.file, encoding="utf-8") if l.strip()]
    out = Path(a.out)
    if len(lines) == 1 and out.suffix == ".wav":
        targets = [out]; out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
        targets = [out / f"sample_{i + 1:02d}.wav" for i in range(len(lines))]
    with open(targets[0].parent / "prompts.tsv", "w", encoding="utf-8") as f:
        for line, path in zip(lines, targets):
            norm = normalize_text(line)
            synth.save_wav(synth.tts(norm), str(path))
            f.write(f"{path.name}\t{line}\t{norm}\n")
            print(path.name, "<-", norm)


if __name__ == "__main__":
    main()
