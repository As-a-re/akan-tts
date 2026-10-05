"""Objective + subjective evaluation helpers.

1) ASR round-trip intelligibility: transcribe synthesised audio with an Akan-capable ASR (MMS-1b-all, adapter 'aka')
   and compute WER/CER against the input text. Caveat: the ASR's own errors are mixed into the score, so also run
   the same ASR on REAL held-out recordings (put them in a folder with the same prompts.tsv layout) as a ceiling.
2) MOS sheet: writes a randomised CSV for listeners (naturalness, intelligibility, mispronunciations, notes).

  python scripts/evaluate.py asr --samples samples/ --out eval/asr_results.csv
  python scripts/evaluate.py mos --samples samples/ --out eval/mos_sheet.csv
"""
import argparse, csv, random
from pathlib import Path


def asr(a):
    import torch, librosa, jiwer
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
    proc.tokenizer.set_target_lang("aka"); model.load_adapter("aka")
    dev = "cuda" if torch.cuda.is_available() else "cpu"; model.to(dev).eval()

    rows = [l.rstrip("\n").split("\t") for l in open(Path(a.samples) / "prompts.tsv", encoding="utf-8")]
    res, refs, hyps = [], [], []
    for fn, raw, norm in rows:
        y, _ = librosa.load(Path(a.samples) / fn, sr=16000)
        inp = proc(y, sampling_rate=16000, return_tensors="pt").to(dev)
        with torch.no_grad():
            ids = torch.argmax(model(**inp).logits, dim=-1)[0]
        hyp = proc.decode(ids).lower().strip()
        refs.append(norm); hyps.append(hyp)
        res.append({"file": fn, "ref": norm, "asr": hyp, "wer": jiwer.wer(norm, hyp), "cer": jiwer.cer(norm, hyp)})
    print(f"Corpus WER={jiwer.wer(refs, hyps):.3f}  CER={jiwer.cer(refs, hyps):.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)


def mos(a):
    rows = [l.rstrip("\n").split("\t") for l in open(Path(a.samples) / "prompts.tsv", encoding="utf-8")]
    random.Random(7).shuffle(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "text", "naturalness_1to5", "intelligibility_1to5",
                    "mispronounced_words", "open_e_ok(Y/N)", "open_o_ok(Y/N)", "tone_or_prosody_notes"])
        for fn, raw, _ in rows:
            w.writerow([fn, raw, "", "", "", "", "", ""])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["asr", "mos"])
    ap.add_argument("--samples", default="samples"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); {"asr": asr, "mos": mos}[a.mode](a)
