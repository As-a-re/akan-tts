"""Fine-tune VITS (Coqui TTS) on the prepared Akan corpus.

The script starts from the public English LJSpeech VITS checkpoint by default.
Coqui Trainer performs partial state-dict restoration when the Akan character
embedding shape differs from the English checkpoint, allowing the incompatible
text-embedding weights to be re-initialized while compatible weights are reused.

Run:
  python scripts/train.py --data data/akan_ljs --out runs/akan_vits --epochs 300

Resume an interrupted run:
  python scripts/train.py --data data/akan_ljs --out runs/akan_vits \
      --continue_path runs/akan_vits/<run_folder>
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import torch
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from normalize import LETTERS


def get_pretrained():
    """Download the public Coqui LJSpeech VITS checkpoint and return its path."""
    from TTS.utils.manage import ModelManager

    path, _, _ = ModelManager().download_model("tts_models/en/ljspeech/vits")
    return path


def validate_data(data_dir: Path):
    required = [data_dir / "metadata_train.csv", data_dir / "metadata_val.csv", data_dir / "wavs"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared dataset is incomplete. Missing: " + ", ".join(missing)
        )

    train_n = sum(1 for _ in open(data_dir / "metadata_train.csv", encoding="utf-8"))
    val_n = sum(1 for _ in open(data_dir / "metadata_val.csv", encoding="utf-8"))
    if train_n < 2 or val_n < 1:
        raise ValueError(f"Dataset split is too small: train={train_n}, val={val_n}")
    print(f"Prepared corpus: {train_n} training utterances, {val_n} validation utterances")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/akan_ljs")
    ap.add_argument("--out", default="runs/akan_vits")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no_pretrained", action="store_true", help="train from scratch")
    ap.add_argument("--continue_path", default="", help="existing training folder to resume")
    ap.add_argument("--fp32", action="store_true", help="disable mixed precision if loss goes NaN")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    data_dir = Path(a.data).resolve()
    out_dir = Path(a.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_data(data_dir)

    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    ds = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata_train.csv",
        meta_file_val="metadata_val.csv",
        path=str(data_dir),
    )
    audio = VitsAudioConfig(
        sample_rate=22050,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None,
    )
    chars = CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>", eos="<EOS>", bos="<BOS>", blank="<BLNK>",
        characters=LETTERS,
        punctuations=" ,.?!-'\"",
        phonemes=None,
    )
    config = VitsConfig(
        audio=audio,
        run_name="akan_vits",
        batch_size=a.batch_size,
        eval_batch_size=min(8, a.batch_size),
        batch_group_size=5,
        num_loader_workers=4,
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=a.epochs,
        text_cleaner="basic_cleaners",
        use_phonemes=False,
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=not a.fp32,
        save_step=1000,
        save_n_checkpoints=3,
        save_best_after=1000,
        output_path=str(out_dir),
        datasets=[ds],
        characters=chars,
        cudnn_benchmark=False,
        lr_gen=a.lr,
        lr_disc=a.lr,
        min_text_len=2,
        max_text_len=250,
        min_audio_len=22050 * 1,
        max_audio_len=22050 * 15,
        test_sentences=[
            "Akwaaba.",
            "Wo ho te sɛn?",
            "Medaase paa.",
            "Mepɛ sɛ mekɔ sukuu.",
        ],
    )

    ap_ = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    train, val = load_tts_samples(ds, eval_split=True)
    model = Vits(config, ap_, tokenizer, speaker_manager=None)

    if a.continue_path:
        trainer_args = TrainerArgs(continue_path=a.continue_path)
    elif a.no_pretrained:
        trainer_args = TrainerArgs()
    else:
        pretrained = get_pretrained()
        print(f"Restoring compatible weights from pretrained checkpoint: {pretrained}")
        print("If tensor shapes differ because of the Akan character inventory, Coqui Trainer will use its partial state-dict restoration path.")
        trainer_args = TrainerArgs(restore_path=pretrained)

    # Record the experiment configuration before training.
    experiment = {
        "data": str(data_dir),
        "output": str(out_dir),
        "epochs": a.epochs,
        "batch_size": a.batch_size,
        "learning_rate": a.lr,
        "mixed_precision": not a.fp32,
        "seed": a.seed,
        "pretrained": not a.no_pretrained and not bool(a.continue_path),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (out_dir / "experiment_config.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )

    trainer = Trainer(
        trainer_args,
        config,
        str(out_dir),
        model=model,
        train_samples=train,
        eval_samples=val,
    )
    trainer.fit()

    print("Training finished. Searching for checkpoints...")
    checkpoints = sorted(out_dir.rglob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in checkpoints[:10]:
        print(p)


if __name__ == "__main__":
    main()
