"""Fine-tune VITS (Coqui TTS) on the prepared Akan corpus.

Starts from the pretrained English LJSpeech VITS checkpoint (acoustic model + HiFi-GAN-style decoder are
trained jointly in VITS, so there is no separate vocoder). The text-embedding layer has a different size
for the Akan character set, so those weights are re-initialised; every other layer is transferred.
Run:  python scripts/train.py --data data/akan_ljs --out runs/akan_vits --epochs 300
Resume:  add --continue_path runs/akan_vits/<run_folder>
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from normalize import LETTERS


def get_pretrained():
    from TTS.utils.manage import ModelManager
    path, _, _ = ModelManager().download_model("tts_models/en/ljspeech/vits")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/akan_ljs")
    ap.add_argument("--out", default="runs/akan_vits")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no_pretrained", action="store_true", help="train from scratch")
    ap.add_argument("--continue_path", default="")
    ap.add_argument("--fp32", action="store_true", help="disable mixed precision if loss goes NaN")
    a = ap.parse_args()

    ds = BaseDatasetConfig(formatter="ljspeech", meta_file_train="metadata_train.csv",
                           meta_file_val="metadata_val.csv", path=a.data)
    audio = VitsAudioConfig(sample_rate=22050, win_length=1024, hop_length=256, num_mels=80,
                            mel_fmin=0, mel_fmax=None)
    chars = CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>", eos="<EOS>", bos="<BOS>", blank="<BLNK>",
        characters=LETTERS, punctuations=" ,.?!-'\"", phonemes=None,
    )
    config = VitsConfig(
        audio=audio, run_name="akan_vits", batch_size=a.batch_size, eval_batch_size=8,
        batch_group_size=5, num_loader_workers=4, num_eval_loader_workers=2,
        run_eval=True, test_delay_epochs=-1, epochs=a.epochs,
        text_cleaner="basic_cleaners", use_phonemes=False, compute_input_seq_cache=True,
        print_step=25, print_eval=True, mixed_precision=not a.fp32,
        save_step=1000, save_n_checkpoints=3, save_best_after=1000,
        output_path=a.out, datasets=[ds], characters=chars, cudnn_benchmark=False,
        lr_gen=a.lr, lr_disc=a.lr, min_text_len=2, max_text_len=250,
        min_audio_len=22050 * 1, max_audio_len=22050 * 15,
        test_sentences=["Akwaaba.", "Wo ho te sɛn?", "Medaase paa.", "Mepɛ sɛ mekɔ sukuu."],
    )
    ap_ = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    train, val = load_tts_samples(ds, eval_split=True)       # uses meta_file_val
    model = Vits(config, ap_, tokenizer, speaker_manager=None)

    targs = TrainerArgs(continue_path=a.continue_path) if a.continue_path else (
        TrainerArgs() if a.no_pretrained else TrainerArgs(restore_path=get_pretrained()))
    trainer = Trainer(targs, config, a.out, model=model, train_samples=train, eval_samples=val)
    trainer.fit()


if __name__ == "__main__":
    main()
