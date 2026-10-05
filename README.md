# Akan (Asante Twi) Text-to-Speech — VITS fine-tuning

**Variety:** Asante Twi (`aka`). **Model:** VITS (end-to-end acoustic model + waveform decoder) fine-tuned from the
English LJSpeech VITS checkpoint, with a grapheme front end that keeps ɛ and ɔ as distinct symbols.
No external TTS API is used; inference runs from the local checkpoint.

## Pipeline
| Stage | File |
|---|---|
| Text normalisation (NFC, ɛ/ɔ variants, tone-mark stripping, digits 0–99 → Twi words) | `scripts/normalize.py` |
| Audio/text preprocessing → LJSpeech-style corpus | `scripts/prepare_data.py` |
| Fine-tuning | `scripts/train.py` |
| Inference from unseen text | `scripts/infer.py` |
| ASR round-trip WER/CER + MOS sheet | `scripts/evaluate.py` |
| One-click run (Colab, GPU) | `akan_tts_colab.ipynb` |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # needs a CUDA GPU for training (Colab T4 is enough)
```

## Reproduce
```bash
python scripts/prepare_data.py --hf_dataset google/WaxalNLP --hf_config twi_tts --split train --text_col transcription --audio_col audio --out data/akan_ljs
python scripts/train.py --data data/akan_ljs --out runs/akan_vits --epochs 300
python scripts/infer.py --ckpt runs/akan_vits/<run>/best_model.pth --config runs/akan_vits/<run>/config.json \
       --file data/test_sentences.txt --out samples --cuda
python scripts/evaluate.py asr --samples samples --out eval/asr_results.csv
python scripts/evaluate.py mos --samples samples --out eval/mos_sheet.csv
```
Trained checkpoint: **<LINK — add after training>**. Samples + prompts: **<LINK>**.

## Own work vs reused components
- **Own:** text normaliser, Twi number expansion, data cleaning/filtering/splitting, training configuration, inference
  wrapper, evaluation scripts, test set, listener sheet.
- **Reused:** Coqui TTS / `coqui-tts` fork (VITS implementation, Trainer; MPL-2.0), pretrained LJSpeech VITS checkpoint
  (Coqui model zoo), Kim et al. 2021 VITS paper, MMS-1b-all ASR (Meta, CC-BY-NC 4.0 — used for evaluation only),
  `jiwer`, `librosa`. **Dataset:** WAXAL (google/WaxalNLP, config `twi_tts`), Google + University of Ghana, CC-BY-4.0 (verify on dataset card); Katumba et al., 2026, arXiv:2602.02734.

## Known limitations
Tone is not marked in Twi orthography and is not modelled explicitly; numbers above 99 are read digit by digit;
the number words and the test sentences should be checked by a native speaker.
