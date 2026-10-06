# Akan (Asante Twi) Text-to-Speech — VITS Fine-tuning

**Selected variety:** Asante Twi (`aka`)

This project implements a local Text-to-Speech pipeline for Akan speech using VITS. It does **not** use an external TTS API for synthesis.

## End-to-end pipeline

1. Verify an openly licensed Akan/Twi dataset and its metadata.
2. Normalize Akan text while preserving `ɛ` and `ɔ`.
3. Expand supported numbers and clean punctuation/Unicode.
4. Resample, mono-convert, trim silence, peak-normalize, and duration-filter audio.
5. Create train/validation LJSpeech-style metadata.
6. Fine-tune VITS from the public English LJSpeech VITS checkpoint, or train from scratch if explicitly selected.
7. Generate speech from unseen Akan text locally from the trained checkpoint.
8. Evaluate synthesized speech with Akan-capable MMS ASR using WER/CER.
9. Generate a randomized listener/MOS sheet covering naturalness, intelligibility, `ɛ`, `ɔ`, and tone/prosody.
10. Package the checkpoint, configuration, prompts, audio samples, statistics, and evaluation outputs.

## Google Colab — recommended

Open `akan_tts_colab.ipynb` in Google Colab and select **Runtime → Change runtime type → GPU**.

The notebook can run in either of these ways:

### Option A — upload the ZIP

Upload `akan-tts.zip` when the notebook asks for it.

### Option B — clone a GitHub repository

Set `REPO_URL` in the notebook to the repository containing this project.

The notebook mounts Google Drive and stores the long-running experiment under:

```text
MyDrive/akan_tts/
├── runs/
├── samples/
├── eval/
├── experiment_manifest.json
└── submission_package/
```

This prevents a Colab runtime reset from deleting the experiment artifacts.

## Dataset

The default configuration is:

```text
Dataset: google/WaxalNLP
Config:  twi_tts
Split:   train
Audio:   audio
Text:    transcription
```

**Do not copy an unverified dataset duration, speaker identity, or license into the report.** The Colab notebook prints the dataset metadata and actual prepared-corpus statistics so those values can be recorded from the experiment.

## Training

The default training command is equivalent to:

```bash
python scripts/train.py \
  --data data/akan_ljs \
  --out runs/akan_vits \
  --epochs 300 \
  --batch_size 16 \
  --lr 1e-4
```

The training script records the experiment configuration and uses Coqui Trainer's restore mechanism for the pretrained VITS checkpoint. If the Akan character inventory differs from the English checkpoint, Coqui Trainer can fall back to partial state-dict restoration for incompatible tensors while restoring compatible weights.

To resume an interrupted run:

```bash
python scripts/train.py \
  --data data/akan_ljs \
  --out runs/akan_vits \
  --continue_path runs/akan_vits/<training-folder>
```

## Inference

After training:

```bash
python scripts/infer.py \
  --ckpt runs/akan_vits/<run>/best_model.pth \
  --config runs/akan_vits/<run>/config.json \
  --file data/test_sentences.txt \
  --out samples \
  --cuda
```

The inference script creates:

```text
samples/
├── sample_01.wav
├── sample_02.wav
├── ...
└── prompts.tsv
```

The supplied test set contains unseen Akan prompts intended for the demonstration requirement.

## Evaluation

ASR round-trip evaluation:

```bash
python scripts/evaluate.py asr \
  --samples samples \
  --out eval/asr_results.csv
```

Listener evaluation sheet:

```bash
python scripts/evaluate.py mos \
  --samples samples \
  --out eval/mos_sheet.csv
```

The listener sheet should be completed by Twi-speaking listeners before the report is finalized.

## Important submission requirement

The project is **not considered complete merely because the code runs**. Before submission, retain evidence of:

- completed preprocessing;
- completed training/fine-tuning;
- a final/best checkpoint;
- at least five synthesized unseen-text examples;
- corresponding text prompts;
- WER/CER results;
- listener evaluation results;
- dataset/license/speaker details;
- completed report with real experimental values;
- presentation containing the required demonstration examples.

## Own work vs reused components

**Project-specific work:**

- Akan text normalization;
- `ɛ`/`ɔ` character handling;
- supported Twi number expansion;
- dataset cleaning/filtering/splitting;
- VITS configuration for the Akan grapheme inventory;
- inference wrapper;
- evaluation wrapper;
- unseen-text test set;
- listener evaluation sheet;
- experiment artifact packaging.

**Reused components:**

- Coqui TTS / maintained `coqui-tts` implementation and Trainer;
- public English LJSpeech VITS pretrained checkpoint;
- VITS architecture;
- MMS ASR for objective evaluation;
- `jiwer`, `librosa`, `datasets`, and other open-source dependencies.

All external resources must be acknowledged in the final report and presentation, together with their applicable licenses.

## Known language limitations

The current design uses graphemes rather than a dedicated Akan G2P system. Standard Twi orthography does not explicitly mark tone, so tone is not independently supervised. The report should therefore discuss observed tone/prosody behavior rather than claiming that tone has been solved. The project also retains `ɛ` and `ɔ` as separate symbols and evaluates them explicitly.
