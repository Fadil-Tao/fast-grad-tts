# Grad-TTS — train-device setup (LJSpeech) + Fast Grad-TTS ML solver

One-shot guide to set up, train vanilla Grad-TTS on LJSpeech, and run inference
with the Maximum Likelihood SDE solver from *Fast Grad-TTS* (Sec. 3.1).

## 1. Environment

Original repo targets Python 3.6 / torch 1.9. On a modern GPU box use Python
3.10–3.11 and a CUDA torch build:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements-modern.txt
```

## 2. Build Monotonic Alignment Search (Cython, required)

```bash
cd model/monotonic_align
python setup.py build_ext --inplace
cd ../..
```

## 3. Dataset — LJSpeech

```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjf LJSpeech-1.1.tar.bz2
# filelists reference paths like DUMMY/LJ050-0234.wav -> point DUMMY at the wavs:
ln -s /abs/path/to/LJSpeech-1.1/wavs DUMMY
```

Filelists already in `resources/filelists/ljspeech/` (train 11946 / valid 94 /
test 487). They expect `DUMMY/<id>.wav|<text>`.

## 4. Config — `params.py`

Single-speaker LJSpeech defaults are already set (`n_spks=1`, 22.05kHz,
`n_feats=80`). Set before training:
- `log_dir` — where checkpoints/tensorboard go
- `batch_size` — fit to GPU VRAM (default 16)
- `pe_scale=1000` (keep, matches released `grad-tts.pt`)

## 5. Train

```bash
export CUDA_VISIBLE_DEVICES=0
python train.py
tensorboard --logdir=<log_dir>   # optional, to watch
```

## 6. Inference — vanilla vs ML solver

Needs a HiFi-GAN vocoder checkpoint in `checkpts/` (download from the repo's
Google Drive; config `checkpts/hifigan-config.json` already present).

```bash
# vanilla Euler solver, 10 steps (baseline)
python inference.py -f resources/filelists/synthesis.txt -c <ckpt>.pt -t 10

# Fast Grad-TTS ML solver, 4 steps (~2.4x faster, near-equal MOS)
python inference.py -f resources/filelists/synthesis.txt -c <ckpt>.pt -t 4 --solver ml
```

`--solver ml` requires NO retraining — it reuses the trained decoder weights and
only changes the reverse-diffusion sampler. RTF printed per utterance.

## What changed vs upstream Grad-TTS

- `model/diffusion.py` — added `reverse_diffusion_ml()` (eq. 6-9) + `solver` arg
  on `Diffusion.forward`.
- `model/tts.py` — `GradTTS.forward` forwards a `solver` argument to the decoder.
- `inference.py` — new `--solver {original,ml}` flag.

Training path is unchanged. Next acceleration step (not yet implemented):
progressive distillation (Sec. 3.2) to reach 2 steps / RTF 0.15.
