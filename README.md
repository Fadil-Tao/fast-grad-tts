# Fast Grad-TTS (Colab)

Reproduction of the three best variants from
[*Fast Grad-TTS: Towards Efficient Diffusion-Based Speech Generation on CPU*](https://www.isca-speech.org/archive/interspeech_2022/vovk22_interspeech.html)
(Vovk et al., Interspeech 2022), packaged as Google Colab notebooks. Trains on
**LJSpeech** with the dataset and checkpoints living in **Google Drive**.

| Variant | Solver | Steps | Paper MOS / RTF | Notebook |
|---|---|---|---|---|
| Grad-TTS-Vanilla-10 | Euler ODE | 10 | 4.00 / 0.68 | `notebooks/Grad-TTS-Vanilla-10/` |
| Grad-TTS-ML-4 | Maximum Likelihood SDE | 4 | 3.93 / 0.28 | `notebooks/Grad-TTS-ML-4/` |
| Grad-TTS-Distilled-2 | DDIM (progressive distillation) | 2 | 3.82 / 0.15 | `notebooks/Grad-TTS-Distilled-2/` |

ML-4 and Distilled-2 build on the **same trained vanilla model** — train it once.
ML-4 needs no extra training (sampler swap only); Distilled-2 distills the vanilla
model down to 2 steps.

## Layout

```
fast-grad-tts/
  grad_tts/        # Grad-TTS source (Huawei) + our additions:
                   #   model/diffusion.py : ML & DDIM solvers (reverse_diffusion_ml/_ddim, ddim_step, _estimate_x0)
                   #   model/tts.py       : `solver` arg + compute_alignment()
                   #   distill.py         : progressive distillation (Sec. 3.2)
                   #   inference.py       : --solver {original,ml,ddim}
  notebooks/
    Grad-TTS-Vanilla-10/   01_setup .. 06_synthesize   (file = Colab cell)
    Grad-TTS-ML-4/         01_setup .. 05_benchmark
    Grad-TTS-Distilled-2/  01_setup .. 05_benchmark
  requirements-modern.txt
```

## How to run on Colab

Each `NN_*.py` file is one **cell**: paste its contents into a Colab cell and run
the cells of a variant in numeric order, in a single GPU session.

1. **Clone** your fork at the start of the session:
   `!git clone https://github.com/<you>/fast-grad-tts /content/fast-grad-tts`
   (cell 1 assumes `REPO_DIR=/content/fast-grad-tts`; edit if different).
2. Run `01_setup.py` (installs deps, builds the Cython MAS extension) and
   `02_mount_drive.py` (mounts Drive, sets paths — **edit the paths at the top**).
3. Train / synthesize per the remaining cells.

### Drive layout expected

```
MyDrive/speech-processing/
  Dataset/                 # LJSpeech-1.1
    wavs/*.wav             # 22.05 kHz
    metadata.csv
  hifigan.pt               # HiFi-GAN vocoder checkpoint (see below)
  logs/                    # checkpoints written here (auto-created)
    Grad-TTS-Vanilla-10/
    Grad-TTS-Distilled-2/
```

### HiFi-GAN vocoder

Synthesis needs a HiFi-GAN checkpoint at `MyDrive/speech-processing/hifigan.pt`.
Download the LJSpeech HiFi-GAN checkpoint from the official Grad-TTS release
(Huawei Speech-Backbones Google Drive) and drop it there. The matching config
`checkpts/hifigan-config.json` already ships in `grad_tts/`.

## Notes

- **numpy<2** is pinned (repo + hifi-gan use legacy numpy APIs). If Colab already
  imported numpy 2, restart the runtime after `01_setup.py`.
- Checkpoints are written to Drive and the train cell resumes from the newest one,
  so training survives Colab disconnects.
- Distillation is heavy: start `ROUNDS`/`EPOCHS_PER_ROUND` small to smoke-test,
  then scale to the paper's 5 rounds × 35 epochs.

See `grad_tts/SETUP_TRAIN.md` for a plain (non-Colab) training walkthrough.
