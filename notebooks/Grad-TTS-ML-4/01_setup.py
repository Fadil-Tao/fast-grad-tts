# %% [cell 1] Setup — install deps, build Monotonic Alignment Search, set paths
# Paste each NN_*.py file into its own Colab cell and run them top to bottom in
# one session. Colab already provides a CUDA build of torch + torchaudio, so we
# only install the remaining requirements here. Run this cell once per session.
import os
import sys
import subprocess

# Where you cloned this repo inside Colab (e.g. after `!git clone <your-fork>`).
REPO_DIR = '/content/fast-grad-tts'
GRAD_TTS_DIR = os.path.join(REPO_DIR, 'grad_tts')

# Python deps. requirements-modern.txt pins numpy<2 because the repo + hifi-gan
# use legacy numpy APIs (np.float etc). If Colab had numpy 2 already imported you
# may need to: Runtime > Restart session, then re-run this cell.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                '-r', os.path.join(REPO_DIR, 'requirements-modern.txt')], check=True)

# Build the Cython Monotonic Alignment Search extension (compiled .so is
# arch-specific, so rebuild it every Colab session — it is quick).
subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
               cwd=os.path.join(GRAD_TTS_DIR, 'model', 'monotonic_align'), check=True)

# Run from grad_tts so its package-relative imports work unchanged
# (`import params`, `from model import GradTTS`, `from data import ...`).
os.chdir(GRAD_TTS_DIR)
if GRAD_TTS_DIR not in sys.path:
    sys.path.insert(0, GRAD_TTS_DIR)

import torch
print('Setup done. cwd =', os.getcwd(), '| CUDA available:', torch.cuda.is_available())
