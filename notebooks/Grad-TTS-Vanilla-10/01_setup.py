# %% [cell 1] Setup — clone repo, install deps, build Monotonic Alignment Search
# Self-contained: paste this into the FIRST Colab cell and run it. It clones the
# repo, installs the few packages Colab lacks, builds the Cython MAS extension and
# sets up the import path. Run the remaining NN_*.py cells (one file = one cell)
# top to bottom afterwards, in the same GPU session.
import os
import sys
import subprocess

REPO_URL = 'https://github.com/Fadil-Tao/fast-grad-tts.git'
REPO_DIR = '/content/fast-grad-tts'
GRAD_TTS_DIR = os.path.join(REPO_DIR, 'grad_tts')

# Clone on first run; if the repo dir exists but is incomplete (stale/empty
# clone) wipe and re-clone; otherwise just fast-forward to the latest commit.
if not os.path.isdir(GRAD_TTS_DIR):
    subprocess.run(['rm', '-rf', REPO_DIR], check=False)
    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(['git', '-C', REPO_DIR, 'pull', '--ff-only'], check=False)

# Colab already ships torch, torchaudio, numpy, scipy, matplotlib, librosa, tqdm
# and Cython, so we only add the few pure-Python packages it lacks. (numpy is not
# touched, to avoid fighting Colab's preinstalled numpy-2 stack.)
subprocess.run([sys.executable, '-m', 'pip', 'install',
                'einops>=0.6', 'inflect>=6.0', 'Unidecode>=1.3'], check=True)

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
