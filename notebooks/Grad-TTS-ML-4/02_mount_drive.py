# %% [cell 2] Mount Drive — ML-4 REUSES the vanilla checkpoint (no retraining)
# The Maximum Likelihood SDE solver is inference-only: it runs on the exact same
# trained weights as Grad-TTS-Vanilla-10, just with a better sampler and fewer
# steps. So point VANILLA_LOG_DIR at the directory where the vanilla notebook
# saved its grad_<epoch>.pt checkpoints.
import os
from google.colab import drive

drive.mount('/content/drive')

# --- EDIT THESE ------------------------------------------------------------
DRIVE_ROOT      = '/content/drive/MyDrive/speech-processing'
VANILLA_LOG_DIR = os.path.join(DRIVE_ROOT, 'logs', 'Grad-TTS-Vanilla-10')  # trained vanilla ckpts
HIFIGAN_CKPT    = os.path.join(DRIVE_ROOT, 'hifigan.pt')
# ---------------------------------------------------------------------------

assert os.path.isdir(VANILLA_LOG_DIR), \
    f'{VANILLA_LOG_DIR} not found — train Grad-TTS-Vanilla-10 first.'
print('VANILLA_LOG_DIR =', VANILLA_LOG_DIR)
print('HIFIGAN_CKPT    =', HIFIGAN_CKPT, '(exists:', os.path.exists(HIFIGAN_CKPT), ')')
