# %% [cell 2] Mount Google Drive and define dataset / checkpoint paths
import os
from google.colab import drive

drive.mount('/content/drive')

# --- EDIT THESE to match your Drive layout ---------------------------------
# LJSpeech is expected at DATASET_DIR/wavs/*.wav (22.05 kHz) + DATASET_DIR/metadata.csv.
DRIVE_ROOT   = '/content/drive/MyDrive/speech-processing'
DATASET_DIR  = os.path.join(DRIVE_ROOT, 'Dataset')                      # LJSpeech root
LOG_DIR      = os.path.join(DRIVE_ROOT, 'logs', 'Grad-TTS-Vanilla-10')  # checkpoints persist here
HIFIGAN_CKPT = os.path.join(DRIVE_ROOT, 'hifigan.pt')                   # vocoder, see README
# ---------------------------------------------------------------------------

os.makedirs(LOG_DIR, exist_ok=True)
assert os.path.isdir(os.path.join(DATASET_DIR, 'wavs')), \
    f'Expected LJSpeech wavs at {DATASET_DIR}/wavs'
print('DATASET_DIR  =', DATASET_DIR)
print('LOG_DIR      =', LOG_DIR)
print('HIFIGAN_CKPT =', HIFIGAN_CKPT, '(exists:', os.path.exists(HIFIGAN_CKPT), ')')
