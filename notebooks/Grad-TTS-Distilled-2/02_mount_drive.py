# %% [cell 2] Mount Drive, set paths, link the dataset
# Distillation needs (a) the trained VANILLA checkpoint as the teacher and (b) the
# LJSpeech data (to draw ground-truth mels for the distillation loss). Students
# are saved to DISTILL_LOG_DIR.
import os
from google.colab import drive

drive.mount('/content/drive')

# --- EDIT THESE ------------------------------------------------------------
DRIVE_ROOT      = '/content/drive/MyDrive/speech-processing'
DATASET_DIR     = os.path.join(DRIVE_ROOT, 'Dataset', 'raw', 'LJSpeech-1.1')  # has wavs/ + metadata.csv
VANILLA_LOG_DIR = os.path.join(DRIVE_ROOT, 'logs', 'Grad-TTS-Vanilla-10')   # teacher ckpts
DISTILL_LOG_DIR = os.path.join(DRIVE_ROOT, 'logs', 'Grad-TTS-Distilled-2')  # students saved here
HIFIGAN_CKPT    = os.path.join(DRIVE_ROOT, 'hifigan.pt')
# ---------------------------------------------------------------------------

os.makedirs(DISTILL_LOG_DIR, exist_ok=True)
assert os.path.isdir(VANILLA_LOG_DIR), \
    f'{VANILLA_LOG_DIR} not found — train Grad-TTS-Vanilla-10 first.'
assert os.path.isdir(os.path.join(DATASET_DIR, 'wavs')), \
    f'Expected LJSpeech wavs at {DATASET_DIR}/wavs'

# Symlink DUMMY -> Drive wavs so the bundled filelists resolve (see vanilla cell 3).
DUMMY = os.path.join(os.getcwd(), 'DUMMY')
if os.path.islink(DUMMY):
    os.remove(DUMMY)
os.symlink(os.path.join(DATASET_DIR, 'wavs'), DUMMY)

print('VANILLA_LOG_DIR =', VANILLA_LOG_DIR)
print('DISTILL_LOG_DIR =', DISTILL_LOG_DIR)
print('HIFIGAN_CKPT    =', HIFIGAN_CKPT, '(exists:', os.path.exists(HIFIGAN_CKPT), ')')
