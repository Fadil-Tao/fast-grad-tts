# %% [cell 3] Point the bundled LJSpeech filelists at your Drive wavs
# The repo ships LJSpeech filelists at resources/filelists/ljspeech/{train,valid,
# test}.txt whose paths look like 'DUMMY/LJ050-0234.wav'. We symlink DUMMY at your
# Drive wavs folder so no path rewriting / resampling is needed (LJSpeech is
# already 22.05 kHz, matching params.py).
import os
import params
from utils import parse_filelist

DUMMY = os.path.join(os.getcwd(), 'DUMMY')   # cwd == grad_tts (set in cell 1)
wavs = os.path.join(DATASET_DIR, 'wavs')

if os.path.islink(DUMMY):
    os.remove(DUMMY)
os.symlink(wavs, DUMMY)

# Sanity: the first training entry must resolve to a real wav.
fl = parse_filelist(params.train_filelist_path)
example = fl[0][0]
assert os.path.exists(example), \
    f'Cannot resolve {example}. Check the DUMMY symlink and that wav ids match LJSpeech.'
print(f'train/valid/test items: {len(fl)} / '
      f'{len(parse_filelist(params.valid_filelist_path))} / '
      f'{len(parse_filelist(params.test_filelist_path))}')
print('resolved example:', example)
