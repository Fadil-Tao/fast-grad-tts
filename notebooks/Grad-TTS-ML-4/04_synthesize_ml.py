# %% [cell 4] Synthesize with the Maximum Likelihood SDE solver (4 steps)
# Same weights as vanilla, but solver='ml' and only 4 reverse-diffusion steps.
# Expect a large speedup vs vanilla-10 with near-identical quality (paper: MOS
# 4.00 -> 3.93, RTF 0.68 -> 0.28 on CPU).
import datetime as dt
import numpy as np
import torch

from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
from scipy.io.wavfile import write
from IPython.display import Audio, display

N_TIMESTEPS = 4
SOLVER = 'ml'
SENTENCES = [
    "Grad TTS turns noise into speech by reversing a diffusion process.",
]

cmu = cmudict.CMUDict('resources/cmu_dictionary')
for i, text in enumerate(SENTENCES):
    x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu),
                                     len(symbols))).cuda()[None]
    x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
    t0 = dt.datetime.now()
    with torch.no_grad():
        y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=N_TIMESTEPS, temperature=1.5,
                                   stoc=False, length_scale=0.91, solver=SOLVER)
        audio = (vocoder(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
    rtf = (dt.datetime.now() - t0).total_seconds() * 22050 / (y_dec.shape[-1] * 256)
    print(f'sentence {i} | solver={SOLVER} steps={N_TIMESTEPS} | RTF {rtf:.3f}')
    write(f'sample_ml_{i}.wav', 22050, audio)
    display(Audio(audio, rate=22050))
