# %% [cell 5] Benchmark — vanilla Euler (10 steps) vs ML SDE (4 steps)
# Same checkpoint, same sentences. Reports mean RTF per solver and plays both so
# you can compare quality by ear. Lower RTF = faster.
import datetime as dt
import numpy as np
import torch

from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
from IPython.display import Audio, display

CONFIGS = [
    ('original', 10),   # vanilla baseline
    ('ml', 4),          # Fast Grad-TTS ML solver
]
SENTENCES = [
    "Grad TTS turns noise into speech by reversing a diffusion process.",
    "The maximum likelihood solver needs far fewer steps.",
]

cmu = cmudict.CMUDict('resources/cmu_dictionary')


def synth(text, solver, steps):
    x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu),
                                     len(symbols))).cuda()[None]
    x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
    t0 = dt.datetime.now()
    with torch.no_grad():
        _, y_dec, _ = model(x, x_lengths, n_timesteps=steps, temperature=1.5,
                            stoc=False, length_scale=0.91, solver=solver)
        audio = (vocoder(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
    rtf = (dt.datetime.now() - t0).total_seconds() * 22050 / (y_dec.shape[-1] * 256)
    return audio, rtf


for solver, steps in CONFIGS:
    rtfs = []
    print(f'\n=== solver={solver} steps={steps} ===')
    for i, text in enumerate(SENTENCES):
        audio, rtf = synth(text, solver, steps)
        rtfs.append(rtf)
        print(f'  sentence {i} | RTF {rtf:.3f}')
        display(Audio(audio, rate=22050))
    print(f'  mean RTF = {np.mean(rtfs):.3f}')
