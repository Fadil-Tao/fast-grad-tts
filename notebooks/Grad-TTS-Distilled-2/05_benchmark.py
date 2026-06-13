# %% [cell 5] Benchmark all three variants on the same sentences
# vanilla weights -> {original 10 steps, ml 4 steps}; distilled student -> {ddim 2
# steps}. Reports mean RTF per config and plays each for quality comparison.
import os
import sys
import glob
import json
import datetime as dt
import numpy as np
import torch

sys.path.append('hifi-gan')
import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
from env import AttrDict
from models import Generator as HiFiGAN
from IPython.display import Audio, display

params.n_spks = 1
nsymbols = len(symbols) + 1 if params.add_blank else len(symbols)


def _build(ckpt):
    m = GradTTS(nsymbols, params.n_spks, params.spk_emb_dim, params.n_enc_channels,
                params.filter_channels, params.filter_channels_dp, params.n_heads,
                params.n_enc_layers, params.enc_kernel, params.enc_dropout, params.window_size,
                params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    m.load_state_dict(torch.load(ckpt, map_location='cuda'))
    return m.cuda().eval()


vanilla_ckpt = sorted(glob.glob(os.path.join(VANILLA_LOG_DIR, 'grad_*.pt')),
                      key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]))[-1]
student_ckpt = min(glob.glob(os.path.join(DISTILL_LOG_DIR, 'student_*.pt')),
                   key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]))
vanilla = _build(vanilla_ckpt)
student = _build(student_ckpt)

with open('checkpts/hifigan-config.json') as f:
    hps = AttrDict(json.load(f))
vocoder = HiFiGAN(hps)
vocoder.load_state_dict(torch.load(HIFIGAN_CKPT, map_location='cuda')['generator'])
vocoder.cuda().eval()
vocoder.remove_weight_norm()

# (label, model, solver, steps)
CONFIGS = [
    ('Vanilla-10', vanilla, 'original', 10),
    ('ML-4',       vanilla, 'ml',       4),
    ('Distilled-2', student, 'ddim',    2),
]
SENTENCES = [
    "Grad TTS turns noise into speech by reversing a diffusion process.",
    "Distillation lets the model generate speech in just two steps.",
]

cmu = cmudict.CMUDict('resources/cmu_dictionary')


def synth(m, text, solver, steps):
    x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu),
                                     len(symbols))).cuda()[None]
    x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
    t0 = dt.datetime.now()
    with torch.no_grad():
        _, y_dec, _ = m(x, x_lengths, n_timesteps=steps, temperature=1.5,
                       stoc=False, length_scale=0.91, solver=solver)
        audio = (vocoder(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
    rtf = (dt.datetime.now() - t0).total_seconds() * 22050 / (y_dec.shape[-1] * 256)
    return audio, rtf


for label, m, solver, steps in CONFIGS:
    rtfs = []
    print(f'\n=== {label} (solver={solver}, steps={steps}) ===')
    for i, text in enumerate(SENTENCES):
        audio, rtf = synth(m, text, solver, steps)
        rtfs.append(rtf)
        print(f'  sentence {i} | RTF {rtf:.3f}')
        display(Audio(audio, rate=22050))
    print(f'  mean RTF = {np.mean(rtfs):.3f}')
