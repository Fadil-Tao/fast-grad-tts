# %% [cell 6] Synthesize with the vanilla Euler solver (10 steps) + HiFi-GAN
# Loads the newest checkpoint from LOG_DIR, generates a mel with the original
# Grad-TTS solver, vocodes it with HiFi-GAN, prints the Real Time Factor (RTF)
# and plays the audio inline.
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
from scipy.io.wavfile import write
from IPython.display import Audio, display

N_TIMESTEPS = 10          # vanilla Grad-TTS uses 10 reverse-diffusion steps
SOLVER = 'original'       # Euler scheme
SENTENCES = [
    "Grad TTS turns noise into speech by reversing a diffusion process.",
]

nsymbols = len(symbols) + 1 if params.add_blank else len(symbols)
model = GradTTS(nsymbols, params.n_spks, params.spk_emb_dim, params.n_enc_channels,
                params.filter_channels, params.filter_channels_dp, params.n_heads,
                params.n_enc_layers, params.enc_kernel, params.enc_dropout, params.window_size,
                params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
ckpt = sorted(glob.glob(os.path.join(params.log_dir, 'grad_*.pt')),
              key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]))[-1]
model.load_state_dict(torch.load(ckpt, map_location='cuda'))
model.cuda().eval()
print('loaded', ckpt)

with open('checkpts/hifigan-config.json') as f:
    hps = AttrDict(json.load(f))
vocoder = HiFiGAN(hps)
vocoder.load_state_dict(torch.load(HIFIGAN_CKPT, map_location='cuda')['generator'])
vocoder.cuda().eval()
vocoder.remove_weight_norm()

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
    write(f'sample_{i}.wav', 22050, audio)
    display(Audio(audio, rate=22050))
