# %% [cell 3] Load the trained vanilla Grad-TTS + HiFi-GAN vocoder
import os
import sys
import glob
import json
import torch

sys.path.append('hifi-gan')
import params
from model import GradTTS
from text.symbols import symbols
from env import AttrDict
from models import Generator as HiFiGAN

params.n_spks = 1
nsymbols = len(symbols) + 1 if params.add_blank else len(symbols)

model = GradTTS(nsymbols, params.n_spks, params.spk_emb_dim, params.n_enc_channels,
                params.filter_channels, params.filter_channels_dp, params.n_heads,
                params.n_enc_layers, params.enc_kernel, params.enc_dropout, params.window_size,
                params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
ckpt = sorted(glob.glob(os.path.join(VANILLA_LOG_DIR, 'grad_*.pt')),
              key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]))[-1]
model.load_state_dict(torch.load(ckpt, map_location='cuda'))
model.cuda().eval()
print('loaded vanilla checkpoint:', ckpt)

with open('checkpts/hifigan-config.json') as f:
    hps = AttrDict(json.load(f))
vocoder = HiFiGAN(hps)
vocoder.load_state_dict(torch.load(HIFIGAN_CKPT, map_location='cuda')['generator'])
vocoder.cuda().eval()
vocoder.remove_weight_norm()
print('vocoder ready')
