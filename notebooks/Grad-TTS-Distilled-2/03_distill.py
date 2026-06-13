# %% [cell 3] Progressive distillation: vanilla teacher -> 2-step student
# Each round halves the DDIM step count (64 -> 32 -> 16 -> 8 -> 4 -> 2). The
# student that takes ONE DDIM step is trained to match the teacher's TWO steps
# (Fast Grad-TTS Sec. 3.2, eq. 10-11). Students are checkpointed to
# DISTILL_LOG_DIR after every round so you can resume across Colab sessions.
#
# NOTE: full 5 rounds x 35 epochs is long on free Colab. Start with small
# ROUNDS / EPOCHS_PER_ROUND to smoke-test, then scale up.
import os
import glob
import torch
from torch.utils.data import DataLoader

import params
from model import GradTTS
from data import TextMelDataset, TextMelBatchCollate
from text.symbols import symbols
from distill import progressive_distill

# --- distillation hyperparameters ---
START_STEPS      = 64     # teacher's step count in round 1
ROUNDS           = 5      # 64 -> 2 in 5 halvings  (set to 1-2 for a quick test)
EPOCHS_PER_ROUND = 35     # paper value; reduce for a quick test
LR               = 1e-5
# ------------------------------------

params.n_spks = 1
nsymbols = len(symbols) + 1 if params.add_blank else len(symbols)

# Data loader (ground-truth mels used as distillation targets).
train_ds = TextMelDataset(params.train_filelist_path, params.cmudict_path, params.add_blank,
                          params.n_fft, params.n_feats, params.sample_rate, params.hop_length,
                          params.win_length, params.f_min, params.f_max)
loader = DataLoader(train_ds, batch_size=params.batch_size, collate_fn=TextMelBatchCollate(),
                    drop_last=True, num_workers=2, shuffle=True)

# Load the trained vanilla model as the initial teacher.
teacher = GradTTS(nsymbols, params.n_spks, params.spk_emb_dim, params.n_enc_channels,
                  params.filter_channels, params.filter_channels_dp, params.n_heads,
                  params.n_enc_layers, params.enc_kernel, params.enc_dropout, params.window_size,
                  params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
vanilla_ckpt = sorted(glob.glob(os.path.join(VANILLA_LOG_DIR, 'grad_*.pt')),
                      key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]))[-1]
teacher.load_state_dict(torch.load(vanilla_ckpt, map_location='cuda'))
teacher = teacher.cuda()
print('teacher =', vanilla_ckpt)

student = progressive_distill(teacher, loader, rounds=ROUNDS, start_steps=START_STEPS,
                              epochs_per_round=EPOCHS_PER_ROUND, lr=LR,
                              out_size=params.out_size, device='cuda', log_dir=DISTILL_LOG_DIR)

final_steps = START_STEPS // (2 ** ROUNDS)
print(f'distillation done — final student samples in {final_steps} DDIM steps')
print('students saved under', DISTILL_LOG_DIR)
