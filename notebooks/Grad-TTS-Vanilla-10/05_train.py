# %% [cell 5] Train vanilla Grad-TTS (checkpoints saved to Drive LOG_DIR)
# Resumes automatically from the newest grad_<epoch>.pt in LOG_DIR, so you can
# rerun this cell across Colab sessions to keep training. EPOCHS_THIS_RUN bounds
# how long a single run goes before the cell returns.
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader

import params
from model import GradTTS
from data import TextMelDataset, TextMelBatchCollate
from text.symbols import symbols

EPOCHS_THIS_RUN = 100   # epochs to run now; the paper trains for much longer

torch.manual_seed(params.seed)
np.random.seed(params.seed)
nsymbols = len(symbols) + 1 if params.add_blank else len(symbols)

train_ds = TextMelDataset(params.train_filelist_path, params.cmudict_path, params.add_blank,
                          params.n_fft, params.n_feats, params.sample_rate, params.hop_length,
                          params.win_length, params.f_min, params.f_max)
loader = DataLoader(train_ds, batch_size=params.batch_size, collate_fn=TextMelBatchCollate(),
                    drop_last=True, num_workers=2, shuffle=True)

model = GradTTS(nsymbols, params.n_spks, params.spk_emb_dim, params.n_enc_channels,
                params.filter_channels, params.filter_channels_dp, params.n_heads,
                params.n_enc_layers, params.enc_kernel, params.enc_dropout, params.window_size,
                params.n_feats, params.dec_dim, params.beta_min, params.beta_max,
                params.pe_scale).cuda()


def _epoch_of(path):
    return int(os.path.basename(path).split('_')[-1].split('.')[0])


start_epoch = 1
ckpts = sorted(glob.glob(os.path.join(params.log_dir, 'grad_*.pt')), key=_epoch_of)
if ckpts:
    model.load_state_dict(torch.load(ckpts[-1], map_location='cuda'))
    start_epoch = _epoch_of(ckpts[-1]) + 1
    print('resumed from', ckpts[-1])
print('total parameters: %.2fm' % (model.nparams / 1e6))

optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

for epoch in range(start_epoch, start_epoch + EPOCHS_THIS_RUN):
    model.train()
    dur_l, prior_l, diff_l = [], [], []
    for batch in loader:
        model.zero_grad()
        x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
        y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
        dur, prior, diff = model.compute_loss(x, x_lengths, y, y_lengths, out_size=params.out_size)
        (dur + prior + diff).backward()
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1)
        optimizer.step()
        dur_l.append(dur.item()); prior_l.append(prior.item()); diff_l.append(diff.item())
    print(f'epoch {epoch} | dur {np.mean(dur_l):.3f} '
          f'prior {np.mean(prior_l):.3f} diff {np.mean(diff_l):.3f}')
    if epoch % params.save_every == 0:
        torch.save(model.state_dict(), os.path.join(params.log_dir, f'grad_{epoch}.pt'))

print('done. latest checkpoint in', params.log_dir)
