# Progressive distillation for Grad-TTS (Fast Grad-TTS, Sec. 3.2).
#
# Adapts Salimans & Ho "Progressive Distillation for Fast Sampling of Diffusion
# Models" (ICLR 2022) to the Mean-Reverting Variance-Preserving DPM used by
# Grad-TTS, with DDIM sampling defined by `Diffusion.ddim_step` (eq. 10) and the
# distillation loss of eq. (11).
#
# Each round halves the number of reverse-diffusion steps: a student that takes
# ONE DDIM step is trained to match the TWO DDIM steps of the teacher. After the
# round the student becomes the next teacher. Starting at `start_steps` and
# running `rounds` rounds reaches `start_steps / 2**rounds` steps (paper: 64 -> 2
# in 5 rounds). Only the diffusion decoder is trained; encoder + duration
# predictor stay frozen (loaded from the trained vanilla checkpoint).

import copy
import os

import torch
from tqdm import tqdm

from model.diffusion import get_noise


@torch.no_grad()
def _teacher_two_steps(decoder, xu, mask, mu, u, h_s, spk=None):
    """Two teacher DDIM steps over the student's single interval h_s:
    u -> u - h_s/2 -> u - h_s. Returns the teacher target (no grad)."""
    h_t = h_s / 2.0
    mid = u - h_t
    x = decoder.ddim_step(xu, mask, mu, u, mid, spk)
    x = decoder.ddim_step(x, mask, mu, mid, u - h_s, spk)
    return x


def _alpha(decoder, u):
    """alpha_u = 1 - gamma_{0,u}^2 = 1 - exp(-int_0^u beta) (loss weight, eq. 11)."""
    time_u = u.unsqueeze(-1).unsqueeze(-1)
    cum_u = get_noise(time_u, decoder.beta_min, decoder.beta_max, cumulative=True)
    return 1.0 - torch.exp(-cum_u)


def distill_round(teacher, loader, student_steps, epochs, lr, out_size,
                  device='cuda', grad_clip=1.0, log_every=50):
    """One distillation round. `teacher` samples in 2*student_steps DDIM steps;
    returns a `student` that samples in `student_steps`. Student is initialised
    from the teacher and only its decoder is optimised."""
    student = copy.deepcopy(teacher).to(device)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    # Freeze everything but the diffusion decoder of the student.
    for p in student.parameters():
        p.requires_grad_(False)
    for p in student.decoder.parameters():
        p.requires_grad_(True)

    optim = torch.optim.Adam(student.decoder.parameters(), lr=lr)
    h_s = 1.0 / student_steps
    n_feats = student.decoder.n_feats
    step = 0
    for epoch in range(1, epochs + 1):
        student.decoder.train()
        with tqdm(loader, desc=f'[{2*student_steps}->{student_steps}] epoch {epoch}') as bar:
            for batch in bar:
                x, x_lengths = batch['x'].to(device), batch['x_lengths'].to(device)
                y, y_lengths = batch['y'].to(device), batch['y_lengths'].to(device)

                # Aligned decoder targets (encoder + MAS), frozen -> no grad.
                x0, mu_y, y_mask = teacher.compute_alignment(
                    x, x_lengths, y, y_lengths, out_size=out_size)

                # Sample u uniformly on the student grid {h_s, 2h_s, .., 1}.
                B = x0.shape[0]
                idx = torch.randint(1, student_steps + 1, (B,), device=device)
                u = idx.to(x0.dtype) * h_s

                # Noisy x_u via forward diffusion (eq. 3); independent of weights.
                xu, _ = student.decoder.forward_diffusion(x0, y_mask, mu_y, u)
                xu = xu.detach()

                # Teacher target: 2 DDIM steps (no grad). Student: 1 step (grad).
                target = _teacher_two_steps(teacher.decoder, xu, y_mask, mu_y, u, h_s)
                pred = student.decoder.ddim_step(xu, y_mask, mu_y, u, u - h_s)

                alpha_u = _alpha(student.decoder, u)
                # Distillation loss, eq. (11): weighted by 1/alpha_u, masked mean.
                loss = torch.sum(((pred - target) ** 2) / alpha_u * y_mask) \
                    / (torch.sum(y_mask) * n_feats)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.decoder.parameters(), grad_clip)
                optim.step()

                if step % log_every == 0:
                    bar.set_postfix(loss=float(loss.item()))
                step += 1

    student.eval()
    for p in student.parameters():
        p.requires_grad_(True)
    return student


def progressive_distill(model, loader, rounds=5, start_steps=64,
                        epochs_per_round=35, lr=1e-5, out_size=None,
                        device='cuda', log_dir='.', grad_clip=1.0):
    """Run `rounds` halving rounds starting from a teacher that samples in
    `start_steps`. Saves the student after every round to
    `log_dir/student_<steps>.pt` (so Colab can resume) and returns the final
    few-step model, to be sampled with `solver='ddim'`, `n_timesteps=final`."""
    os.makedirs(log_dir, exist_ok=True)
    teacher = model.to(device).eval()
    teacher_steps = start_steps
    for k in range(1, rounds + 1):
        student_steps = teacher_steps // 2
        assert student_steps >= 1, "start_steps too small for this many rounds"
        print(f'== Distillation round {k}/{rounds}: '
              f'{teacher_steps} -> {student_steps} steps ==')
        student = distill_round(teacher, loader, student_steps,
                                epochs_per_round, lr, out_size,
                                device=device, grad_clip=grad_clip)
        ckpt_path = os.path.join(log_dir, f'student_{student_steps}.pt')
        torch.save(student.state_dict(), ckpt_path)
        print(f'   saved {ckpt_path}')
        teacher = student
        teacher_steps = student_steps
    return teacher
