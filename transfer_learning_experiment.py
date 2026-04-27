# -*- coding: utf-8 -*-
"""Transfer learning for UAV low-altitude turbulence.
Keeps the current model and separate val/cal/test splits.
Uses a harder rollout task u_t -> u_{t+8} with stronger dynamics.
Adds few-shot sweep, progressive unfreezing, L2-SP, interval width,
and task shortcut diagnostics.
"""

import copy
import json
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from Attention_Residuals_implementation import (
    AttentionResidualTurbulenceModel,
    PhysicsInformedLoss,
    make_optimizer,
)
from stage3_validation import validate_attention_residuals


def auto_calibrate_scale_wide(model, loader, device="cpu", target=0.93):
    cand = np.linspace(0.15, 1.20, 43)
    best_s, best_gap, best_m = float(cand[0]), 1e9, None
    for s in cand:
        m = validate_attention_residuals(model, loader, device=device, calib_scale=float(s))
        gap = abs(m["uncertainty_coverage"] - target)
        if gap < best_gap:
            best_s, best_gap, best_m = float(s), gap, m
    return best_s, best_m


class DomainShiftTurbulenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        n_samples=160,
        grid_size=16,
        domain="source",
        re_tau=1000,
        rollout_steps=8,
    ):
        self.n_samples, self.grid_size = n_samples, grid_size
        self.nu, self.domain = 1.0 / re_tau, domain
        self.rollout_steps = rollout_steps
        self._generate_data()

    def _sample_slope(self):
        if self.domain == "source_wider_plus":
            return np.random.uniform(1.55, 1.95)
        if self.domain == "source_wider":
            return np.random.uniform(1.40, 2.00)
        if self.domain == "source":
            return np.random.uniform(1.45, 1.95)
        if self.domain == "source_narrow":
            return np.random.uniform(1.50, 1.75)
        return np.random.uniform(1.65, 1.90)

    def _generate_velocity_field(self):
        N = self.grid_size
        u_hat = np.random.randn(3, N, N, N) + 1j * np.random.randn(3, N, N, N)
        k = np.fft.fftfreq(N, 1.0 / N)
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
        E_k = np.zeros_like(k_mag)
        mask = k_mag > 0.1
        E_k[mask] = k_mag[mask] ** (-self._sample_slope())
        u_hat *= np.sqrt(E_k + 1e-8)

        if self.domain in {"source", "source_wider", "source_wider_plus"}:
            if self.domain == "source_wider_plus":
                aniso_prob = 0.9
                aniso_u_lo, aniso_u_hi = 1.10, 1.35
                aniso_v_lo, aniso_v_hi = 0.72, 0.92
            elif self.domain == "source_wider":
                aniso_prob = 0.8
                aniso_u_lo, aniso_u_hi = 1.05, 1.35
                aniso_v_lo, aniso_v_hi = 0.75, 1.0
            else:
                aniso_prob = 0.5
                aniso_u_lo, aniso_u_hi = 1.0, 1.25
                aniso_v_lo, aniso_v_hi = 0.80, 1.0
            if np.random.rand() < aniso_prob:
                aniso_u = np.random.uniform(aniso_u_lo, aniso_u_hi)
                aniso_v = np.random.uniform(aniso_v_lo, aniso_v_hi)
                u_hat[0] *= aniso_u
                u_hat[1] *= aniso_v

        if self.domain == "target":
            u_hat[0] *= 1.25
            u_hat[1] *= 0.80

        k_dot_u = kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]
        k2 = k_mag**2 + 1e-8
        u_hat[0] -= (kx * k_dot_u) / k2
        u_hat[1] -= (ky * k_dot_u) / k2
        u_hat[2] -= (kz * k_dot_u) / k2
        u = np.real(np.fft.ifftn(u_hat, axes=(1, 2, 3)))

        if self.domain in {"source", "source_wider", "source_wider_plus"}:
            if self.domain == "source_wider_plus":
                shear_prob = 0.9
                shear_lo, shear_hi = 0.18, 0.30
                z_scale_lo, z_scale_hi = 0.78, 0.92
            elif self.domain == "source_wider":
                shear_prob = 0.7
                shear_lo, shear_hi = 0.10, 0.25
                z_scale_lo, z_scale_hi = 0.80, 0.98
            else:
                shear_prob = 0.4
                shear_lo, shear_hi = 0.05, 0.20
                z_scale_lo, z_scale_hi = 0.85, 1.0
            if np.random.rand() < shear_prob:
                shear_strength = np.random.uniform(shear_lo, shear_hi)
                shear = np.linspace(-0.8, 0.8, N)[None, :, None, None]
                u[0] += shear_strength * shear[0]
                z_scale = np.random.uniform(z_scale_lo, z_scale_hi)
                u[2] *= z_scale

        if self.domain == "target":
            shear = np.linspace(-0.8, 0.8, N)[None, :, None, None]
            u[0] += 0.25 * shear[0]
            u[2] *= 0.85
        return u

    def _advect_diffuse(self, u):
        u_hat = np.fft.fftn(u, axes=(1, 2, 3))
        N = self.grid_size
        k = np.fft.fftfreq(N, 1.0 / N) * 2.0 * np.pi
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        k2 = kx**2 + ky**2 + kz**2

        if self.domain in {"source", "source_wider", "source_wider_plus"}:
            if self.domain == "source_wider_plus":
                nu = self.nu * np.random.uniform(1.10, 1.30)
            elif self.domain == "source_wider":
                nu = self.nu * np.random.uniform(1.05, 1.25)
            else:
                nu = self.nu * np.random.uniform(1.0, 1.20)
        elif self.domain == "target":
            nu = self.nu * 1.15
        else:
            nu = self.nu

        decay = np.exp(-nu * k2 * 0.02)
        u_hat *= decay
        u_next = np.real(np.fft.ifftn(u_hat, axes=(1, 2, 3)))

        grad_x0 = np.gradient(u[0], axis=0)
        grad_y1 = np.gradient(u[1], axis=1)
        grad_z2 = np.gradient(u[2], axis=2)
        cross01 = u[0] * u[1]
        cross12 = u[1] * u[2]

        u_next[0] -= 0.10 * u[0] * grad_x0
        u_next[1] -= 0.08 * u[1] * grad_y1
        u_next[2] -= 0.06 * u[2] * grad_z2

        u_next[0] += 0.07 * np.roll(u[1], shift=1, axis=1)
        u_next[1] += 0.06 * np.roll(u[0], shift=1, axis=2)
        u_next[2] += 0.05 * np.roll(u[0], shift=1, axis=0)

        u_next[0] += 0.04 * cross01
        u_next[1] += 0.03 * cross12
        u_next[2] += 0.03 * np.tanh(u[0] * u[2])
        u_next -= 0.015 * u * np.abs(u)

        if self.domain == "target":
            z = np.linspace(0.0, 1.0, N)[None, None, None, :]
            shear_drive = np.exp(-((np.arange(N) - (N // 4)) ** 2) / (2.0 * 3.0**2))
            shear_drive = shear_drive[:, None, None]
            u_next[0] += 0.10 * z[0]
            u_next[0] += 0.03 * shear_drive * np.roll(u[1], shift=1, axis=0)
            u_next[1] += 0.02 * np.roll(u[2], shift=1, axis=1)
        elif self.domain in {"source", "source_wider", "source_wider_plus"}:
            z = np.linspace(0.0, 1.0, N)[None, None, None, :]
            if self.domain == "source_wider_plus":
                drive_prob = 0.9
                drive_lo, drive_hi = 0.07, 0.11
                extra_roll_prob = 0.95
                shear_drive = np.exp(-((np.arange(N) - (N // 4)) ** 2) / (2.0 * 3.0**2))
                shear_drive = shear_drive[:, None, None]
                u_next[0] += 0.03 * shear_drive * np.roll(u[1], shift=1, axis=0)
                u_next[1] += 0.02 * np.roll(u[2], shift=1, axis=1)
            elif self.domain == "source_wider":
                drive_prob = 0.6
                drive_lo, drive_hi = 0.04, 0.10
                extra_roll_prob = 0.8
                u_next[0] += 0.02 * np.roll(u[1], shift=1, axis=0)
            else:
                drive_prob = 0.3
                drive_lo, drive_hi = 0.02, 0.07
                extra_roll_prob = 0.5
            if np.random.rand() < drive_prob:
                drive_strength = np.random.uniform(drive_lo, drive_hi)
                u_next[0] += drive_strength * z[0]
                if np.random.rand() < extra_roll_prob:
                    u_next[1] += 0.01 * np.roll(u[2], shift=1, axis=1)
                if self.domain == "source_wider" and np.random.rand() < 0.5:
                    u_next[0] += 0.02 * np.roll(u[1], shift=1, axis=0)
            else:
                u_next[0] += 0.02 * np.roll(u[2], shift=1, axis=2)
        else:
            u_next[0] += 0.02 * np.roll(u[2], shift=1, axis=2)

        return u_next.astype(np.float32)

    def _generate_data(self):
        self.data = []
        for _ in range(self.n_samples):
            u_t = self._generate_velocity_field()
            u_cur = u_t.copy()
            for _ in range(self.rollout_steps):
                u_cur = self._advect_diffuse(u_cur)
            self.data.append((u_t.reshape(3, -1).T, u_cur.reshape(3, -1).T))

    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        u_t, u_t1 = self.data[idx]
        return torch.from_numpy(u_t).float(), torch.from_numpy(u_t1).float()


class MultiKTurbulenceDataset(torch.utils.data.Dataset):
    """Source dataset that mixes multiple rollout horizons."""

    def __init__(
        self,
        n_samples=160,
        grid_size=16,
        re_tau=1000,
        rollout_steps_list=(1, 2, 4, 8),
        domain="source",
    ):
        self.datasets = []
        self.cumulative_lengths = []
        if len(rollout_steps_list) == 0:
            return

        base = n_samples // len(rollout_steps_list)
        remainder = n_samples % len(rollout_steps_list)
        total = 0
        for i, rollout_steps in enumerate(rollout_steps_list):
            per_k = base + (1 if i < remainder else 0)
            if per_k <= 0:
                continue
            ds = DomainShiftTurbulenceDataset(
                n_samples=per_k,
                grid_size=grid_size,
                domain=domain,
                re_tau=re_tau,
                rollout_steps=rollout_steps,
            )
            self.datasets.append(ds)
            total += len(ds)
            self.cumulative_lengths.append(total)

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                local_idx = idx if i == 0 else idx - self.cumulative_lengths[i - 1]
                return self.datasets[i][local_idx]
        raise IndexError(f"Index {idx} out of range")


class L2SPRegularizer:
    def __init__(self, source_state, model, alpha=1e-3, beta=1e-4):
        self.src = {
            k: v.detach().clone()
            for k, v in source_state.items()
            if k in model.state_dict() and not k.startswith("head_")
        }
        self.alpha, self.beta = alpha, beta

    def __call__(self, model):
        reg = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.src:
                reg += self.alpha * (p - self.src[name].to(p.device)).pow(2).sum()
            elif name.startswith("head_"):
                reg += self.beta * p.pow(2).sum()
        return reg


def create_model(model_size="small", **overrides):
    presets = {
        "tiny": {"dim": 16, "n_layers": 2, "n_heads": 2, "sparse_k": 2},
        "small": {"dim": 32, "n_layers": 3, "n_heads": 4, "sparse_k": 2},
        "base": {"dim": 64, "n_layers": 6, "n_heads": 4, "sparse_k": 2},
        "large": {"dim": 128, "n_layers": 8, "n_heads": 8, "sparse_k": 2},
    }
    if model_size not in presets:
        raise ValueError(f"Unknown model_size={model_size}. Choose from {sorted(presets.keys())}.")

    cfg = {
        "in_ch": 3,
        "out_ch": 3,
        "residual_output": True,
        **presets[model_size],
        **overrides,
    }

    return AttentionResidualTurbulenceModel(
        in_ch=cfg["in_ch"],
        dim=cfg["dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        out_ch=cfg["out_ch"],
        sparse_k=cfg["sparse_k"],
        residual_output=cfg["residual_output"],
    )


def reset_heads(m):
    m.head_mu.reset_parameters()
    m.head_lv.reset_parameters()


def set_trainable(m, mode):
    for name, p in m.named_parameters():
        if mode == "heads":
            p.requires_grad = name.startswith("head_")
        elif mode == "last":
            p.requires_grad = (
                name.startswith("head_")
                or name.startswith("blocks.2")
                or name.startswith("norm")
            )
        else:
            p.requires_grad = True


def split_target_dataset(ds, shots=(4, 8, 12, 24), val_n=18, cal_n=12, test_n=60, split_seed=42, min_pool=None):
    idx = np.random.RandomState(split_seed).permutation(len(ds))
    test_idx = idx[-test_n:]
    cal_idx = idx[-(test_n + cal_n):-test_n]
    val_idx = idx[-(test_n + cal_n + val_n):-(test_n + cal_n)]
    pool = idx[:-(test_n + cal_n + val_n)]
    if min_pool is not None and len(pool) < int(min_pool):
        raise ValueError(
            f"Training pool too small: pool={len(pool)} < min_pool={min_pool}. "
            f"Increase dataset size or reduce val/cal/test sizes."
        )
    splits = {str(n): torch.utils.data.Subset(ds, pool[:n]) for n in shots if n <= len(pool)}
    return (
        splits,
        torch.utils.data.Subset(ds, val_idx),
        torch.utils.data.Subset(ds, cal_idx),
        torch.utils.data.Subset(ds, test_idx),
    )


def make_loaders(train_ds, val_ds, cal_ds, test_ds, bs=4):
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=min(len(train_ds), bs), shuffle=True),
        torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False),
        torch.utils.data.DataLoader(cal_ds, batch_size=bs, shuffle=False),
        torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False),
    )


def compute_loader_rel_error(model, loader, device="cpu"):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mu, _ = model(xb)
            preds.append(mu)
            targets.append(yb)
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    return float(torch.norm(pred - target) / (torch.norm(target) + 1e-8))


def compute_state_drift(model, reference_state, include_heads=True):
    model_state = model.state_dict()
    sq_sum = 0.0
    ref_sq_sum = 0.0
    for name, ref_tensor in reference_state.items():
        if name not in model_state:
            continue
        if not include_heads and name.startswith("head_"):
            continue
        cur_tensor = model_state[name].detach().float().cpu()
        ref_tensor = ref_tensor.detach().float().cpu()
        sq_sum += float((cur_tensor - ref_tensor).pow(2).sum().item())
        ref_sq_sum += float(ref_tensor.pow(2).sum().item())
    return float(np.sqrt(sq_sum) / (np.sqrt(ref_sq_sum) + 1e-12))


def fit_model_mse(model, train_dl, val_dl, device="cpu", epochs=120, lr=3e-4, loggers=None):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best_v, best_state = float("inf"), None
    loggers = loggers or {}
    rel_error_loader = loggers.get("rel_error_loader")
    reference_state = loggers.get("reference_state")
    every = int(loggers.get("every", max(1, epochs // 6)))
    history = loggers.get("history")
    for ep in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            mu, _ = model(xb)
            loss = (mu - yb).pow(2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())
        model.eval()
        vals = []
        with torch.no_grad():
            for xv, yv in val_dl:
                xv, yv = xv.to(device), yv.to(device)
                mu, _ = model(xv)
                vals.append((mu - yv).pow(2).mean().item())
        sched.step()
        v = float(np.mean(vals))
        if history is not None and (ep == 0 or ep == epochs - 1 or (ep + 1) % every == 0):
            rec = {
                "epoch": int(ep + 1),
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_loss": v,
            }
            if rel_error_loader is not None:
                rec["val_rel_error"] = compute_loader_rel_error(model, rel_error_loader, device=device)
            if reference_state is not None:
                rec["drift_from_source_all"] = compute_state_drift(model, reference_state, include_heads=True)
                rec["drift_from_source_backbone"] = compute_state_drift(model, reference_state, include_heads=False)
            history.append(rec)
        if v < best_v:
            best_v = v
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def fit_model_mse_warmstart(
    model,
    train_dl,
    val_dl,
    device="cpu",
    warmup_epochs=20,
    warmup_lr=1e-4,
    main_epochs=70,
    muon_lr=0.008,
    adamw_lr=8e-4,
    l2sp=None,
):
    model = fit_model_mse(
        model,
        train_dl,
        val_dl,
        device=device,
        epochs=warmup_epochs,
        lr=warmup_lr,
    )
    model = fit_model(
        model,
        train_dl,
        val_dl,
        device=device,
        epochs=main_epochs,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        l2sp=l2sp,
    )
    return model


def fit_model(model, train_dl, val_dl, device="cpu", epochs=100, muon_lr=0.008, adamw_lr=8e-4, l2sp=None, loggers=None, max_phy_w=0.03, phy_warmup=30):
    model = model.to(device)
    loss_fn = PhysicsInformedLoss(max_phy_w=max_phy_w, warmup=phy_warmup)
    muon, adamw = make_optimizer(model, muon_lr=muon_lr, adamw_lr=adamw_lr)
    sm = CosineAnnealingLR(muon, T_max=epochs, eta_min=1e-4)
    sa = CosineAnnealingLR(adamw, T_max=epochs, eta_min=1e-5)
    best_v, best_state = float("inf"), None
    loggers = loggers or {}
    rel_error_loader = loggers.get("rel_error_loader")
    reference_state = loggers.get("reference_state")
    every = int(loggers.get("every", max(1, epochs // 6)))
    history = loggers.get("history")
    for ep in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            muon.zero_grad()
            adamw.zero_grad()
            loss = loss_fn(*model(xb), yb, ep)["total"]
            if l2sp is not None:
                loss = loss + l2sp(model)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            muon.step(); adamw.step()
            train_losses.append(loss.item())
        model.eval(); vals = []
        with torch.no_grad():
            for xv, yv in val_dl:
                xv, yv = xv.to(device), yv.to(device)
                vals.append(loss_fn(*model(xv), yv, ep)["total"].item())
        sm.step()
        sa.step()
        v = float(np.mean(vals))
        if history is not None and (ep == 0 or ep == epochs - 1 or (ep + 1) % every == 0):
            rec = {
                "epoch": int(ep + 1),
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_loss": v,
            }
            if rel_error_loader is not None:
                rec["val_rel_error"] = compute_loader_rel_error(model, rel_error_loader, device=device)
            if reference_state is not None:
                rec["drift_from_source_all"] = compute_state_drift(model, reference_state, include_heads=True)
                rec["drift_from_source_backbone"] = compute_state_drift(model, reference_state, include_heads=False)
            history.append(rec)
        if v < best_v:
            best_v = v
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def progressive_unfreeze(source_state, train_dl, val_dl, device="cpu"):
    m = create_model()
    m.load_state_dict(copy.deepcopy(source_state))
    reset_heads(m)
    reg = L2SPRegularizer(source_state, m)
    set_trainable(m, "heads")
    m = fit_model(
        m, train_dl, val_dl, device=device,
        epochs=25, muon_lr=0.004, adamw_lr=4e-4, l2sp=reg,
    )
    set_trainable(m, "last")
    m = fit_model(
        m, train_dl, val_dl, device=device,
        epochs=25, muon_lr=0.002, adamw_lr=2e-4, l2sp=reg,
    )
    set_trainable(m, "all")
    return fit_model(
        m, train_dl, val_dl, device=device,
        epochs=40, muon_lr=0.001, adamw_lr=1e-4, l2sp=reg,
    )


def compute_gaussian_nll(model, loader, device="cpu"):
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mu, lv = model(xb)
            lv = lv.clamp(-10.0, 10.0)
            nll = 0.5 * (lv + (yb - mu).pow(2) * torch.exp(-lv))
            losses.append(float(nll.mean().item()))
    return float(np.mean(losses)) if losses else float("nan")


def compute_mean_sigma(model, loader, device="cpu"):
    model.eval()
    sigmas = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            _, lv = model(xb)
            sigma = torch.exp(0.5 * lv.clamp(-10.0, 10.0))
            sigmas.append(float(sigma.mean().item()))
    return float(np.mean(sigmas)) if sigmas else float("nan")


def compute_loader_mse(model, loader, device="cpu"):
    model.eval()
    mses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mu, _ = model(xb)
            mses.append(float((mu - yb).pow(2).mean().item()))
    return float(np.mean(mses)) if mses else float("nan")


def compute_loader_crps_proxy(model, loader, device="cpu"):
    model.eval()
    scores = []
    const = 1.0 / np.sqrt(np.pi)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mu, lv = model(xb)
            sigma = torch.exp(0.5 * lv.clamp(-10.0, 10.0)).clamp(min=1e-6)
            z = ((yb - mu).abs() / sigma).cpu().numpy()
            sigma_np = sigma.cpu().numpy()
            phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
            Phi = 0.5 * (1.0 + np.vectorize(np.math.erf)(z / np.sqrt(2.0)))
            crps = sigma_np * (z * (2.0 * Phi - 1.0) + 2.0 * phi - const)
            scores.append(float(np.mean(crps)))
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_model(model, cal_dl, test_dl, device="cpu"):
    scale, cal_m = auto_calibrate_scale_wide(model, cal_dl, device=device, target=0.93)
    test_m = validate_attention_residuals(model, test_dl, device=device, calib_scale=scale)
    widths = []
    model.eval(); model.to(device)
    with torch.no_grad():
        for xb, _ in test_dl:
            xb = xb.to(device)
            _, lv = model(xb)
            sigma = (torch.exp(0.5 * lv).clamp(max=10.0) * scale).cpu().numpy()
            widths.append(float(np.mean(2.0 * 1.96 * sigma)))
    return {
        "scale": float(scale),
        "coverage": float(test_m["uncertainty_coverage"]),
        "calibration_coverage": float(cal_m["uncertainty_coverage"]),
        "rel_error": float(test_m["rel_error"]),
        "interval_width": float(np.mean(widths)),
        "test_mse": compute_loader_mse(model, test_dl, device=device),
        "test_nll": compute_gaussian_nll(model, test_dl, device=device),
        "test_mean_sigma": compute_mean_sigma(model, test_dl, device=device),
        "test_crps_proxy": compute_loader_crps_proxy(model, test_dl, device=device),
    }


def evaluate_identity_baseline(cal_dl, test_dl):
    cal_x, cal_y, test_x, test_y = [], [], [], []
    for xb, yb in cal_dl:
        cal_x.append(xb)
        cal_y.append(yb)
    for xb, yb in test_dl:
        test_x.append(xb)
        test_y.append(yb)

    cal_x = torch.cat(cal_x)
    cal_y = torch.cat(cal_y)
    test_x = torch.cat(test_x)
    test_y = torch.cat(test_y)

    cal_abs = (cal_y - cal_x).abs().reshape(-1).numpy()
    q = min(np.ceil(0.93 * (len(cal_abs) + 1)) / len(cal_abs), 1.0)
    scale = float(np.quantile(cal_abs, q))
    test_abs = (test_y - test_x).abs()
    rel_error = float(torch.norm(test_y - test_x) / torch.norm(test_y))
    coverage = float((test_abs.reshape(-1).numpy() <= scale).mean())
    interval_width = float(2.0 * scale)
    return {
        "scale": scale,
        "coverage": coverage,
        "calibration_coverage": float((cal_abs <= scale).mean()),
        "rel_error": rel_error,
        "interval_width": interval_width,
    }


def compute_task_diagnostics(loader):
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
    x = torch.cat(xs)
    y = torch.cat(ys)
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)
    cos_sim = torch.nn.functional.cosine_similarity(x_flat, y_flat, dim=1).mean()
    residual = y - x
    return {
        "identity_baseline_rel_error": float(torch.norm(residual) / torch.norm(y)),
        "input_output_cosine_similarity": float(cos_sim),
        "residual_complexity_ratio": float(residual.var() / (y.var() + 1e-8)),
    }


def summarize_metric(per_seed_results, shots, strategies, metric_name):
    summary = {}
    for strat in strategies:
        summary[strat] = {}
        for n in shots:
            values = [seed_res["experiments"][str(n)][strat][metric_name] for seed_res in per_seed_results]
            summary[strat][str(n)] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(v) for v in values],
            }
    return summary


def summarize_scalar(per_seed_results, field_name):
    values = [seed_res[field_name] for seed_res in per_seed_results]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "values": [float(v) for v in values],
    }


def summarize_nested_scalar(per_seed_results, group_name, field_name):
    values = [seed_res[group_name][field_name] for seed_res in per_seed_results]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "values": [float(v) for v in values],
    }


def summarize_gap_trend(per_seed_results, shots, transfer_key="progressive_unfreeze_l2sp"):
    per_seed = []
    for seed_res in per_seed_results:
        gaps = []
        monotonic = True
        prev_gap = None
        for n in shots:
            scratch = seed_res["experiments"][str(n)]["scratch"]["rel_error"]
            transfer = seed_res["experiments"][str(n)][transfer_key]["rel_error"]
            gap = float(transfer - scratch)
            gaps.append(gap)
            if prev_gap is not None and gap > prev_gap + 1e-8:
                monotonic = False
            prev_gap = gap
        slope = float(np.polyfit(shots, gaps, 1)[0])
        per_seed.append({
            "seed": seed_res["seed"],
            "gaps": {str(n): float(g) for n, g in zip(shots, gaps)},
            "is_monotonic_nonincreasing": monotonic,
            "linear_slope": slope,
        })

    monotonic_count = sum(item["is_monotonic_nonincreasing"] for item in per_seed)
    slopes = [item["linear_slope"] for item in per_seed]
    mean_gaps = {}
    for n in shots:
        gap_values = [item["gaps"][str(n)] for item in per_seed]
        mean_gaps[str(n)] = {
            "mean": float(np.mean(gap_values)),
            "std": float(np.std(gap_values)),
            "values": [float(v) for v in gap_values],
        }
    return {
        "transfer_key": transfer_key,
        "per_seed": per_seed,
        "monotonic_nonincreasing_count": monotonic_count,
        "total_seeds": len(per_seed),
        "mean_gap_by_shot": mean_gaps,
        "linear_slope": {
            "mean": float(np.mean(slopes)),
            "std": float(np.std(slopes)),
            "values": [float(v) for v in slopes],
        },
    }


def run_single_seed(seed, shots, device, use_multi_k_source=False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_multi_k_source:
        source_ds = MultiKTurbulenceDataset(
            n_samples=160,
            grid_size=16,
            re_tau=1000,
            rollout_steps_list=(1, 2, 4, 8),
            domain="source",
        )
    else:
        source_ds = DomainShiftTurbulenceDataset(
            n_samples=160,
            grid_size=16,
            domain="source",
            rollout_steps=8,
        )
    src_train, src_val = torch.utils.data.random_split(
        source_ds,
        [128, 32],
        generator=torch.Generator().manual_seed(seed),
    )
    src_train_dl = torch.utils.data.DataLoader(src_train, batch_size=4, shuffle=True)
    src_val_dl = torch.utils.data.DataLoader(src_val, batch_size=4, shuffle=False)

    target_ds = DomainShiftTurbulenceDataset(
        n_samples=120,
        grid_size=16,
        domain="target",
        rollout_steps=8,
    )
    splits, val_ds, cal_ds, test_ds = split_target_dataset(
        target_ds, shots=shots, split_seed=seed
    )

    src_model = fit_model_mse(
        create_model(), src_train_dl, src_val_dl, device=device, epochs=120, lr=3e-4
    )
    src_state = copy.deepcopy(src_model.state_dict())

    src_cal_dl, src_test_dl = make_loaders(
        torch.utils.data.Subset(source_ds, range(4)), src_val, src_val, src_val
    )[2:]
    source_metrics = evaluate_model(src_model, src_cal_dl, src_test_dl, device=device)
    source_on_source_rel_error = float(source_metrics["rel_error"])
    source_identity = evaluate_identity_baseline(src_cal_dl, src_test_dl)

    _, _, target_cal_dl, target_test_dl = make_loaders(
        torch.utils.data.Subset(target_ds, range(4)), val_ds, cal_ds, test_ds
    )
    task_diagnostics = compute_task_diagnostics(target_test_dl)
    identity_baseline = evaluate_identity_baseline(target_cal_dl, target_test_dl)
    source_on_target_zero_shot = evaluate_model(src_model, target_cal_dl, target_test_dl, device=device)

    experiments = {}
    for n in shots:
        train_dl, val_dl, cal_dl, test_dl = make_loaders(splits[str(n)], val_ds, cal_ds, test_ds)
        exp = {}
        exp["scratch"] = evaluate_model(
            fit_model(
                create_model(), train_dl, val_dl, device=device,
                epochs=120, muon_lr=0.01, adamw_lr=1e-3,
            ),
            cal_dl,
            test_dl,
            device=device,
        )

        naive = create_model()
        naive.load_state_dict(copy.deepcopy(src_state))
        reset_heads(naive)
        exp["naive_finetune"] = evaluate_model(
            fit_model(
                naive, train_dl, val_dl, device=device,
                epochs=90, muon_lr=0.002, adamw_lr=2e-4,
            ),
            cal_dl,
            test_dl,
            device=device,
        )

        warmstart = create_model()
        warmstart.load_state_dict(copy.deepcopy(src_state))
        reset_heads(warmstart)
        exp["warmstart_transfer"] = evaluate_model(
            fit_model_mse_warmstart(
                warmstart, train_dl, val_dl, device=device,
                warmup_epochs=20, warmup_lr=1e-4,
                main_epochs=70, muon_lr=0.002, adamw_lr=2e-4,
            ),
            cal_dl,
            test_dl,
            device=device,
        )

        weak_ft = create_model()
        weak_ft.load_state_dict(copy.deepcopy(src_state))
        reset_heads(weak_ft)
        exp["weak_finetune"] = evaluate_model(
            fit_model(
                weak_ft, train_dl, val_dl, device=device,
                epochs=20, muon_lr=0.001, adamw_lr=1e-4,
            ),
            cal_dl,
            test_dl,
            device=device,
        )

        frozen = create_model()
        frozen.load_state_dict(copy.deepcopy(src_state))
        reset_heads(frozen)
        set_trainable(frozen, "heads")
        exp["frozen_backbone"] = evaluate_model(
            fit_model(
                frozen, train_dl, val_dl, device=device,
                epochs=90, muon_lr=0.004, adamw_lr=4e-4,
            ),
            cal_dl,
            test_dl,
            device=device,
        )

        strong_l2sp_model = create_model()
        strong_l2sp_model.load_state_dict(copy.deepcopy(src_state))
        reset_heads(strong_l2sp_model)
        strong_reg = L2SPRegularizer(src_state, strong_l2sp_model, alpha=1e-2, beta=1e-3)
        exp["strong_l2sp"] = evaluate_model(
            fit_model(
                strong_l2sp_model, train_dl, val_dl, device=device,
                epochs=90, muon_lr=0.002, adamw_lr=2e-4,
                l2sp=strong_reg,
            ),
            cal_dl,
            test_dl,
            device=device,
        )

        long_frozen = create_model()
        long_frozen.load_state_dict(copy.deepcopy(src_state))
        reset_heads(long_frozen)
        set_trainable(long_frozen, "heads")
        exp["frozen_long"] = evaluate_model(
            fit_model(
                long_frozen, train_dl, val_dl, device=device,
                epochs=120, muon_lr=0.004, adamw_lr=4e-4,
            ),
            cal_dl,
            test_dl,
            device=device,
        )

        prog = progressive_unfreeze(src_state, train_dl, val_dl, device=device)
        exp["progressive_unfreeze_l2sp"] = evaluate_model(prog, cal_dl, test_dl, device=device)
        experiments[str(n)] = exp

    return {
        "seed": seed,
        "source_pretraining": source_metrics,
        "source_identity_baseline": source_identity,
        "source_on_source_rel_error": source_on_source_rel_error,
        "source_on_target_zero_shot": source_on_target_zero_shot,
        "task_diagnostics": task_diagnostics,
        "identity_baseline": identity_baseline,
        "experiments": experiments,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shots = [4, 8, 12, 24]
    seeds = [42, 52, 62]
    use_multi_k_source = False
    strategies = [
        "scratch",
        "naive_finetune",
        "warmstart_transfer",
        "weak_finetune",
        "frozen_backbone",
        "strong_l2sp",
        "frozen_long",
        "progressive_unfreeze_l2sp",
    ]

    print("=" * 72)
    print("Attention Residuals - Transfer Learning for UAV Low-Altitude Turbulence")
    print("=" * 72)
    print("Device: {}".format(device))
    print("Seeds: {}\n".format(seeds))

    per_seed_results = []
    for seed in seeds:
        print("[Seed {}] Running rollout transfer experiment...".format(seed))
        seed_result = run_single_seed(seed, shots, device, use_multi_k_source=use_multi_k_source)
        per_seed_results.append(seed_result)
        diag = seed_result["task_diagnostics"]
        print(
            "  target identity_rel_error={:.4f} cos_sim={:.4f} residual_ratio={:.4f}".format(
                diag["identity_baseline_rel_error"],
                diag["input_output_cosine_similarity"],
                diag["residual_complexity_ratio"],
            )
        )
        print(
            "  source_on_source_rel_error={:.4f} source_on_target_zero_shot={:.4f}\n".format(
                seed_result["source_on_source_rel_error"],
                seed_result["source_on_target_zero_shot"]["rel_error"],
            )
        )

    rel_error_summary = summarize_metric(per_seed_results, shots, strategies, "rel_error")
    coverage_summary = summarize_metric(per_seed_results, shots, strategies, "coverage")
    width_summary = summarize_metric(per_seed_results, shots, strategies, "interval_width")
    gap_trend = summarize_gap_trend(per_seed_results, shots)

    results = {
        "protocol": {
            "few_shot_sizes": shots,
            "task": "u_t -> u_{t+8}",
            "source_pretraining_protocol": "mse_adam_best_checkpoint",
            "seeds": seeds,
            "separate_val_cal_test": True,
            "strategies": strategies,
            "use_multi_k_source": use_multi_k_source,
            "source_domain_variant": "widened_source",
        },
        "per_seed": per_seed_results,
        "aggregates": {
            "source_on_source_rel_error": summarize_scalar(per_seed_results, "source_on_source_rel_error"),
            "source_pretraining_rel_error": summarize_nested_scalar(per_seed_results, "source_pretraining", "rel_error"),
            "source_on_target_zero_shot_rel_error": summarize_nested_scalar(per_seed_results, "source_on_target_zero_shot", "rel_error"),
            "task_identity_rel_error": summarize_nested_scalar(per_seed_results, "task_diagnostics", "identity_baseline_rel_error"),
            "task_cosine_similarity": summarize_nested_scalar(per_seed_results, "task_diagnostics", "input_output_cosine_similarity"),
            "task_residual_complexity_ratio": summarize_nested_scalar(per_seed_results, "task_diagnostics", "residual_complexity_ratio"),
            "identity_baseline_rel_error": summarize_nested_scalar(per_seed_results, "identity_baseline", "rel_error"),
            "rel_error": rel_error_summary,
            "coverage": coverage_summary,
            "interval_width": width_summary,
            "gap_trend_progressive_vs_scratch": gap_trend,
        },
    }

    print("[Summary] RelErr mean ± std")
    print("=" * 72)
    print("{:28s} | {:>15s} | {:>15s} | {:>15s} | {:>15s}".format(
        "Strategy", "n=4", "n=8", "n=12", "n=24"
    ))
    print("-" * 72)
    for strat in strategies:
        vals = []
        for n in shots:
            metric = rel_error_summary[strat][str(n)]
            vals.append("{:.4f}±{:.4f}".format(metric["mean"], metric["std"]))
        print("{:28s} | {:>15s} | {:>15s} | {:>15s} | {:>15s}".format(strat, *vals))

    print("\n[Summary] Gap trend (progressive - scratch)")
    print("-" * 72)
    for n in shots:
        gap = gap_trend["mean_gap_by_shot"][str(n)]
        print("n={}: {:.4f} ± {:.4f}".format(n, gap["mean"], gap["std"]))
    print(
        "monotonic_nonincreasing_count={}/{} slope_mean={:.6f} slope_std={:.6f}".format(
            gap_trend["monotonic_nonincreasing_count"],
            gap_trend["total_seeds"],
            gap_trend["linear_slope"]["mean"],
            gap_trend["linear_slope"]["std"],
        )
    )
    print(
        "source_on_source_rel_error={:.4f}±{:.4f} source_on_target_zero_shot={:.4f}±{:.4f}".format(
            results["aggregates"]["source_on_source_rel_error"]["mean"],
            results["aggregates"]["source_on_source_rel_error"]["std"],
            results["aggregates"]["source_on_target_zero_shot_rel_error"]["mean"],
            results["aggregates"]["source_on_target_zero_shot_rel_error"]["std"],
        )
    )

    with open(r'L:\v3\AI foundation\CDS521 Course Dissertation\transfer_learning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to transfer_learning_results.json")
    print("=" * 72)
