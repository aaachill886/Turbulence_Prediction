# -*- coding: utf-8 -*-
"""
Attention Residuals for Industrial Turbulence Prediction
Inspired by cross-layer attention aggregation (arXiv:2603.15031, MoonshotAI, 2026-03-15)

Core spirit: Pursuing perfect deterministic prediction (optimal plan)
-> guaranteed collapse beyond Lyapunov T* (worst outcome).
Inverse: embrace statistical description -> superior industrial reliability.

Current risks per implementation:
  1. Helmholtz projection: FFT branch assumes cubic periodic grids; fallback otherwise
  2. Muon optimizer: promising but still validate stability on physics-constrained losses
  3. NLL loss: logvar can collapse; monitor head_lv outputs
  4. Phase-3 PGD: eps must be tuned per dataset

Run: python Attention_Residuals_implementation.py
"""

import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Tuple, List, Dict, Optional


# ================================================================
# SECTION 1: Muon Optimizer  (replaces SAM)
# ================================================================
# SAM  : perturbs weights toward worst-case direction, steps back -> 2x cost
# Muon : orthogonalizes gradient via Newton-Schulz iteration     -> ~1x cost
# Both target flat minima. Muon does so without explicit worst-case search.
# Reductio alignment: passively avoids sharp loss regions by geometry.
#
# RISK: NS requires 2D matrices. Apply Muon to Linear.weight only.
#       Use AdamW for bias, LayerNorm params, and output heads.
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

    @staticmethod
    def _ns(G, steps):
        assert G.ndim == 2
        G = G / (G.norm() + 1e-8)
        t = G.shape[0] > G.shape[1]
        if t: G = G.T
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G
        for _ in range(steps):
            A = X @ X.T
            X = a*X + b*(A@X) + c*(A@A@X)
        return X.T if t else X

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None or p.grad.ndim != 2: continue
                grad = p.grad
                s = self.state[p]
                if 'buf' not in s: s['buf'] = torch.zeros_like(grad)
                s['buf'].mul_(g['momentum']).add_(grad)
                gn = grad + g['momentum'] * s['buf']
                go = self._ns(gn, g['ns_steps'])
                p.add_(go * (gn.norm() / (go.norm() + 1e-8)), alpha=-g['lr'])
        return loss


def make_optimizer(model, muon_lr=0.02, adamw_lr=3e-4):
    """2D weights -> Muon; everything else -> AdamW."""
    w2 = [p for p in model.parameters() if p.ndim == 2 and p.requires_grad]
    w1 = [p for p in model.parameters() if p.ndim != 2 and p.requires_grad]
    return Muon(w2, lr=muon_lr), torch.optim.AdamW(w1, lr=adamw_lr, weight_decay=1e-4)


# ================================================================
# SECTION 2: Attention Residuals Block  (arXiv:2603.15031)
# ================================================================
# h_l = SUM_{j<l} a_{l,j}*h_j + Attn_l(h_{l-1}) + FFN_l(h_{l-1})
# RISK: O(L^2 B N D) memory. Use sparse_k to aggregate every k-th layer.
class AttentionResidualBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, sparse_k=None):
        super().__init__()
        assert dim % num_heads == 0
        self.sparse_k = sparse_k
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cq    = nn.Linear(dim, dim, bias=False)
        self.ck    = nn.Linear(dim, dim, bias=False)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, x, history):
        B, N, D = x.shape
        n1 = self.norm1(x)
        ao, aw = self.attn(n1, n1, n1, need_weights=True)
        hist = history[::self.sparse_k] if (self.sparse_k and history) else history
        if hist:
            Lh = len(hist)
            hs = torch.stack(hist, 1)                   # (B,Lh,N,D)
            q  = self.cq(x).reshape(B*N, 1, D)
            k  = self.ck(hs.permute(0,2,1,3).reshape(B*N,Lh,D))
            v  = hs.permute(0,2,1,3).reshape(B*N,Lh,D)
            cr = torch.bmm(
                torch.softmax(torch.bmm(q,k.transpose(1,2))/D**0.5,dim=-1),v
            ).reshape(B,N,D)
        else:
            cr = torch.zeros_like(x)
        return x + ao + cr + self.ffn(self.norm2(x)), aw


# ================================================================
# SECTION 3: Attention Residuals Turbulence Model
# ================================================================
class AttentionResidualTurbulenceModel(nn.Module):
    """Outputs (mean, log_variance) for industrial turbulence prediction."""
    def __init__(self, in_ch=4, dim=256, n_layers=12, n_heads=8,
                 out_ch=4, dropout=0.1, sparse_k=None):
        super().__init__()
        self.embed   = nn.Linear(in_ch, dim)
        self.blocks  = nn.ModuleList([
            AttentionResidualBlock(dim, n_heads, dropout, sparse_k)
            for _ in range(n_layers)])
        self.norm    = nn.LayerNorm(dim)
        self.head_mu = nn.Linear(dim, out_ch)
        self.head_lv = nn.Linear(dim, out_ch)

    def forward(self, x):
        h, hist = self.embed(x), []
        for blk in self.blocks:
            h, _ = blk(h, hist)
            hist.append(h)
        h = self.norm(h)
        return self.head_mu(h), self.head_lv(h)

    @torch.no_grad()
    def ensemble_predict(self, x, n=50):
        """MC Dropout ensemble. Call model.eval() after if continuing training."""
        self.train()
        preds = [self(x)[0] + torch.exp(0.5*self(x)[1]).clamp(max=10.)*torch.randn_like(x[...,:self.head_mu.out_features]) for _ in range(n)]
        p = torch.stack(preds)
        return p.mean(0), p.std(0)


# Backward-compatible alias for stage scripts and older checkpoints/docs
QIAPTModel = AttentionResidualTurbulenceModel


# ================================================================
# SECTION 4: Physics-Informed Loss
# ================================================================
# RISK: finite-diff continuity_loss is inaccurate at high Re.
#       Production: replace torch.diff with FFT spectral derivative.
class PhysicsInformedLoss(nn.Module):
    def __init__(self, max_phy_w=0.1, warmup=50, dx=0.01):
        super().__init__()
        self.max_phy_w = max_phy_w
        self.warmup = warmup
        self.dx = dx

    def nll(self, mu, lv, target):
        lv = lv.clamp(-10.0, 10.0)
        prec = torch.exp(-lv).clamp(max=1e6)
        return (0.5*(lv + (target-mu).pow(2)*prec)).mean()

    def continuity(self, u):
        div = torch.diff(u[...,0], dim=-1) / self.dx
        raw = div.clamp(-1e4, 1e4).pow(2)
        # Normalize: raw residual is O(1/dx^2) >> data loss without this fix.
        return (raw / (raw.detach().mean() + 1e-6)).mean()

    def forward(self, mu, lv, target, epoch):
        w = self.max_phy_w * min(1.0, epoch / max(self.warmup, 1))
        ld = self.nll(mu, lv, target)
        lp = self.continuity(mu[...,:3])
        return {"total": ld + w*lp, "data": ld, "physics": lp}


# ================================================================
# SECTION 5: Helmholtz-Projected PGD
# ================================================================
# RISK: projection assumes cubic periodic grid for FFT branch.
#       If token count is not a perfect cube, it falls back to mean-subtraction.
def helmholtz_project(delta):
    """
    Divergence-free projection.

    Preferred path (production-like): 3D FFT Helmholtz projection on periodic cubic grids.
      - Supports flattened shape (B, N, C) with N = M^3 and C>=3
      - Supports grid shape     (B, M, M, M, C) with C>=3

    Fallback path (legacy approximation): mean-subtraction when shape is incompatible.
    """
    if delta.shape[-1] < 3:
        return delta - delta.mean(dim=-2, keepdim=True)

    # Accept either flattened tokens or explicit 3D grid
    if delta.ndim == 3:  # (B, N, C)
        B, N, C = delta.shape
        M = round(N ** (1.0 / 3.0))
        if M * M * M != N:
            return delta - delta.mean(dim=-2, keepdim=True)
        u = delta.reshape(B, M, M, M, C)
        need_flatten = True
    elif delta.ndim == 5:  # (B, M, M, M, C)
        B, M, M2, M3, C = delta.shape
        if not (M == M2 == M3):
            return delta - delta.mean(dim=-2, keepdim=True)
        u = delta
        need_flatten = False
    else:
        return delta - delta.mean(dim=-2, keepdim=True)

    vec = u[..., :3]
    other = u[..., 3:] if C > 3 else None

    # FFT over spatial axes
    vhat = torch.fft.fftn(vec, dim=(1, 2, 3))  # complex tensor, shape (B,M,M,M,3)

    k = torch.fft.fftfreq(M, d=1.0, device=delta.device, dtype=delta.dtype)
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    kx = kx[None, ...]
    ky = ky[None, ...]
    kz = kz[None, ...]

    k2 = kx * kx + ky * ky + kz * kz
    k2_safe = torch.where(k2 == 0, torch.ones_like(k2), k2)

    k_dot_v = kx * vhat[..., 0] + ky * vhat[..., 1] + kz * vhat[..., 2]

    vhat0 = vhat[..., 0] - kx * k_dot_v / k2_safe
    vhat1 = vhat[..., 1] - ky * k_dot_v / k2_safe
    vhat2 = vhat[..., 2] - kz * k_dot_v / k2_safe
    vhat_proj = torch.stack([vhat0, vhat1, vhat2], dim=-1)

    # Explicitly zero out the k=0 mode correction ambiguity
    vhat_proj[:, 0, 0, 0, :] = 0

    vproj = torch.fft.ifftn(vhat_proj, dim=(1, 2, 3)).real

    if other is not None:
        out = torch.cat([vproj, other], dim=-1)
    else:
        out = vproj

    return out.reshape(B, M * M * M, C) if need_flatten else out


def pgd_adversarial(model, x, y, loss_fn, eps=0.01, alpha=0.002, steps=5, epoch=100):
    delta = torch.zeros_like(x)
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = loss_fn(*model(x+delta), y, epoch)["total"]
        loss.backward()
        with torch.no_grad():
            delta = delta.detach() + alpha * delta.grad.sign()
            delta = helmholtz_project(delta)
            delta = delta.clamp(-eps, eps)
    return (x + delta.detach()).detach()


# ================================================================
# SECTION 6: Three-Phase Training Workflow
# ================================================================
# Phase 1 (ep   0-50):  NLL data loss only
# Phase 2 (ep  51-150): + physics warmup (NS continuity)
# Phase 3 (ep 151-200): + Helmholtz-PGD adversarial finetuning
#
# RISK: Muon lr (0.02) >> AdamW lr (3e-4). If loss spikes, reduce muon_lr to 0.01.
def train_qiapt(model, train_dl, val_dl, epochs=200, adv_start=150, device="cpu"):
    model = model.to(device)
    loss_fn = PhysicsInformedLoss()
    muon, adamw = make_optimizer(model)
    sm = CosineAnnealingLR(muon,  T_max=epochs, eta_min=1e-4)
    sa = CosineAnnealingLR(adamw, T_max=epochs, eta_min=3e-6)
    log = {"train": [], "val": []}

    for ep in range(epochs):
        model.train()
        tl = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            if ep >= adv_start:
                xb = pgd_adversarial(model, xb, yb, loss_fn, epoch=ep)
            muon.zero_grad(); adamw.zero_grad()
            L = loss_fn(*model(xb), yb, ep)["total"]
            L.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            muon.step(); adamw.step()
            tl.append(L.item())
        sm.step(); sa.step()

        model.eval()
        vl = []
        with torch.no_grad():
            for xv, yv in val_dl:
                xv, yv = xv.to(device), yv.to(device)
                vl.append(loss_fn(*model(xv), yv, ep)["total"].item())
        log["train"].append(float(np.mean(tl)))
        log["val"].append(float(np.mean(vl)))
        if ep % 10 == 0:
            phase = "P1" if ep<50 else "P2" if ep<adv_start else "P3-Adv"
            print(f"ep={ep:3d} [{phase}] train={log['train'][-1]:.4f} val={log['val'][-1]:.4f}")
    return log


# ================================================================
# SECTION 7: Validation of New Risks + Unit Tests
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Attention Residuals Unit Tests + Risk Validation")
    print("=" * 60)
    torch.manual_seed(42)
    B, N, C = 2, 64, 4

    # --- Model forward pass ---
    m = QIAPTModel(in_ch=C, dim=32, n_layers=3, n_heads=4, out_ch=C, sparse_k=2)
    x = torch.randn(B, N, C)
    mu, lv = m(x)
    assert mu.shape == lv.shape == (B, N, C)
    print(f"[PASS] Forward: {x.shape} -> mu{mu.shape} lv{lv.shape}")

    # --- RISK 3: logvar collapse check ---
    lv_mean = lv.mean().item()
    lv_flag = "[WARN logvar collapsed]" if lv_mean < -10 else "[OK]"
    print(f"[INFO] logvar mean={lv_mean:.3f} {lv_flag}")

    # --- Ensemble predict ---
    em, es = m.ensemble_predict(x, n=5)
    assert em.shape == es.shape == (B, N, C)
    print(f"[PASS] Ensemble: mean{em.shape} std{es.shape}")
    m.eval()

    # --- Loss ---
    lf = PhysicsInformedLoss()
    y  = torch.randn_like(mu)
    L  = lf(mu, lv, y, epoch=100)
    assert "total" in L
    print(f"[PASS] Loss: total={L['total'].item():.4f} data={L['data'].item():.4f} physics={L['physics'].item():.4f}")

    # --- RISK 1: Helmholtz projection check (3D FFT branch) ---
    delta = torch.randn(B, N, C)

    def approx_divergence_l2(field, eps=1e-12):
        """Compute RMS divergence on flattened cubic grid (B,N,C)."""
        Bb, Nn, Cc = field.shape
        M = round(Nn ** (1.0 / 3.0))
        if M * M * M != Nn or Cc < 3:
            return float('nan')
        u = field[..., :3].reshape(Bb, M, M, M, 3)
        uhat = torch.fft.fftn(u, dim=(1, 2, 3))
        k = torch.fft.fftfreq(M, d=1.0, device=field.device, dtype=field.dtype)
        kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
        kx = kx[None, ...]
        ky = ky[None, ...]
        kz = kz[None, ...]
        div_hat = 1j * (kx * uhat[..., 0] + ky * uhat[..., 1] + kz * uhat[..., 2])
        div = torch.fft.ifftn(div_hat, dim=(1, 2, 3)).real
        return div.pow(2).mean().sqrt().item()

    delta_proj = helmholtz_project(delta)
    div_before = approx_divergence_l2(delta)
    div_after = approx_divergence_l2(delta_proj)
    print(f"[INFO] Helmholtz projection (FFT): div-rms before={div_before:.6f} after={div_after:.6f}")

    # --- PGD adversarial ---
    xa = pgd_adversarial(m, x, y, lf, steps=2, epoch=160)
    assert xa.shape == x.shape
    perturb = (xa - x).abs().max().item()
    assert perturb <= 0.011, f"PGD exceeded eps: {perturb}"
    print(f"[PASS] PGD: shape={xa.shape} max_perturb={perturb:.4f}")

    # --- RISK 2: Muon optimizer step ---
    muon, adamw = make_optimizer(m)
    muon.zero_grad(); adamw.zero_grad()
    L2 = lf(*m(x), y, epoch=10)["total"]
    L2.backward()
    muon.step(); adamw.step()
    print(f"[PASS] Muon+AdamW step completed (loss={L2.item():.4f})")

    # --- RISK 4: PGD eps budget ---
    xa_big = pgd_adversarial(m, x, y, lf, eps=0.05, steps=3, epoch=160)
    perturb_big = (xa_big - x).abs().max().item()
    print(f"[INFO] PGD eps=0.05: max_perturb={perturb_big:.4f} (tune this per dataset)")

    print()
    print("All tests PASSED.")
    print("Risk summary:")
    print("  RISK 1 (Helmholtz): FFT projection on cubic periodic grids; fallback for incompatible shapes")
    print("  RISK 2 (Muon):      step succeeded -- validate loss curve stability over epochs")
    print(f"  RISK 3 (NLL):       logvar mean={lv_mean:.3f} -- monitor for collapse")
    print("  RISK 4 (PGD eps):   tune eps per dataset before Phase-3 finetuning")
