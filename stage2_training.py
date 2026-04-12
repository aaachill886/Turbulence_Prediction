# -*- coding: utf-8 -*-
"""
Attention Residuals Validation - Stage 2: Training
Trains model for 200 epochs (~10-15 minutes)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


class SyntheticTurbulenceDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=120, grid_size=16, re_tau=1000):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.nu = 1.0 / re_tau
        self._generate_data()

    def _generate_velocity_field(self):
        N = self.grid_size
        u_hat = np.random.randn(3, N, N, N) + 1j * np.random.randn(3, N, N, N)
        k = np.fft.fftfreq(N, 1.0 / N)
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
        E_k = np.zeros_like(k_mag)
        mask = k_mag > 0.1
        E_k[mask] = k_mag[mask] ** (-5.0 / 3.0)
        u_hat *= np.sqrt(E_k + 1e-8)
        k_dot_u = kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]
        k2 = k_mag**2 + 1e-8
        u_hat[0] -= (kx * k_dot_u) / k2
        u_hat[1] -= (ky * k_dot_u) / k2
        u_hat[2] -= (kz * k_dot_u) / k2
        return np.real(np.fft.ifftn(u_hat, axes=(1, 2, 3)))

    def _advect_diffuse(self, u):
        u_hat = np.fft.fftn(u, axes=(1, 2, 3))
        N = self.grid_size
        k = np.fft.fftfreq(N, 1.0 / N)
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        k2 = kx**2 + ky**2 + kz**2
        u_hat *= np.exp(-self.nu * k2 * 0.01)
        return np.real(np.fft.ifftn(u_hat, axes=(1, 2, 3)))

    def _generate_data(self):
        self.data = []
        for _ in range(self.n_samples):
            u_t = self._generate_velocity_field()
            u_t1 = self._advect_diffuse(u_t)
            self.data.append((u_t.reshape(3, -1).T, u_t1.reshape(3, -1).T))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        u_t, u_t1 = self.data[idx]
        return torch.from_numpy(u_t).float(), torch.from_numpy(u_t1).float()


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("Attention Residuals - Stage 2: Training (200 epochs)")
    print("=" * 70)

    from Attention_Residuals_implementation import AttentionResidualTurbulenceModel, PhysicsInformedLoss, make_optimizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}\n".format(device))

    print("[1/3] Generating dataset and split (train/val)...")
    dataset = SyntheticTurbulenceDataset(n_samples=120, grid_size=16)
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False)
    print("  total={} train={} val={}  grid=16^3=4096 pts\n".format(len(dataset), len(train_set), len(val_set)))

    print("[2/3] Creating model...")
    model = AttentionResidualTurbulenceModel(in_ch=3, dim=32, n_layers=3, n_heads=4, out_ch=3, sparse_k=2)
    print("  Parameters: {:,}\n".format(sum(p.numel() for p in model.parameters())))

    print("[3/3] Training 200 epochs (NLL + warmup physics)...")
    model = model.to(device)
    loss_fn = PhysicsInformedLoss(max_phy_w=0.03, warmup=30)
    muon, adamw = make_optimizer(model, muon_lr=0.008, adamw_lr=8e-4)
    sm = CosineAnnealingLR(muon, T_max=200, eta_min=1e-4)
    sa = CosineAnnealingLR(adamw, T_max=200, eta_min=1e-5)

    train_losses = []
    val_losses = []
    for ep in range(200):
        model.train()
        ep_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            muon.zero_grad()
            adamw.zero_grad()
            L = loss_fn(*model(xb), yb, ep)["total"]
            L.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            muon.step()
            adamw.step()
            ep_losses.append(L.item())

        model.eval()
        ep_val_losses = []
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                Lv = loss_fn(*model(xv), yv, ep)["total"]
                ep_val_losses.append(Lv.item())

        sm.step()
        sa.step()
        train_losses.append(float(np.mean(ep_losses)))
        val_losses.append(float(np.mean(ep_val_losses)))
        if ep % 10 == 0:
            print("  ep={:3d}  train_loss={:.4f}  val_loss={:.4f}".format(
                ep, train_losses[-1], val_losses[-1]))

    print("  Training done.")
    print("  Train loss: {:.4f} -> {:.4f}  (ratio: {:.3f}x)".format(
        train_losses[0], train_losses[-1],
        train_losses[-1] / (train_losses[0] + 1e-8)))
    print("  Val loss:   {:.4f} -> {:.4f}  (ratio: {:.3f}x)\n".format(
        val_losses[0], val_losses[-1],
        val_losses[-1] / (val_losses[0] + 1e-8)))

    torch.save(model.state_dict(), r'L:\v3\AI foundation\CDS521 Course Dissertation\attention_residuals_model_trained.pt')
    print("  Model saved to attention_residuals_model_trained.pt")
    print("=" * 70)
