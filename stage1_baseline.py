# -*- coding: utf-8 -*-
"""
Attention Residuals Validation - Stage 1: Baseline Measurement
Measures untrained model performance (~10 seconds)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict


class SyntheticTurbulenceDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=120, grid_size=16, re_tau=1000):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.nu = 1.0 / re_tau
        self._generate_data()

    def _generate_velocity_field(self):
        N = self.grid_size
        u_hat = np.random.randn(3,N,N,N) + 1j*np.random.randn(3,N,N,N)
        k = np.fft.fftfreq(N, 1.0/N)
        kx,ky,kz = np.meshgrid(k,k,k,indexing='ij')
        k_mag = np.sqrt(kx**2+ky**2+kz**2)
        E_k = np.zeros_like(k_mag)
        mask = k_mag > 0.1
        E_k[mask] = k_mag[mask] ** (-5.0 / 3.0)
        u_hat *= np.sqrt(E_k+1e-8)
        k_dot_u = kx*u_hat[0]+ky*u_hat[1]+kz*u_hat[2]
        k2 = k_mag**2+1e-8
        u_hat[0] -= (kx*k_dot_u)/k2
        u_hat[1] -= (ky*k_dot_u)/k2
        u_hat[2] -= (kz*k_dot_u)/k2
        return np.real(np.fft.ifftn(u_hat, axes=(1,2,3)))

    def _advect_diffuse(self, u):
        u_hat = np.fft.fftn(u, axes=(1,2,3))
        N = self.grid_size
        k = np.fft.fftfreq(N, 1.0/N)
        kx,ky,kz = np.meshgrid(k,k,k,indexing='ij')
        k2 = kx**2+ky**2+kz**2
        u_hat *= np.exp(-self.nu*k2*0.01)
        return np.real(np.fft.ifftn(u_hat, axes=(1,2,3)))

    def _generate_data(self):
        self.data = []
        for _ in range(self.n_samples):
            u_t = self._generate_velocity_field()
            u_t1 = self._advect_diffuse(u_t)
            self.data.append((u_t.reshape(3,-1).T, u_t1.reshape(3,-1).T))

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        u_t, u_t1 = self.data[idx]
        return torch.from_numpy(u_t).float(), torch.from_numpy(u_t1).float()


def validate_qiapt(model, test_loader, device="cpu", n_ensemble=3):
    model.to(device)
    model.eval()
    all_rel_errors = []
    all_coverages = []
    with torch.no_grad():
        for u_t, u_t1_true in test_loader:
            u_t = u_t.to(device)
            u_t1_true = u_t1_true.cpu().numpy()
            preds = []
            for _ in range(n_ensemble):
                mu, lv = model(u_t)
                std = torch.exp(0.5*lv).clamp(max=10.)
                preds.append((mu + std*torch.randn_like(mu)).cpu().numpy())
            pred_mean = np.mean(preds, axis=0)
            pred_std  = np.std(preds, axis=0)
            rel_err = np.linalg.norm(pred_mean-u_t1_true)/(np.linalg.norm(u_t1_true)+1e-8)
            all_rel_errors.append(rel_err)
            lower = pred_mean - 1.96*pred_std
            upper = pred_mean + 1.96*pred_std
            coverage = np.mean((u_t1_true >= lower) & (u_t1_true <= upper))
            all_coverages.append(coverage)
    return {
        "rel_error": float(np.mean(all_rel_errors)),
        "uncertainty_coverage": float(np.mean(all_coverages)),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Attention Residuals - Stage 1: Baseline Measurement (Untrained Model)")
    print("=" * 70)

    from Attention_Residuals_implementation import AttentionResidualTurbulenceModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}\n".format(device))

    print("[1/2] Generating synthetic turbulence dataset...")
    dataset = SyntheticTurbulenceDataset(n_samples=120, grid_size=16)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    print("  {} samples  grid=16^3=4096 pts\n".format(len(dataset)))

    print("[2/2] Creating model and measuring baseline...")
    model = AttentionResidualTurbulenceModel(in_ch=3, dim=32, n_layers=3, n_heads=4, out_ch=3, sparse_k=2)
    print("  Parameters: {:,}".format(sum(p.numel() for p in model.parameters())))
    print("  Measuring untrained performance...\n")

    baseline = validate_qiapt(model, test_loader, device=device, n_ensemble=3)

    print("=" * 70)
    print("BASELINE RESULTS (Attention Residuals, Untrained Model)")
    print("=" * 70)
    print("  Relative prediction error:  {:.4f}".format(baseline["rel_error"]))
    print("  95% uncertainty coverage:   {:.3f}".format(baseline["uncertainty_coverage"]))
    print()
    print("  Note: High error is expected (untrained model)")
    print("        But uncertainty coverage should be ~0.95 (well-calibrated)")
    print()

    # Save baseline for later comparison
    import json
    with open(r'L:\v3\AI foundation\CDS521 Course Dissertation\attention_residuals_baseline.json', 'w') as f:
        json.dump(baseline, f)
    print("  Baseline saved to attention_residuals_baseline.json")
    print("=" * 70)
