# Transfer Learning for Low-Altitude Turbulence Prediction

## Overview

UAVs operating in complex low-altitude environments encounter turbulence patterns that are expensive to simulate and dangerous to encounter unprepared. Deep learning surrogates can predict future turbulent states in milliseconds, but they require target-domain labelled data that is scarce in practice.

**Transfer learning** — pre-training on a broad source distribution and fine-tuning on a small target dataset — is the natural solution. This project rigorously tests whether that promise holds for an 8-step rollout turbulence prediction task on a 16×16×16 velocity grid, using a controlled synthetic domain-shift framework.

### The Pipeline
Source Data (160 samples) Target Data (n = 2, 4, 8 shots)
│ │
Pre-train (MSE, 120 ep) Fine-tune (NLL, 30 ep)
│ │
└──── Source Weights θ_S ────────────┘
│
┌────────┴────────┐
│ Auto-Gate │
│ 8-epoch probe │
├──────┬───────────┤
│Scratch│Progressive│
└──────┴───────────┘
│
Evaluate on
60 test samples

## Key Findings

| Finding | Evidence |
|---|---|
| **Zero-shot transfer improves monotonically** | Widening source distribution reduces zero-shot ε_rel from **0.966** (narrow) → **0.841** (base) → **0.704** (wider) → **0.688** (wider+) |
| **Fine-tuning erases all initialisation differences** | After 30 epochs, every strategy converges to ε_rel ≈ **0.495–0.516** regardless of initialisation |
| **Bottleneck is approximation error** | Increasing n from 2 → 60 changes fine-tuned ε_rel by < 0.001 at 120 epochs — model capacity (d=32, L=3) is the binding constraint |
| **Progressive unfreezing can be harmful** | At n=4, forced progressive achieves ε_rel = **1.549** vs scratch's **0.516** (+200% penalty) |
| **Auto-gate correctly vetoes progressive** | Negative probe gain (Δ_gain = −0.62 to −1.33) in all conditions; gate selects scratch every time |
| **Transfer's residual value** | Convergence acceleration (~6–10 epochs faster) and tighter uncertainty intervals (IW ≈ 0.35 vs 0.50) |


## Repository Structure
Turbulence_Prediction/
│
├── Attention_Residuals_implementation.py # Core model architecture
│ ├── Muon optimizer (Newton-Schulz)
│ ├── AttentionResidualBlock (sparse top-k attention)
│ ├── AttentionResidualTurbulenceModel (Gaussian output heads)
│ ├── PhysicsInformedLoss (NLL + continuity)
│ ├── Helmholtz projection (FFT divergence-free)
│ └── PGD adversarial fine-tuning
│
├── transfer_learning_experiment.py # Transfer learning framework
│ ├── DomainShiftTurbulenceDataset (4 source + 1 target domain)
│ ├── MultiKTurbulenceDataset (multi-horizon source)
│ ├── L2SPRegularizer
│ ├── 8 transfer strategies (scratch, naive, warmstart, weak, frozen, L2-SP, progressive, frozen_long)
│ ├── fit_model / fit_model_mse / fit_model_mse_warmstart
│ ├── evaluate_model (calibrated coverage, interval width, CRPS)
│ └── Multi-seed aggregation utilities
│
├── transferability_diagnosis.py # Angle A + B diagnostic experiments
│ ├── Source coverage ablation (narrow → base → wider → wider+)
│ ├── Training intensity sweep (20, 30, 120 epochs × 5 shot counts)
│ ├── Zero-shot evaluation
│ └── Full strategy bundle evaluation with drift tracking
│
├── fast_gate_validation.py # Auto-gate mechanism (v3)
│ ├── Dual 8-epoch probe (scratch vs progressive)
│ ├── Three-condition gate: regime ∧ stability ∧ gain margin
│ ├── Chunked early stopping (chunk=4, patience=3)
│ └── Wall-time benchmarking
│
├── stage1_baseline.py # Initial baseline training
├── stage2_training.py # Three-phase training workflow
├── stage3_validation.py # Validation and uncertainty metrics
│
├── transferability_revised_results.json # Full Angle A+B results (~800KB)
├── transferability_fast_gate_results_v3.json # Auto-gate results
├── attention_residuals_baseline.json # Baseline model metrics
├── attention_residuals_model_trained.pt # Pre-trained model checkpoint
│
└── README.md # This file

### Attention-Residual Transformer

The model uses cross-layer attention aggregation inspired by [arXiv:2603.15031]:
Input u_t ∈ ℝ^(N×3) → Embed(3→d) → [Block₁ → Block₂ → Block₃] → LayerNorm → head_μ, head_lv
↑ ↑ ↑
h₀ history h₀,h₁ h₀,h₁,h₂

Each `AttentionResidualBlock` computes:
h_l = h_{l-1} + Attn(h_{l-1}) + CrossAttn(h_{l-1}, history[::sparse_k]) + FFN(h_{l-1})


| Parameter | Small (used) | Base | Large |
|---|---|---|---|
| Hidden dim `d` | 32 | 64 | 128 |
| Layers `L` | 3 | 6 | 8 |
| Heads `H` | 4 | 4 | 8 |
| Sparse `k` | 2 | 2 | 2 |
| Approx. params | ~15K | ~60K | ~250K |

### Optimiser: Muon + AdamW

- **Muon** (2D weight matrices): Orthogonalises gradients via Newton-Schulz iteration → flat minima without explicit perturbation (cf. SAM)
- **AdamW** (biases, LayerNorm, heads): Standard adaptive optimiser with weight decay

```python  
# Automatic parameter routing  
w2d = [p for p in model.parameters() if p.ndim == 2]   # → Muon  
w1d = [p for p in model.parameters() if p.ndim != 2]   # → AdamW  

L_NLL = (1/T) Σ [ log σ² / 2 + (u - μ)² / (2σ²) ]

Post-hoc calibration searches s ∈ [0.15, 1.20] to achieve 93% coverage

### Quick Start

Run the auto-gate experiment (python fast_gate_validation.py)
Outputs:

transferability_fast_gate_results_v3.json — full metrics
transferability_fast_gate_plot_v3.png — comparison plot

Run the full diagnostic suite (python transferability_diagnosis.py)
Outputs:
transferability_revised_results.json — Angle A + B results

Run the base model validation (python Attention_Residuals_implementation.py)
Outputs:
Unit test results for all components (Muon, Helmholtz, PGD, loss)

Run multi-seed transfer experiment (python transfer_learning_experiment.py)
Outputs:
transfer_learning_results.json — 3-seed × 4-shot × 8-strategy results


### Reproducibility

All experiments use deterministic seeding:
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

### Requirements

Python >= 3.10
torch >= 2.0.0
numpy >= 1.24.0
matplotlib >= 3.7.0

Install:
bash
pip install torch numpy matplotlib
