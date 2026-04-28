# Transfer Learning for Low-Altitude Turbulence Prediction

## Overview

UAVs operating in complex low-altitude environments encounter turbulence patterns that are expensive to simulate and dangerous to encounter unprepared. Deep learning surrogates can predict future turbulent states in milliseconds, but they require target-domain labelled data that is scarce in practice.

**Transfer learning** — pre-training on a broad source distribution and fine-tuning on a small target dataset — is the natural solution. This project rigorously tests whether that promise holds for an 8-step rollout turbulence prediction task on a 16×16×16 velocity grid, using a controlled synthetic domain-shift framework.


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
