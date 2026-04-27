# -*- coding: utf-8 -*-
"""Fast validation v3: practical auto-gate + early stopping."""

import copy
import inspect
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from transfer_learning_experiment import (
    DomainShiftTurbulenceDataset,
    L2SPRegularizer,
    compute_loader_rel_error,
    create_model,
    evaluate_model,
    fit_model,
    fit_model_mse,
    make_loaders,
    reset_heads,
    set_trainable,
    split_target_dataset,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
FOCUS_SHOTS = (2, 4, 8)
TARGET_EPOCHS = 30
PROBE_EPOCHS = 8
CHUNK_EPOCHS = 4
TARGET_N_SAMPLES = 220
SOURCE_DOMAIN = "source_wider_plus"
PROBE_MARGIN = 0.03
PROBE_REL_CEILING = 0.9
EARLY_STOP_PATIENCE = 3
MIN_DELTA = 0.001
PROGRESSIVE_MIN_SHOTS = 4
RUN_BASELINES = True


def _seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _make_model():
    sig = inspect.signature(create_model)
    if "model_size" in sig.parameters:
        return create_model(model_size="small")
    return create_model()


def _train_source(seed=SEED):
    _seed_all(seed)
    ds = DomainShiftTurbulenceDataset(n_samples=160, grid_size=16, domain=SOURCE_DOMAIN, rollout_steps=8)
    tr, va = torch.utils.data.random_split(ds, [128, 32], generator=torch.Generator().manual_seed(seed))
    tr_dl = torch.utils.data.DataLoader(tr, batch_size=4, shuffle=True)
    va_dl = torch.utils.data.DataLoader(va, batch_size=4, shuffle=False)
    m = fit_model_mse(_make_model(), tr_dl, va_dl, device=DEVICE, epochs=120, lr=3e-4)
    return copy.deepcopy(m.state_dict())


def _fit_chunk(model, train_dl, val_dl, epochs, muon_lr, adamw_lr, l2sp=None):
    return fit_model(model, train_dl, val_dl, device=DEVICE, epochs=epochs, muon_lr=muon_lr, adamw_lr=adamw_lr, l2sp=l2sp)


def _continue_scratch(model, train_dl, val_dl, used_epochs, total_epochs):
    best_rel = compute_loader_rel_error(model, val_dl, device=DEVICE)
    best_state = copy.deepcopy(model.state_dict())
    traj, stale = [{"epoch": int(used_epochs), "val_rel_error": float(best_rel)}], 0
    while used_epochs < total_epochs:
        chunk = min(CHUNK_EPOCHS, total_epochs - used_epochs)
        model = _fit_chunk(model, train_dl, val_dl, chunk, 0.01, 1e-3)
        used_epochs += chunk
        val_rel = compute_loader_rel_error(model, val_dl, device=DEVICE)
        traj.append({"epoch": int(used_epochs), "val_rel_error": float(val_rel)})
        if val_rel < best_rel - MIN_DELTA:
            best_rel, best_state, stale = val_rel, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= EARLY_STOP_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, used_epochs, traj


def _continue_progressive(model, reg, train_dl, val_dl, used_epochs, total_epochs):
    best_rel = compute_loader_rel_error(model, val_dl, device=DEVICE)
    best_state = copy.deepcopy(model.state_dict())
    traj, stale = [{"epoch": int(used_epochs), "val_rel_error": float(best_rel), "phase": "probe"}], 0
    phases = [("last", 0.002, 2e-4), ("all", 0.001, 1e-4)]
    phase_idx = 0
    while used_epochs < total_epochs:
        phase, muon_lr, adamw_lr = phases[min(phase_idx, 1)]
        set_trainable(model, phase)
        chunk = min(CHUNK_EPOCHS, total_epochs - used_epochs)
        model = _fit_chunk(model, train_dl, val_dl, chunk, muon_lr, adamw_lr, l2sp=reg)
        used_epochs += chunk
        phase_idx += 1
        val_rel = compute_loader_rel_error(model, val_dl, device=DEVICE)
        traj.append({"epoch": int(used_epochs), "val_rel_error": float(val_rel), "phase": phase})
        if val_rel < best_rel - MIN_DELTA:
            best_rel, best_state, stale = val_rel, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= EARLY_STOP_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, used_epochs, traj


def _scratch_probe(train_dl, val_dl):
    m = _make_model()
    m = _fit_chunk(m, train_dl, val_dl, PROBE_EPOCHS, 0.01, 1e-3)
    return m, float(compute_loader_rel_error(m, val_dl, device=DEVICE)), PROBE_EPOCHS


def _progressive_probe(src_state, train_dl, val_dl):
    h_ep = max(1, PROBE_EPOCHS // 2)
    l_ep = max(1, PROBE_EPOCHS - h_ep)
    m = _make_model()
    m.load_state_dict(copy.deepcopy(src_state))
    reset_heads(m)
    reg = L2SPRegularizer(src_state, m, alpha=5e-3, beta=1e-4)
    set_trainable(m, "heads")
    m = _fit_chunk(m, train_dl, val_dl, h_ep, 0.004, 4e-4, l2sp=reg)
    set_trainable(m, "last")
    m = _fit_chunk(m, train_dl, val_dl, l_ep, 0.002, 2e-4, l2sp=reg)
    return m, reg, float(compute_loader_rel_error(m, val_dl, device=DEVICE)), h_ep + l_ep


def _run_auto_gate(src_state, train_dl, val_dl, cal_dl, test_dl, n_shots):
    t0 = time.perf_counter()
    t = time.perf_counter(); sm, srel, su = _scratch_probe(train_dl, val_dl); st = time.perf_counter() - t
    t = time.perf_counter(); pm, reg, prel, pu = _progressive_probe(src_state, train_dl, val_dl); pt = time.perf_counter() - t
    gain = srel - prel

    scratch_probe_ok = srel <= PROBE_REL_CEILING
    progressive_probe_ok = prel <= PROBE_REL_CEILING
    regime_ok = n_shots >= PROGRESSIVE_MIN_SHOTS
    use_prog = regime_ok and progressive_probe_ok and (gain >= PROBE_MARGIN)

    # If probes are both unstable, fall back to scratch path for robustness and speed.
    if (not scratch_probe_ok) and (not progressive_probe_ok):
        use_prog = False

    if use_prog:
        chosen = "fast_progressive"
        fm, used, traj = _continue_progressive(pm, reg, train_dl, val_dl, pu, TARGET_EPOCHS)
    else:
        chosen = "scratch"
        fm, used, traj = _continue_scratch(sm, train_dl, val_dl, su, TARGET_EPOCHS)

    met = evaluate_model(fm, cal_dl, test_dl, device=DEVICE)
    met.update({
        "used_epochs": int(used), "chosen_by_probe": chosen,
        "regime_allows_progressive": bool(regime_ok),
        "probe_scratch_ok": bool(scratch_probe_ok),
        "probe_progressive_ok": bool(progressive_probe_ok),
        "probe_rel_ceiling": float(PROBE_REL_CEILING),
        "probe_scratch_val_rel_error": float(srel), "probe_progressive_val_rel_error": float(prel),
        "probe_gain_progressive": float(gain), "probe_margin": float(PROBE_MARGIN),
        "scratch_probe_time_sec": float(st), "progressive_probe_time_sec": float(pt),
        "early_stop_trajectory": traj, "gate_wall_time_sec": float(time.perf_counter() - t0),
    })
    return met


def _run_scratch_baseline(train_dl, val_dl, cal_dl, test_dl):
    m, _, used = _scratch_probe(train_dl, val_dl)
    m, used, traj = _continue_scratch(m, train_dl, val_dl, used, TARGET_EPOCHS)
    met = evaluate_model(m, cal_dl, test_dl, device=DEVICE)
    met["used_epochs"] = int(used); met["early_stop_trajectory"] = traj
    return met


def _run_fast_progressive_baseline(src_state, train_dl, val_dl, cal_dl, test_dl):
    m, reg, _, used = _progressive_probe(src_state, train_dl, val_dl)
    m, used, traj = _continue_progressive(m, reg, train_dl, val_dl, used, TARGET_EPOCHS)
    met = evaluate_model(m, cal_dl, test_dl, device=DEVICE)
    met["used_epochs"] = int(used); met["early_stop_trajectory"] = traj
    return met


def run_fast_validation(seed=SEED):
    _seed_all(seed)
    src_state = _train_source(seed)
    target_ds = DomainShiftTurbulenceDataset(n_samples=TARGET_N_SAMPLES, grid_size=16, domain="target", rollout_steps=8)
    splits, val_ds, cal_ds, test_ds = split_target_dataset(target_ds, shots=FOCUS_SHOTS, split_seed=seed, min_pool=max(FOCUS_SHOTS))
    results = {
        "seed": seed, "shots": list(FOCUS_SHOTS), "target_epochs": TARGET_EPOCHS,
        "probe_epochs": PROBE_EPOCHS, "chunk_epochs": CHUNK_EPOCHS, "source_domain": SOURCE_DOMAIN,
        "probe_margin": PROBE_MARGIN, "probe_rel_ceiling": PROBE_REL_CEILING, "early_stop_patience": EARLY_STOP_PATIENCE,
        "min_delta": MIN_DELTA, "progressive_min_shots": PROGRESSIVE_MIN_SHOTS, "experiments": {},
    }
    for n in FOCUS_SHOTS:
        tr, va, ca, te = make_loaders(splits[str(n)], val_ds, cal_ds, test_ds)
        results["experiments"][str(n)] = {}
        if RUN_BASELINES:
            t = time.perf_counter(); s = _run_scratch_baseline(tr, va, ca, te); s["wall_time_sec"] = time.perf_counter() - t
            t = time.perf_counter(); p = _run_fast_progressive_baseline(src_state, tr, va, ca, te); p["wall_time_sec"] = time.perf_counter() - t
            results["experiments"][str(n)]["scratch_early_stop"] = s
            results["experiments"][str(n)]["fast_progressive_early_stop"] = p
        t = time.perf_counter(); g = _run_auto_gate(src_state, tr, va, ca, te, n); g["wall_time_sec"] = time.perf_counter() - t
        results["experiments"][str(n)]["auto_gate"] = g
    return results


def plot_results(results, out_png):
    shots = [str(s) for s in results["shots"]]; x = np.arange(len(shots))
    series = ["scratch_early_stop", "fast_progressive_early_stop", "auto_gate"] if RUN_BASELINES else ["auto_gate"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in series:
        rel = [results["experiments"][s][name]["rel_error"] for s in shots]
        tim = [results["experiments"][s][name]["wall_time_sec"] for s in shots]
        axes[0].plot(x, rel, marker="o", label=name)
        axes[1].plot(x, tim, marker="o", label=name)
    axes[0].set_xticks(x, shots); axes[0].set_title("Rel error (lower is better)"); axes[0].set_xlabel("shots"); axes[0].set_ylabel("rel_error"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_xticks(x, shots); axes[1].set_title("Wall time per setting (sec)"); axes[1].set_xlabel("shots"); axes[1].set_ylabel("seconds"); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)


def print_summary(results):
    print("=" * 72); print("FAST GATE VALIDATION SUMMARY (v3)"); print("=" * 72)
    for n in results["shots"]:
        key = str(n); print(f"\nshots={n}")
        if RUN_BASELINES:
            s = results["experiments"][key]["scratch_early_stop"]; p = results["experiments"][key]["fast_progressive_early_stop"]
            print(f"  scratch_es       rel={s['rel_error']:.4f} time={s['wall_time_sec']:.1f}s epochs={s['used_epochs']}")
            print(f"  fast_prog_es     rel={p['rel_error']:.4f} time={p['wall_time_sec']:.1f}s epochs={p['used_epochs']}")
        g = results["experiments"][key]["auto_gate"]
        print(f"  auto_gate        rel={g['rel_error']:.4f} time={g['wall_time_sec']:.1f}s epochs={g['used_epochs']} chosen={g['chosen_by_probe']} gain={g['probe_gain_progressive']:.4f} probe_s={g['probe_scratch_val_rel_error']:.4f} probe_p={g['probe_progressive_val_rel_error']:.4f}")


if __name__ == "__main__":
    results = run_fast_validation(seed=SEED)
    print_summary(results)
    out_json = Path("transferability_fast_gate_results_v3.json")
    out_png = Path("transferability_fast_gate_plot_v3.png")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    plot_results(results, out_png)
    print(f"\nSaved JSON: {out_json}")
    print(f"Saved plot: {out_png}")
