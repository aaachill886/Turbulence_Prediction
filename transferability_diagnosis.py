# -*- coding: utf-8 -*-
"""Revised transferability diagnosis experiments.

Most likely transfer-gain directions in current evidence:
1) strengthen source-target dynamics overlap (source_wider_plus),
2) preserve useful source prior while adapting gradually (progressive unfreeze),
3) keep uncertainty-aware metrics and weak controls for causal diagnosis.
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch

from transfer_learning_experiment import (
    DomainShiftTurbulenceDataset,
    L2SPRegularizer,
    MultiKTurbulenceDataset,
    compute_state_drift,
    create_model,
    evaluate_model,
    fit_model,
    fit_model_mse,
    make_loaders,
    progressive_unfreeze,
    reset_heads,
    set_trainable,
    split_target_dataset,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SOURCE_VARIANTS = ["source_narrow", "source", "source_wider", "source_wider_plus"]
FOCUS_SHOTS = (2, 4, 8, 24, 60)
TARGET_EPOCH_SWEEP = [20, 30, 120]
TARGET_N_SAMPLES = 220


def _train_source_model(seed, source_domain="source", use_multi_k_source=False, model_size="small", arch_overrides=None):
    arch_overrides = arch_overrides or {}
    if use_multi_k_source:
        source_ds = MultiKTurbulenceDataset(
            n_samples=160,
            grid_size=16,
            re_tau=1000,
            rollout_steps_list=(1, 2, 4, 8),
            domain=source_domain,
        )
    else:
        source_ds = DomainShiftTurbulenceDataset(
            n_samples=160,
            grid_size=16,
            domain=source_domain,
            rollout_steps=8,
        )

    src_train, src_val = torch.utils.data.random_split(
        source_ds,
        [128, 32],
        generator=torch.Generator().manual_seed(seed),
    )
    src_train_dl = torch.utils.data.DataLoader(src_train, batch_size=4, shuffle=True)
    src_val_dl = torch.utils.data.DataLoader(src_val, batch_size=4, shuffle=False)
    src_model = fit_model_mse(
        create_model(model_size=model_size, **arch_overrides),
        src_train_dl,
        src_val_dl,
        device=DEVICE,
        epochs=120,
        lr=3e-4,
    )
    return source_ds, src_model, copy.deepcopy(src_model.state_dict())


def _evaluate_zero_shot(model, target_ds, val_ds, cal_ds, test_ds):
    _, _, target_cal_dl, target_test_dl = make_loaders(
        torch.utils.data.Subset(target_ds, range(4)),
        val_ds,
        cal_ds,
        test_ds,
    )
    return evaluate_model(model, target_cal_dl, target_test_dl, device=DEVICE)


def _fit_and_eval_nll(model, train_dl, val_dl, cal_dl, test_dl, fit_kwargs, source_state=None):
    history = []
    loggers = {
        "history": history,
        "rel_error_loader": val_dl,
        "every": max(1, fit_kwargs.get("epochs", 30) // 6),
    }
    if source_state is not None:
        loggers["reference_state"] = source_state

    fitted = fit_model(
        model,
        train_dl,
        val_dl,
        device=DEVICE,
        loggers=loggers,
        **fit_kwargs,
    )
    metrics = evaluate_model(fitted, cal_dl, test_dl, device=DEVICE)
    metrics["train_protocol"] = "nll"
    if source_state is not None:
        metrics["final_drift_from_source_all"] = compute_state_drift(fitted, source_state, include_heads=True)
        metrics["final_drift_from_source_backbone"] = compute_state_drift(fitted, source_state, include_heads=False)
    metrics["trajectory"] = history
    return metrics


def _fit_and_eval_mse(model, train_dl, val_dl, cal_dl, test_dl, epochs, lr, source_state=None):
    history = []
    loggers = {
        "history": history,
        "rel_error_loader": val_dl,
        "every": max(1, epochs // 6),
    }
    if source_state is not None:
        loggers["reference_state"] = source_state

    fitted = fit_model_mse(
        model,
        train_dl,
        val_dl,
        device=DEVICE,
        epochs=epochs,
        lr=lr,
        loggers=loggers,
    )
    metrics = evaluate_model(fitted, cal_dl, test_dl, device=DEVICE)
    metrics["train_protocol"] = "mse"
    if source_state is not None:
        metrics["final_drift_from_source_all"] = compute_state_drift(fitted, source_state, include_heads=True)
        metrics["final_drift_from_source_backbone"] = compute_state_drift(fitted, source_state, include_heads=False)
    metrics["trajectory"] = history
    return metrics


def _evaluate_revised_bundle(src_state, train_dl, val_dl, cal_dl, test_dl, target_epochs=30, model_size="small"):
    exp = {}

    exp["scratch"] = _fit_and_eval_nll(
        create_model(model_size=model_size),
        train_dl,
        val_dl,
        cal_dl,
        test_dl,
        fit_kwargs={"epochs": target_epochs, "muon_lr": 0.01, "adamw_lr": 1e-3},
    )

    exp["scratch_weak"] = _fit_and_eval_nll(
        create_model(model_size=model_size),
        train_dl,
        val_dl,
        cal_dl,
        test_dl,
        fit_kwargs={"epochs": 20, "muon_lr": 0.001, "adamw_lr": 1e-4},
    )

    exp["scratch_mse"] = _fit_and_eval_mse(
        create_model(model_size=model_size),
        train_dl,
        val_dl,
        cal_dl,
        test_dl,
        epochs=target_epochs,
        lr=3e-4,
    )

    weak_ft = create_model(model_size=model_size)
    weak_ft.load_state_dict(copy.deepcopy(src_state))
    reset_heads(weak_ft)
    exp["weak_finetune"] = _fit_and_eval_nll(
        weak_ft,
        train_dl,
        val_dl,
        cal_dl,
        test_dl,
        fit_kwargs={"epochs": 20, "muon_lr": 0.001, "adamw_lr": 1e-4},
        source_state=src_state,
    )

    transfer_mse = create_model(model_size=model_size)
    transfer_mse.load_state_dict(copy.deepcopy(src_state))
    reset_heads(transfer_mse)
    exp["transfer_mse"] = _fit_and_eval_mse(
        transfer_mse,
        train_dl,
        val_dl,
        cal_dl,
        test_dl,
        epochs=target_epochs,
        lr=1e-4,
        source_state=src_state,
    )

    frozen_long = create_model(model_size=model_size)
    frozen_long.load_state_dict(copy.deepcopy(src_state))
    reset_heads(frozen_long)
    set_trainable(frozen_long, "heads")
    exp["frozen_long"] = _fit_and_eval_nll(
        frozen_long,
        train_dl,
        val_dl,
        cal_dl,
        test_dl,
        fit_kwargs={"epochs": target_epochs, "muon_lr": 0.004, "adamw_lr": 4e-4},
        source_state=src_state,
    )

    strong_l2sp_model = create_model(model_size=model_size)
    strong_l2sp_model.load_state_dict(copy.deepcopy(src_state))
    reset_heads(strong_l2sp_model)
    strong_reg = L2SPRegularizer(src_state, strong_l2sp_model, alpha=5e-2, beta=1e-3)
    exp["strong_l2sp_alpha5e-2"] = _fit_and_eval_nll(
        strong_l2sp_model,
        train_dl,
        val_dl,
        cal_dl,
        test_dl,
        fit_kwargs={
            "epochs": target_epochs,
            "muon_lr": 0.002,
            "adamw_lr": 2e-4,
            "l2sp": strong_reg,
        },
        source_state=src_state,
    )

    last_block = create_model(model_size=model_size)
    last_block.load_state_dict(copy.deepcopy(src_state))
    reset_heads(last_block)
    set_trainable(last_block, "last")
    exp["last_block_only"] = _fit_and_eval_nll(
        last_block,
        train_dl,
        val_dl,
        cal_dl,
        test_dl,
        fit_kwargs={"epochs": target_epochs, "muon_lr": 0.002, "adamw_lr": 2e-4},
        source_state=src_state,
    )

    # Most likely to preserve transferable prior under limited budget.
    prog = progressive_unfreeze(src_state, train_dl, val_dl, device=DEVICE)
    prog_metrics = evaluate_model(prog, cal_dl, test_dl, device=DEVICE)
    prog_metrics["train_protocol"] = "nll_progressive"
    prog_metrics["final_drift_from_source_all"] = compute_state_drift(prog, src_state, include_heads=True)
    prog_metrics["final_drift_from_source_backbone"] = compute_state_drift(prog, src_state, include_heads=False)
    exp["progressive_unfreeze"] = prog_metrics

    return exp


def run_revised_diagnosis(seed=42, focus_shots=FOCUS_SHOTS, target_epoch_sweep=TARGET_EPOCH_SWEEP):
    np.random.seed(seed)
    torch.manual_seed(seed)

    target_ds = DomainShiftTurbulenceDataset(
        n_samples=TARGET_N_SAMPLES,
        grid_size=16,
        domain="target",
        rollout_steps=8,
    )
    splits, val_ds, cal_ds, test_ds = split_target_dataset(
        target_ds,
        shots=focus_shots,
        split_seed=seed,
        min_pool=max(focus_shots),
    )

    missing_shots = [n for n in focus_shots if str(n) not in splits]
    if missing_shots:
        raise ValueError(
            f"Requested shots {missing_shots} exceed available training pool. "
            f"Increase TARGET_N_SAMPLES or reduce fixed val/cal/test sizes."
        )

    results = {
        "seed": seed,
        "focus_shots": list(focus_shots),
        "target_epoch_sweep": list(target_epoch_sweep),
        "target_n_samples": TARGET_N_SAMPLES,
        "angle_a": {"zero_shot": {}, "few_shot": {}},
        "angle_b": {},
    }

    trained_sources = {}
    for source_variant in SOURCE_VARIANTS:
        _, src_model, src_state = _train_source_model(
            seed,
            source_domain=source_variant,
            use_multi_k_source=False,
            model_size="small",
        )
        trained_sources[source_variant] = {"model": src_model, "state": src_state}
        results["angle_a"]["zero_shot"][source_variant] = _evaluate_zero_shot(
            src_model,
            target_ds,
            val_ds,
            cal_ds,
            test_ds,
        )

    _, multi_k_model, _ = _train_source_model(seed, source_domain="source", use_multi_k_source=True, model_size="small")
    results["angle_a"]["zero_shot"]["source_multi_k"] = _evaluate_zero_shot(
        multi_k_model,
        target_ds,
        val_ds,
        cal_ds,
        test_ds,
    )

    for source_variant, payload in trained_sources.items():
        results["angle_a"]["few_shot"][source_variant] = {}
        for n in focus_shots:
            train_dl, val_dl, cal_dl, test_dl = make_loaders(splits[str(n)], val_ds, cal_ds, test_ds)
            results["angle_a"]["few_shot"][source_variant][str(n)] = _evaluate_revised_bundle(
                payload["state"],
                train_dl,
                val_dl,
                cal_dl,
                test_dl,
                target_epochs=30,
                model_size="small",
            )

    base_state = trained_sources["source_wider_plus"]["state"]
    for target_epochs in target_epoch_sweep:
        results["angle_b"][str(target_epochs)] = {}
        for n in focus_shots:
            train_dl, val_dl, cal_dl, test_dl = make_loaders(splits[str(n)], val_ds, cal_ds, test_ds)
            results["angle_b"][str(target_epochs)][str(n)] = _evaluate_revised_bundle(
                base_state,
                train_dl,
                val_dl,
                cal_dl,
                test_dl,
                target_epochs=target_epochs,
                model_size="small",
            )

    return results


def _summarize_metrics(metrics):
    return (
        f"rel={metrics['rel_error']:.4f} mse={metrics['test_mse']:.4f} "
        f"nll={metrics['test_nll']:.4f} sigma={metrics['test_mean_sigma']:.4f} "
        f"crps={metrics['test_crps_proxy']:.4f} width={metrics['interval_width']:.4f}"
    )


def interpret_revised_results(results):
    print("=" * 72)
    print("REVISED TRANSFERABILITY DIAGNOSIS REPORT")
    print("=" * 72)

    print("\n[Angle A] Source overlap sweep: zero-shot")
    for key, metrics in results["angle_a"]["zero_shot"].items():
        print(f"  {key:16s}: {_summarize_metrics(metrics)}")

    print("\n[Angle B] Revised controls (base source=source_wider_plus)")
    for epochs, shot_results in results["angle_b"].items():
        print(f"\n  target_epochs={epochs}")
        for n, metrics in shot_results.items():
            print(f"    n={n}")
            for key in [
                "scratch",
                "scratch_weak",
                "scratch_mse",
                "weak_finetune",
                "transfer_mse",
                "frozen_long",
                "strong_l2sp_alpha5e-2",
                "last_block_only",
                "progressive_unfreeze",
            ]:
                print(f"      {key:22s}: {_summarize_metrics(metrics[key])}")


if __name__ == "__main__":
    results = run_revised_diagnosis(seed=42, focus_shots=FOCUS_SHOTS, target_epoch_sweep=TARGET_EPOCH_SWEEP)
    interpret_revised_results(results)

    out_path = Path("transferability_revised_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")
