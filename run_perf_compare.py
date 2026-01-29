#!/usr/bin/env python3
"""
Compares test metrics: baseline (no sample_weighting, no oversample) vs
mitigation (sample_weighting + oversample_pre_failure). Default: GNN only.
Set RUN_LSTM=1 to include LSTM.
"""
import copy
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from Libs.config import load_dataset_config, load_experiment_config
from Libs.scripts.train_dl_baseline import train_dl_baseline
from Libs.scripts.train_gnn import train_gnn

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def _patch(d: dict, path: list, val) -> None:
    cur = d
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = val


def _fmt(x, default="—"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    return f"{x:.4f}"


def main() -> None:
    ds = load_dataset_config()
    exp = load_experiment_config()

    baseline_ds = copy.deepcopy(ds)
    baseline_exp = copy.deepcopy(exp)
    new_ds = copy.deepcopy(ds)
    new_exp = copy.deepcopy(exp)

    _patch(baseline_ds, ["preprocessing", "data_split", "oversample_pre_failure_times"], 1)
    _patch(baseline_exp, ["training", "loss_function", "sample_weighting", "enable"], False)
    n_epochs = 15
    _patch(baseline_exp, ["training", "num_epochs"], n_epochs)
    _patch(baseline_exp, ["training", "batch_size"], 8)

    _patch(new_ds, ["preprocessing", "data_split", "oversample_pre_failure_times"], 2)
    _patch(new_exp, ["training", "loss_function", "sample_weighting", "enable"], True)
    _patch(new_exp, ["training", "loss_function", "sample_weighting", "positive_weight"], 2.0)
    _patch(new_exp, ["training", "num_epochs"], n_epochs)
    _patch(new_exp, ["training", "batch_size"], 8)

    run_lstm = os.environ.get("RUN_LSTM", "").lower() in ("1", "true", "yes")

    print("=" * 72)
    print("Performance: baseline vs mitigation (sample_weighting + oversample)")
    print("  Baseline: sample_weighting=off, oversample_pre_failure_times=1")
    print("  New:      sample_weighting=on (positive_weight=2.0), oversample=2")
    print(f"  Common:   num_epochs={n_epochs}, batch_size=8, train_pre_only=false")
    print("=" * 72)

    def row(name: str, m: dict) -> list:
        return [
            name,
            _fmt(m.get("test_rmse")),
            _fmt(m.get("test_mae")),
            _fmt(m.get("test_r2")),
            _fmt(m.get("test_r2_rul_positive")),
        ]

    print("\n[1/2] GNN baseline...")
    _, m_gnn_b = train_gnn(dataset_config=baseline_ds, experiment_config=baseline_exp)
    print("[2/2] GNN mitigation...")
    _, m_gnn_n = train_gnn(dataset_config=new_ds, experiment_config=new_exp)

    rows = [row("GNN baseline", m_gnn_b), row("GNN mitigation", m_gnn_n)]
    if run_lstm:
        print("[3/4] LSTM baseline...")
        _, m_lstm_b = train_dl_baseline("lstm", dataset_config=baseline_ds, experiment_config=baseline_exp)
        print("[4/4] LSTM mitigation...")
        _, m_lstm_n = train_dl_baseline("lstm", dataset_config=new_ds, experiment_config=new_exp)
        rows.extend([row("LSTM baseline", m_lstm_b), row("LSTM mitigation", m_lstm_n)])

    h = ["Config", "Test RMSE", "Test MAE", "Test R²", "Test R² (RUL>0)"]
    w = [max(len(h[i]), max(len(r[i]) for r in rows)) for i in range(len(h))]
    fmt = "  ".join(f"{{:{w[i]}}}" for i in range(len(h)))
    print()
    print(fmt.format(*h))
    print("-" * (sum(w) + 2 * (len(w) - 1)))
    for r in rows:
        print(fmt.format(*r))

    def delta_rmse(a, b):
        va, vb = (a or np.nan), (b or np.nan)
        if np.isnan(va) or np.isnan(vb) or va <= 0:
            return np.nan
        return (vb - va) / va * 100  # positive = worse

    d_rmse = delta_rmse(m_gnn_b.get("test_rmse"), m_gnn_n.get("test_rmse"))

    print()
    print("GNN (mitigation vs baseline):")
    print(f"  Test RMSE: {_fmt(m_gnn_b.get('test_rmse'))} -> {_fmt(m_gnn_n.get('test_rmse'))}  (delta {d_rmse:+.1f}%, + = mitigation worse)")
    print(f"  Test R²:   {_fmt(m_gnn_b.get('test_r2'))} -> {_fmt(m_gnn_n.get('test_r2'))}")
    print(f"  Test R² (RUL>0): {_fmt(m_gnn_b.get('test_r2_rul_positive'))} -> {_fmt(m_gnn_n.get('test_r2_rul_positive'))}")
    if run_lstm:
        d_rmse_l = delta_rmse(m_lstm_b.get("test_rmse"), m_lstm_n.get("test_rmse"))
        print(f"  LSTM Test RMSE: {_fmt(m_lstm_b.get('test_rmse'))} -> {_fmt(m_lstm_n.get('test_rmse'))}  (delta {d_rmse_l:+.1f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
