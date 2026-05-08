from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from gee_data_loader import load_gee_training_csv
from lssvm_model import LSSVM
from main import (
    _get_config_path,
    _classification_metrics,
    _resolve_path,
    _stratified_train_val_split,
    _validate_training_data,
    load_runtime_config,
)


def train_lssvm() -> tuple[Path, Path, dict]:
    cfg = load_runtime_config()
    backend_dir = Path(__file__).parent

    data_dir = _resolve_path(backend_dir, cfg.gee_data_dir)
    training_csv = data_dir / "training_samples.csv"
    if not training_csv.is_file():
        raise FileNotFoundError(f"Training CSV not found: {training_csv}")

    model_path = _resolve_path(backend_dir, cfg.lssvm_model_path)
    if model_path.suffix.lower() != ".npz":
        model_path = model_path.with_suffix(".npz")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_gee_training_csv(str(training_csv))
    _validate_training_data(X_train, y_train)

    X_fit, y_fit, X_val, y_val = _stratified_train_val_split(
        X_train,
        y_train,
        val_ratio=0.2,
        seed=cfg.ca_seed,
    )

    t0 = time.time()
    model = LSSVM(gamma=cfg.lssvm_gamma, sigma=cfg.lssvm_sigma)
    model.fit(X_fit, y_fit)
    train_time = time.time() - t0

    y_fit_pred = model.predict(X_fit)
    y_fit_score = model.predict_proba(X_fit)
    m_train = _classification_metrics(y_fit, y_fit_pred, y_fit_score)

    if X_val is not None and y_val is not None and len(y_val) > 0:
        y_val_pred = model.predict(X_val)
        y_val_score = model.predict_proba(X_val)
        m_val = _classification_metrics(y_val, y_val_pred, y_val_score)
    else:
        m_val = None

    metrics = {
        "total_samples": int(len(y_train)),
        "train_samples": int(len(y_fit)),
        "val_samples": 0 if y_val is None else int(len(y_val)),
        "train_fire": int((y_fit == 1).sum()),
        "train_nofire": int((y_fit == -1).sum()),
        "val_fire": 0 if y_val is None else int((y_val == 1).sum()),
        "val_nofire": 0 if y_val is None else int((y_val == -1).sum()),
        "train_time_s": round(train_time, 3),
        "train_accuracy": round(m_train["accuracy"], 4),
        "fire_accuracy": round(m_train["recall"], 4),
        "nofire_accuracy": round(m_train["specificity"], 4),
        "train_precision": round(m_train["precision"], 4),
        "train_recall": round(m_train["recall"], 4),
        "train_specificity": round(m_train["specificity"], 4),
        "train_f1": round(m_train["f1"], 4),
        "train_balanced_accuracy": round(m_train["balanced_accuracy"], 4),
        "train_roc_auc": None if m_train["roc_auc"] is None else round(m_train["roc_auc"], 4),
        "val_accuracy": None if m_val is None else round(m_val["accuracy"], 4),
        "val_precision": None if m_val is None else round(m_val["precision"], 4),
        "val_recall": None if m_val is None else round(m_val["recall"], 4),
        "val_specificity": None if m_val is None else round(m_val["specificity"], 4),
        "val_f1": None if m_val is None else round(m_val["f1"], 4),
        "val_balanced_accuracy": None if m_val is None else round(m_val["balanced_accuracy"], 4),
        "val_roc_auc": None if m_val is None or m_val["roc_auc"] is None else round(m_val["roc_auc"], 4),
        "val_tp": None if m_val is None else int(m_val["tp"]),
        "val_tn": None if m_val is None else int(m_val["tn"]),
        "val_fp": None if m_val is None else int(m_val["fp"]),
        "val_fn": None if m_val is None else int(m_val["fn"]),
        "lssvm_b": round(float(model.b or 0.0), 4),
        "lssvm_gamma": float(cfg.lssvm_gamma),
        "lssvm_sigma": float(cfg.lssvm_sigma),
        "model_path": str(model_path),
        "config_path": str(_get_config_path()),
    }

    model.save(model_path)

    metrics_path = model_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return model_path, metrics_path, metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save the LSSVM model using simulation_config.json")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to simulation_config.json. If omitted, backend/simulation_config.json is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.config:
        os.environ["SIM_CONFIG_PATH"] = str(Path(args.config).resolve())

    model_path, metrics_path, metrics = train_lssvm()

    print("LSSVM training complete")
    print(f"Model saved:   {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print(
        "Summary: "
        f"train_acc={metrics['train_accuracy']:.4f}, "
        f"val_acc={metrics['val_accuracy'] if metrics['val_accuracy'] is not None else 'n/a'}, "
        f"samples={metrics['train_samples']}"
    )


if __name__ == "__main__":
    main()
