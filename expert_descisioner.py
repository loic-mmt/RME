"""Orchestrator for Regime Mixture of Experts (RME).

Pipeline:
1) Ingest daily data for one ticker.
2) Build market features + forward targets.
3) Fit HMM on train split only.
4) Infer states/probabilities on train/val/test.
5) Fit one expert per state on train rows.
6) Predict on val/test with regime-based routing.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features import compute_market_features
from regime_detection import (
    HMM_FEATURES,
    fit_hmm_features,
    hmm_proba_from_model,
    hmm_states_from_model,
)
from utils import (
    add_row_and_split_columns,
    build_forward_targets,
    clean_feature_target_frame,
    ensure_datetime_sorted,
    forecast_state_probs,
    merge_dataframes,
    read_parquet_dataset,
    set_global_seed,
)


DATA_DIR = Path("data/cac40_daily.parquet")
EXPERT_MODULE_BY_STATE = {
    0: "regime4.model4",  # instable
    1: "regime1.model1",  # low vol / range
    2: "regime2.model2",  # high vol / stress
    3: "regime3.model3",  # trend fort
}
STATE_LABELS = {
    0: "unstable",
    1: "low_vol_range",
    2: "high_vol_stress",
    3: "trend",
}


@dataclass
class DecisionerConfig:
    data_dir: Path = DATA_DIR
    ticker: str = "EN.PA"
    seed: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    target_horizon: int = 5
    target_col: str = "target_z_fwd_5"
    state_forecast_horizon: int = 10
    hard_gating: bool = True
    threshold_prob_up: float = 0.70
    threshold_return: float = 0.0


class MeanFallbackExpert:
    """Fallback expert used when a regime model is not implemented yet."""

    def __init__(self, target_col: str):
        self.target_col = target_col
        self.mean_value = 0.0

    def fit(self, df: pd.DataFrame) -> "MeanFallbackExpert":
        if df is None or df.empty or self.target_col not in df.columns:
            self.mean_value = 0.0
            return self
        y = pd.to_numeric(df[self.target_col], errors="coerce").dropna()
        self.mean_value = float(y.mean()) if not y.empty else 0.0
        return self

    def predict(self, X: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.Series):
            n = 1
        elif isinstance(X, pd.DataFrame):
            n = len(X)
        else:
            arr = np.asarray(X)
            n = 1 if arr.ndim <= 1 else arr.shape[0]
        return np.full(shape=(n,), fill_value=self.mean_value, dtype=float)


def ingestion(data_dir: Path, ticker: str | None = None) -> pd.DataFrame:
    df = read_parquet_dataset(base_dir=data_dir)
    if df is None or df.empty:
        raise ValueError("Empty dataset.")
    df = ensure_datetime_sorted(df, date_col="date")
    if ticker is not None:
        df = df[df["ticker"] == ticker].copy()
        if df.empty:
            raise ValueError(f"No rows found for ticker={ticker}.")
    return df.reset_index(drop=True)


def build_feature_frame(df: pd.DataFrame, cfg: DecisionerConfig) -> pd.DataFrame:
    feat = compute_market_features(df).copy()
    if "date" not in feat.columns:
        raise ValueError("Missing `date` after feature engineering.")

    feat = ensure_datetime_sorted(feat, date_col="date")
    feat = add_row_and_split_columns(
        feat,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        row_col="_row",
        split_col="_split",
    )
    feat = build_forward_targets(
        feat,
        horizon=cfg.target_horizon,
        price_col="adj_close",
        vol_col="vol_20",
        ret_target_col=f"target_ret_fwd_{cfg.target_horizon}",
        z_target_col=f"target_z_fwd_{cfg.target_horizon}",
    )
    return feat


def select_hmm_features(df_feat: pd.DataFrame) -> list[str]:
    excluded = {"date", "adj_close"}
    cols = [c for c in HMM_FEATURES if c in df_feat.columns and c not in excluded]
    if not cols:
        raise ValueError("No HMM features available in dataframe.")
    return cols


def fit_and_label_hmm(df_feat: pd.DataFrame, hmm_cols: list[str]) -> tuple[Any, pd.DataFrame]:
    train_mask = df_feat["_split"] == "train"
    hmm_train = df_feat.loc[train_mask, hmm_cols]
    hmm_model = fit_hmm_features(hmm_train)

    hmm_full = df_feat.loc[:, hmm_cols]
    states = hmm_states_from_model(hmm_model, hmm_full)
    proba = hmm_proba_from_model(hmm_model, hmm_full)
    hmm_out = pd.concat([states, proba], axis=1).reset_index().rename(columns={"index": "_row"})

    merged = merge_dataframes(df_feat, hmm_out, on="_row", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    return hmm_model, merged


def split_clean_frames(
    merged: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> dict[str, pd.DataFrame]:
    split_frames: dict[str, pd.DataFrame] = {}
    keep_cols = ["date", "_row", "_split", "state"]

    for split_name in ("train", "val", "test"):
        frame = merged[merged["_split"] == split_name].copy()
        if frame.empty:
            split_frames[split_name] = frame
            continue
        split_frames[split_name] = clean_feature_target_frame(
            frame,
            feature_cols=feature_cols,
            target_col=target_col,
            keep_cols=keep_cols,
        )
    return split_frames


def _load_expert_module(module_path: str) -> Any | None:
    try:
        return importlib.import_module(module_path)
    except Exception:
        return None


def fit_state_experts(train_df: pd.DataFrame, target_col: str) -> dict[int, Any]:
    experts: dict[int, Any] = {}
    if train_df is None or train_df.empty:
        return experts

    for state in sorted(train_df["state"].dropna().astype(int).unique().tolist()):
        state_df = train_df[train_df["state"] == state].copy()
        module_path = EXPERT_MODULE_BY_STATE.get(state)
        module = _load_expert_module(module_path) if module_path else None

        # Expected future API in regime models: fit_expert(df, target_col=...)
        if module is not None and hasattr(module, "fit_expert"):
            expert = module.fit_expert(state_df, target_col=target_col)
        else:
            expert = MeanFallbackExpert(target_col=target_col).fit(state_df)

        experts[state] = expert
    return experts


def _predict_one(expert: Any, x_row: pd.Series) -> float:
    x_df = x_row.to_frame().T
    if hasattr(expert, "predict"):
        pred = expert.predict(x_df)
    elif callable(expert):
        pred = expert(x_df)
    else:
        raise ValueError("Expert has no usable predict interface.")

    arr = np.asarray(pred, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[0])


def predict_aggregation(
    experts: dict[int, Any],
    state_probs: np.ndarray,
    x_row: pd.Series,
    hard_gating: bool = True,
) -> tuple[float, int]:
    probs = np.asarray(state_probs, dtype=float).reshape(-1)
    if probs.size == 0:
        return 0.0, -1

    if hard_gating:
        state = int(np.argmax(probs))
        expert = experts.get(state)
        if expert is None:
            return 0.0, state
        return _predict_one(expert, x_row), state

    prediction = 0.0
    for state, p in enumerate(probs.tolist()):
        expert = experts.get(state)
        if expert is None:
            continue
        prediction += float(p) * _predict_one(expert, x_row)
    return float(prediction), int(np.argmax(probs))


def decision_from_prediction(
    pred_value: float,
    threshold_return: float = 0.0,
) -> str:
    if pred_value > threshold_return:
        return "buy"
    if pred_value < -threshold_return:
        return "sell"
    return "hold"


def forecast_state_path(
    hmm_model: Any,
    current_state_probs: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    if horizon < 1:
        raise ValueError("`horizon` must be >= 1.")
    transmat = np.asarray(hmm_model.transmat_, dtype=float)

    path = []
    probs = []
    p = np.asarray(current_state_probs, dtype=float).reshape(-1)
    for _ in range(horizon):
        p = forecast_state_probs(p, transition_matrix=transmat, horizon=1)
        probs.append(p)
        path.append(int(np.argmax(p)))
    return np.asarray(path, dtype=int), np.vstack(probs)


def train_pipeline(cfg: DecisionerConfig) -> dict[str, Any]:
    set_global_seed(cfg.seed)

    raw = ingestion(cfg.data_dir, ticker=cfg.ticker)
    feat = build_feature_frame(raw, cfg)
    hmm_cols = select_hmm_features(feat)
    hmm_model, merged = fit_and_label_hmm(feat, hmm_cols)

    proba_cols = [c for c in merged.columns if c.startswith("p_state_")]
    feature_cols = [*hmm_cols, *proba_cols]

    split_frames = split_clean_frames(merged, feature_cols=feature_cols, target_col=cfg.target_col)
    train_df = split_frames.get("train", pd.DataFrame())
    experts = fit_state_experts(train_df, target_col=cfg.target_col)

    return {
        "raw": raw,
        "features": feat,
        "merged": merged,
        "hmm_model": hmm_model,
        "hmm_cols": hmm_cols,
        "feature_cols": feature_cols,
        "proba_cols": proba_cols,
        "split_frames": split_frames,
        "experts": experts,
    }


def walk_forward(cfg: DecisionerConfig, on_split: str = "test", max_steps: int | None = None) -> pd.DataFrame:
    artifacts = train_pipeline(cfg)
    split_frames = artifacts["split_frames"]
    eval_df = split_frames.get(on_split, pd.DataFrame()).copy()
    if eval_df.empty:
        raise ValueError(f"Split `{on_split}` is empty.")

    if max_steps is not None:
        eval_df = eval_df.head(max_steps).copy()

    hmm_model = artifacts["hmm_model"]
    experts = artifacts["experts"]
    feature_cols: list[str] = artifacts["feature_cols"]
    proba_cols: list[str] = artifacts["proba_cols"]

    rows = []
    for _, row in eval_df.iterrows():
        current_probs = row[proba_cols].to_numpy(dtype=float)
        state_path, state_probs = forecast_state_path(
            hmm_model=hmm_model,
            current_state_probs=current_probs,
            horizon=cfg.state_forecast_horizon,
        )
        routed_probs = state_probs[-1]
        pred_value, routed_state = predict_aggregation(
            experts=experts,
            state_probs=routed_probs,
            x_row=row[feature_cols],
            hard_gating=cfg.hard_gating,
        )
        action = decision_from_prediction(pred_value, threshold_return=cfg.threshold_return)

        rows.append(
            {
                "date": row["date"],
                "_row": int(row["_row"]),
                "current_state": int(row["state"]),
                "routed_state": int(routed_state),
                "forecast_state_path": state_path.tolist(),
                "prediction": float(pred_value),
                "action": action,
                "target": float(row[cfg.target_col]),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    cfg = DecisionerConfig()
    signals = walk_forward(cfg, on_split="test", max_steps=50)
    print(signals.head(10))
    print("signals shape:", signals.shape)


if __name__ == "__main__":
    main()
