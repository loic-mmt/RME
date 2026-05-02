import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import merge_dataframes, read_parquet_dataset, train_val_test
from regime_detection import (
    HMM_FEATURES,
    fit_hmm_features,
    hmm_proba_from_model,
    hmm_states_from_model,
)
from features import compute_market_features

DATA_DIR = "data/cac40_daily.parquet"
TICKER = "EN.PA"

STATE_ID = 1
TARGET_HORIZON = 5
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3

import random
SEED = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_targets(df: pd.DataFrame, horizon: int = TARGET_HORIZON) -> pd.DataFrame:
    out = df.copy()
    out[f"target_ret_fwd_{horizon}"] = out["adj_close"].pct_change(horizon).shift(-horizon)
    out[f"target_z_fwd_{horizon}"] = out[f"target_ret_fwd_{horizon}"] / (out["vol_20"] + 1e-8)
    return out


def prepare_data(
    ticker: str = TICKER,
    state_id: int | None = STATE_ID,
    target_col: str = f"target_z_fwd_{TARGET_HORIZON}",
):
    prices = read_parquet_dataset(DATA_DIR)
    if "date" not in prices.columns:
        raise ValueError("Colonne 'date' manquante.")
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    df = prices.dropna(subset=["date"]).copy()
    series = df[df["ticker"] == ticker].copy()
    if series.empty:
        raise ValueError("Dataset vide apres filtrage.")

    series = series.sort_values("date").reset_index(drop=True)
    train_raw, val_raw, test_raw = train_val_test(series)
    i1 = len(train_raw)
    i2 = i1 + len(val_raw)

    df_feat = compute_market_features(series).sort_values("date").reset_index(drop=True).copy()
    df_feat["_row"] = np.arange(len(df_feat))
    df_feat["_split"] = "test"
    df_feat.loc[df_feat["_row"] < i1, "_split"] = "train"
    df_feat.loc[(df_feat["_row"] >= i1) & (df_feat["_row"] < i2), "_split"] = "val"

    hmm_cols = [c for c in HMM_FEATURES if c in df_feat.columns and c not in {"date", "adj_close"}]
    missing = [c for c in HMM_FEATURES if c not in df_feat.columns and c not in {"date"}]
    if missing:
        print("Colonnes HMM manquantes :", missing)
    if not hmm_cols:
        raise ValueError("Aucune colonne HMM disponible.")

    hmm_input = df_feat.loc[:, hmm_cols]
    hmm_train_input = hmm_input[df_feat["_split"] == "train"].copy()
    hmm_model = fit_hmm_features(hmm_train_input)

    states = hmm_states_from_model(hmm_model, hmm_input)
    proba = hmm_proba_from_model(hmm_model, hmm_input)
    outputs = pd.concat([states, proba], axis=1).reset_index().rename(columns={"index": "_row"})

    merged = merge_dataframes(df_feat, outputs, on="_row", how="inner")
    merged = _build_targets(merged, horizon=TARGET_HORIZON)
    merged = merged.sort_values("date").reset_index(drop=True)

    if state_id is not None:
        merged = merged[merged["state"] == state_id].copy()
    if merged.empty:
        raise ValueError(f"Aucune ligne pour state={state_id}.")

    proba_cols = [c for c in merged.columns if c.startswith("p_state_")]
    feature_cols = [*hmm_cols, *proba_cols]

    def _make_split_frame(split_name: str) -> pd.DataFrame:
        frame = merged[merged["_split"] == split_name].copy()
        if frame.empty:
            raise ValueError(f"Split `{split_name}` est vide après merge HMM.")
        frame = frame[["date", "_row", "_split", *feature_cols, target_col]].copy()
        frame[feature_cols] = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
        frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
        frame = frame.dropna().reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"Split `{split_name}` vide après dropna.")
        return frame

    train_df = _make_split_frame("train")
    val_df = _make_split_frame("val")
    test_df = _make_split_frame("test")

    def _to_tensor(frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        X = torch.tensor(frame[feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        y = torch.tensor(frame[target_col].to_numpy(dtype=np.float32).reshape(-1, 1), dtype=torch.float32)
        return X, y

    X_train, y_train = _to_tensor(train_df)
    X_val, y_val = _to_tensor(val_df)
    X_test, y_test = _to_tensor(test_df)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    split_frames = {"train": train_df, "val": val_df, "test": test_df}
    return train_loader, val_loader, test_loader, len(feature_cols), split_frames, merged


class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    in_dim: int,
    epochs: int = EPOCHS,
    lr: float = LR,
):
    model = MLP(in_dim=in_dim).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        val_loss = _evaluate(model, val_loader, criterion)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d} | train={train_loss:.6f} | val={val_loss:.6f}")

    return model


def predict(x_batch, model):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_batch)
    return y_pred


def main():
    train_loader, val_loader, test_loader, in_dim, split_frames, merged = prepare_data()
    model = train_model(train_loader, val_loader, in_dim=in_dim)
    final_val = _evaluate(model, val_loader, nn.MSELoss())
    final_test = _evaluate(model, test_loader, nn.MSELoss())
    print(f"Final val MSE: {final_val:.6f}")
    print(f"Final test MSE: {final_test:.6f}")
    print("Split shapes:", {k: v.shape for k, v in split_frames.items()})
    print("Dataset with regimes shape:", merged.shape)


if __name__ == "__main__":
    main()
