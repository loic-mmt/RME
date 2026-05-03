import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

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

from expert_descisioner import DecisionerConfig, DATA_DIR


ClassifConfig = {
    "ticker": DecisionerConfig.ticker,
    "sequence_length": 60,      # nombre de jours passés utilisés
    "prediction_horizon": 1,    # prédire J+1
    "batch_size": 64,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 12,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "model_path": "regime1/classif_lstm_model_1.pt",
    "seed": DecisionerConfig.seed
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


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
    # Target : direction du rendement futur
    future_return = feat["adj_close"].shift(-ClassifConfig["prediction_horizon"]) / feat["adj_close"] - 1
    feat["target"] = (future_return > 0).astype(int)
    feat.dropna(inplace=True)

    return feat


# Dataset fenêtre glissante
class SequenceDataset(Dataset): 
    def __init__(self, X, y, sequence_lenght):
        self.X = X
        self.y = y
        self.sequence_lenght = sequence_lenght

    def __len__(self):
        return len(self.X) - self.sequence_lenght + 1
    
    def __getitem__(self, index):
        x_seq = self.X[index:index + self.sequence_lenght]
        y_label = self.y[index + self.sequence_lenght - 1]

        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(y_label, dtype=torch.long)
        )


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size = 128,
        num_layers = 2,
        dropout = 0.3,
        output_size = 2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch, sequence_lenght, input_size)
        lstm_output, (h_n, c_n) = self.lstm(x)

        # Dernière sortie temporelle
        last_output = lstm_output[:, -1, :]

        # Stabilisation
        last_output = self.layer_norm(last_output)

        logits = self.classifier(last_output)

        return logits
    


def prepare_data(df, feature_cols, target_col= "target"):
    n = len(df)

    train_end = int(n * ClassifConfig["train_ratio"])
    val_end = int(n * ClassifConfig["train_ratio"]) + ClassifConfig["val_ratio"]

    train_df = df.iloc[: train_end].copy()
    val_df = df.iloc[train_end: val_end].copy()
    test_df = df.iloc[val_end:].copy()

    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val = scaler.transform(val_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    train_dataset = SequenceDataset(X_train, y_train, ClassifConfig["sequence_length"])
    val_dataset = SequenceDataset(X_val, y_val, ClassifConfig["sequence_length"])
    test_dataset = SequenceDataset(X_test, y_test, ClassifConfig["sequence_length"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=ClassifConfig["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=ClassifConfig["batch_size"],
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size= ClassifConfig["batch_size"],
        shuffle=False
    )

    return train_loader, val_loader, test_loader, scaler, train_df, val_df, test_df


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0
    all_preds = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(y_batch.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * X_batch.size(0)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)

    return avg_loss, acc,  np.array(all_preds), np.array(all_targets), np.array(all_probs)


def train_model(model, train_loader, val_loader):
    y_train_all = []

    for _, y_batch in train_loader:
        y_train_all.extend(y_batch.numpy())

    y_train_all = np.array(y_train_all)
    class_counts = np.bincount(y_train_all)
    class_weights = len(y_train_all) / (2 * class_counts)

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = ClassifConfig["learning_rate"],
        weight_decay=ClassifConfig["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(ClassifConfig["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc, _, _, _ = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "config": ClassifConfig
            }, ClassifConfig["model_path"])
        
        else:
            patience_counter += 1

        if patience_counter >= ClassifConfig["patience"]:
            print("Early stopping déclanché.")
            break

    print("Meilleur modèle sauvegardé :", ClassifConfig["model_path"])


def predict_latest(model, df, feature_cols, scaler):
    model.eval()

    latest_data = df[feature_cols].iloc[-ClassifConfig["sequence_length"]:]
    latest_scaled = scaler.transform(latest_data)

    x = torch.tensor(latest_scaled, dtype=torch.float32)
    x = x.unsqueeze(0) #shape: (1, sequence_length, input_size)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        prob_down = probs[0, 0].item()
        prob_up = probs[0, 1].item()
        prediction = torch.argmax(probs, dim=1).item()

    result = {
        "prediction": "UP" if prediction == 1 else "DOWN",
        "probability_up": prob_up,
        "probability_down": prob_down
    }
    return result


def main(cfg: DecisionerConfig, ):
    set_global_seed(cfg.seed)

    print("Récupération des données...")
    df = ingestion(DATA_DIR, "EN.PA")

    print("Création des features...")
    df = build_feature_frame(df)

    feature_cols = [

    ]

    print("Nombre d'observations :", len(df))
    print("Nombre de features :", len(feature_cols))

    train_loader, val_loader, test_loader, scaler, train_df, val_df, test_df = prepare_data(
        df,
        feature_cols
    )

    input_size = len(feature_cols)

    model = LSTM(
        input_size=input_size,
        hidden_size=ClassifConfig["hidden_size"],
        num_layers=ClassifConfig["num_layers"],
        dropout=ClassifConfig["dropout"],
        output_size=2
    ).to(device)

    print(model)

    print("Entraînement du modèle...")
    train_model(model, train_loader, val_loader)

    print("Chargement du meilleur modèle...")
    checkpoint = torch.load(ClassifConfig["model_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    print("Évaluation sur le test set...")

    test_loss, test_acc, test_preds, test_targets, test_probs = evaluate(
        model,
        test_loader,
        criterion
    )

    print("\nRésultats test :")
    print("Test Loss:", round(test_loss, 4))
    print("Test Accuracy:", round(test_acc, 4))
    print("Precision:", round(precision_score(test_targets, test_preds), 4))
    print("Recall:", round(recall_score(test_targets, test_preds), 4))
    print("F1 Score:", round(f1_score(test_targets, test_preds), 4))
    print("\nClassification report :")
    print(classification_report(test_targets, test_preds, target_names=["DOWN", "UP"]))
    print("Prédiction sur les dernières données...")
    latest_prediction = predict_latest(
        model,
        df,
        feature_cols,
        scaler
    )

    print("\nDernière prédiction :")
    print(latest_prediction)

if __name__ == "__main__":

    main()