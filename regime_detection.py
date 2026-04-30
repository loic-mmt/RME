import numpy as np
import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

from features import features, compute_market_features
from utils import read_parquet_dataset, train_val_test


DATA_DIR = "data/cac40_daily.parquet"
SEED = 1
np.random.seed(1)

HMM_FEATURES = [
    "date",
    "adj_close",
    "ret_2",
    "ret_5",
    "ret_20",
    "ret_60",
    "dist_ma_20",
    "ma_ratio_20_60",
    "bollinger_z_20",
    "drawdown_20",
    "drawdown_60",
    "vol_20",
    "vol_60",
    "vol_ratio_5_20",
    "atr_norm",
    "amihud_illiq",
    "ret_x_illiq",
    "autocorr_20",
    "variance_ratio_20",
    "market_ret_5",
    "market_ret_20",
    "market_vol_20",
    "market_drawdown_20",
    "beta_market_60",
    "corr_market_60",
    "vix_level",
    "market_breadth",
]


def standardise(df):
    scaler = StandardScaler()
    standardised = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return standardised


def compute_returns(price_series, cols = None):
    """Create log returns for OHLCV columns."""
    if price_series is None or price_series.empty:
        return pd.DataFrame()
    
    out = price_series.copy()
    if cols is None:
        cols = ["open", "high", "low", "close", "adj_close", "volume"]
        missing = []

        for col in cols :
            if col not in out.columns:
                missing.append(col)
                continue
            
            if col == "volume":
                out["volume_ret"] = np.log1p(out["volume"]) - np.log1p(out["volume"].shift(1))
            else :
                safe_prices = out[col].clip(lower= 1e-12)
                out[f"{col}_ret"] = np.log(safe_prices / safe_prices.shift(1))

    print(f"Missing columns : {missing}")
    return out


def compute_drawdown_series(equity_curve: pd.DataFrame) -> pd.DataFrame:
    eq = equity_curve['equity'].astype(float)
    rolling_peak = eq.cummax()
    drawdown = eq / rolling_peak - 1.0
    out = pd.DataFrame({'equity': eq, 'rolling_peak': rolling_peak, 'drawdown': drawdown}, index=equity_curve.index)
    return out


def compute_benchmark(df, capital):
    returns_df = compute_returns(df, cols=["adj_close"])
    if returns_df.empty or "adj_close_ret" not in returns_df.columns:
        return capital
    log_returns = returns_df["adj_close_ret"].dropna()
    return capital * np.exp(log_returns.sum())


def compute_trend_score(returns: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    if returns is None or returns.empty:
        return pd.DataFrame()
    score = sum(returns.rolling(window)) / window
    return score


def compute_realised_vol(returns : pd.DataFrame, window : int = 10) -> pd.DataFrame:
    if returns is None or returns.empty:
        return pd.DataFrame()
    score = np.std(returns.rolling(window)) * np.sqrt(window)
    return score


def compute_vol_of_vol(vol : pd.DataFrame, window : int = 10) -> pd.DataFrame:
    if vol is None or vol.empty:
        return pd.DataFrame()
    score = np.std(vol.rolling(window)) * np.sqrt(window)
    return score


def compute_correlation(returns: pd.DataFrame):
    if returns is None or returns.empty:
        return pd.DataFrame()
    df = returns.groupby("ticker")
    correlation = np.corrcoef(df)
    return correlation


def fit_hmm_features(X: pd.DataFrame, n_states: int = 4, covariance_type: str = "diag"):
    """Fit a Gaussian HMM on multivariate features."""
    if X is None or X.empty:
        raise ValueError("X is empty.")
    X = X.apply(pd.to_numeric, errors="coerce").dropna()
    if X.empty:
        raise ValueError("X has only NaNs after dropna.")
    
    model = GaussianHMM(n_components= n_states, covariance_type= covariance_type, n_iter=200, random_state=SEED)
    model.fit(X.to_numpy())
    return model


def hmm_states_from_model(model: GaussianHMM, X: pd.DataFrame)-> pd.Series:
    """Return most likely state for each observation."""
    if X is None or X.empty:
        raise ValueError("X is empty.")
    X = X.apply(pd.to_numeric, errors="coerce").dropna()
    if X.empty:
        raise ValueError("X has only NaNs after dropna.")
    
    states = model.predict(X.to_numpy())
    return pd.Series(states, index = X.index, name = "state")


def hmm_proba_from_model(model: GaussianHMM, X: pd.DataFrame) -> pd.DataFrame:
    """Return state probabilities for each observation."""
    if X is None or X.empty:
        raise ValueError("X is empty.")
    X = X.apply(pd.to_numeric, errors="coerce").dropna()
    if X.empty:
        raise ValueError("X has only NaNs after dropna.")
    
    proba = model.predict_proba(X.to_numpy())
    cols = [f"p_state_{i}" for i in range(proba.shape[1])]
    return pd.DataFrame(proba, index= X.index, columns= cols)


def fit_regime_model(df_z: pd.DataFrame, mkt_returns: pd.Series | None = None):
    """Fit the HMM model for regimes."""
    hmm_features = fit_hmm_features(df_z)
    return hmm_features


def build_regime_outputs(model, df_z: pd.DataFrame) -> pd.DataFrame:
    """Build regime states and probabilities output frame."""
    states = hmm_states_from_model(model, df_z)
    proba = hmm_proba_from_model(model, df_z)
    return pd.concat([states, proba], axis=1)


def regime_pipeline(df):
    hmm = fit_regime_model(df)
    outputs = build_regime_outputs(hmm, df)
    outputs = outputs.reset_index().rename(columns={"index": "_row"})
    proba_cols = [c for c in outputs.columns if c.startswith("p_state_")]

    if "date" in df.columns:
        outputs["date"] = pd.to_datetime(df.loc[outputs["_row"], "date"].to_numpy(), errors="coerce")
        outputs = outputs[["date", "_row", "state", *proba_cols]]
    else:
        outputs = outputs[["_row", "state", *proba_cols]]

    print("outputs shape:", outputs.shape)
    print(outputs.head(3))
    print(outputs.dtypes)
    return outputs



def plot_regimes(
    returns: pd.DataFrame,
    states: pd.DataFrame | pd.Series,
    return_col: str = "ret_5",
    price_col: str = "adj_close",
):
    """Plot returns, state probabilities, and price, all aligned with HMM states."""
    if returns is None or len(returns) == 0:
        raise ValueError("`returns` is empty.")
    if states is None or len(states) == 0:
        raise ValueError("`states` is empty.")

    base = returns.copy().reset_index(drop=True)
    base["_row"] = np.arange(len(base))

    if return_col not in base.columns:
        raise ValueError(f"Missing required return column: `{return_col}`.")
    if price_col not in base.columns:
        raise ValueError(f"Missing required price column: `{price_col}`.")

    if isinstance(states, pd.Series):
        regimes = states.to_frame(name="state").reset_index().rename(columns={"index": "_row"})
    else:
        regimes = states.copy()
        if "state" not in regimes.columns:
            if regimes.shape[1] == 1:
                regimes = regimes.rename(columns={regimes.columns[0]: "state"})
            else:
                raise ValueError("`states` must contain a `state` column.")
        if "_row" not in regimes.columns:
            if "date" in regimes.columns and np.issubdtype(regimes["date"].dtype, np.number):
                regimes = regimes.rename(columns={"date": "_row"})
            else:
                regimes = regimes.reset_index().rename(columns={"index": "_row"})

    proba_cols = [c for c in regimes.columns if c.startswith("p_state_")]
    merged = base.merge(regimes[["_row", "state", *proba_cols]], on="_row", how="left")
    merged["state"] = pd.to_numeric(merged["state"], errors="coerce")
    merged["state_int"] = merged["state"].round().astype("Int64")

    x_col = "_row"
    if "date" in merged.columns:
        dates = pd.to_datetime(merged["date"], errors="coerce")
        if dates.notna().any():
            merged["date"] = dates
            x_col = "date"

    unique_states = sorted(merged["state_int"].dropna().unique().tolist())
    cmap = plt.get_cmap("tab10", max(len(unique_states), 1))
    state_colors = {state: cmap(idx) for idx, state in enumerate(unique_states)}

    def _plot_state_overlay(ax, y_col: str, title: str):
        ax.plot(merged[x_col], merged[y_col], color="lightgray", lw=1.0, alpha=0.9)
        valid = merged.dropna(subset=["state_int", y_col]).copy()
        if not valid.empty:
            valid = valid.sort_values(x_col).reset_index(drop=True)
            state_values = valid["state_int"].to_numpy()
            start = 0
            for i in range(1, len(valid) + 1):
                if i == len(valid) or state_values[i] != state_values[i - 1]:
                    seg = valid.iloc[start:i]
                    s = int(state_values[i - 1])
                    ax.plot(seg[x_col], seg[y_col], color=state_colors[s], lw=2.0)
                    start = i
            for state in unique_states:
                ax.plot([], [], color=state_colors[state], lw=2.0, label=f"state {state}")
            ax.legend(loc="upper left", ncol=3, fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(y_col)
        ax.grid(alpha=0.2)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.0, 1.6]},
    )

    # 1) Returns with state overlay
    _plot_state_overlay(axes[0], return_col, "Returns Colored By State")

    # 2) State probabilities
    if proba_cols:
        for col in proba_cols:
            axes[1].plot(merged[x_col], merged[col], lw=1.0, label=col)
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].set_ylabel("Probability")
        axes[1].legend(loc="upper left", ncol=4, fontsize=8)
    else:
        axes[1].step(merged[x_col], merged["state"], where="mid", lw=1.0, color="tab:blue", label="state")
        axes[1].set_ylabel("State")
        axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(alpha=0.2)

    # 3) Price with state overlay
    _plot_state_overlay(axes[2], price_col, "Price Colored By State")
    axes[2].set_xlabel("Date" if x_col == "date" else "Observation")

    plt.tight_layout()
    plt.show()




def main(ticker):
    df = read_parquet_dataset(DATA_DIR)
    if "date" not in df.columns:
        raise ValueError("Colonne 'date' manquante.")
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    series = df[df["ticker"] == ticker].copy()

    if series.empty:
        raise ValueError("Dataset vide apres filtrage.")
    train, val , test = train_val_test(series)
    df_feat = compute_market_features(train)

    cols = [c for c in HMM_FEATURES if c in df_feat.columns]
    missing = [c for c in HMM_FEATURES if c not in df_feat.columns]

    if missing:
        print("Colonnes manquantes :", missing)

    out = df_feat.loc[:, cols]
    print(out.tail())
    outputs = regime_pipeline(out)

    plot_regimes(out, outputs)
    

if __name__ == "__main__":
    main("EN.PA")
