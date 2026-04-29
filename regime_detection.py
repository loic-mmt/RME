import numpy as np
import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data/cac40_daily.parquet"


def read_parquet_dataset(
    base_dir: Path,
    columns: list[str] | None = None,
    filter_expr: ds.Expression | None = None,
) -> pd.DataFrame:
    """Read a hive-partitioned parquet dataset into a DataFrame.

    Parameters
    ----------
    base_dir : Path
        Dataset root directory.
    columns : list[str] | None
        Optional list of columns to project.
    filter_expr : ds.Expression | None
        Optional Arrow dataset filter expression.

    Returns
    -------
    pd.DataFrame
        Materialized data as a pandas DataFrame.
    """
    dataset = ds.dataset(str(base_dir), format="parquet", partitioning="hive")
    if not dataset or dataset is None:
        raise ValueError("dataset empty")
    table = dataset.to_table(filter=filter_expr, columns=columns)
    return table.to_pandas()


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


def train_val_test(df, split_index1: int = 0.7, split_index2 : int = 0.85):
    split_index1 = len(df) * split_index1
    split_index2 = len(df) * split_index2
    train = df[:split_index1]
    val = df[split_index1:split_index2]
    test = df[split_index2:]
    return train, val, test


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