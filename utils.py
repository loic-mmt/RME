import numpy as np
import pandas as pd
import random
from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
from sklearn.preprocessing import StandardScaler


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


def train_val_test(df, split_index1: float = 0.7, split_index2: float = 0.85):
    n = len(df)
    i1 = int(n * split_index1)
    i2 = int(n * split_index2)

    train = df.iloc[:i1].copy()
    val = df.iloc[i1:i2].copy()
    test = df.iloc[i2:].copy()
    return train, val, test


def merge_dataframes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str],
    how: str = "inner",
    suffixes: tuple[str, str] = ("_x", "_y"),
    validate: str | None = None,
) -> pd.DataFrame:
    """Merge two DataFrames with basic input/column checks."""
    if left is None or left.empty:
        raise ValueError("`left` dataframe is empty.")
    if right is None or right.empty:
        raise ValueError("`right` dataframe is empty.")

    keys = [on] if isinstance(on, str) else list(on)
    if not keys:
        raise ValueError("`on` must contain at least one merge key.")

    missing_left = [col for col in keys if col not in left.columns]
    missing_right = [col for col in keys if col not in right.columns]
    if missing_left:
        raise ValueError(f"Missing merge keys in left dataframe: {missing_left}")
    if missing_right:
        raise ValueError(f"Missing merge keys in right dataframe: {missing_right}")

    return left.merge(
        right,
        on=keys,
        how=how,
        suffixes=suffixes,
        validate=validate,
    )

def standardise(df):
    scaler = StandardScaler()
    standardised = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return standardised


def compute_drawdown_series(equity_curve: pd.DataFrame) -> pd.DataFrame:
    eq = equity_curve['equity'].astype(float)
    rolling_peak = eq.cummax()
    drawdown = eq / rolling_peak - 1.0
    out = pd.DataFrame({'equity': eq, 'rolling_peak': rolling_peak, 'drawdown': drawdown}, index=equity_curve.index)
    return out



def compute_returns(price_series, cols = None):
    """Create log returns for OHLCV columns."""
    if price_series is None or price_series.empty:
        return pd.DataFrame()
    
    out = price_series.copy()
    missing = []
    if cols is None:
        cols = ["open", "high", "low", "close", "adj_close", "volume"]

    for col in cols:
        if col not in out.columns:
            missing.append(col)
            continue
        
        if col == "volume":
            out["volume_ret"] = np.log1p(out["volume"]) - np.log1p(out["volume"].shift(1))
        else:
            safe_prices = out[col].clip(lower=1e-12)
            out[f"{col}_ret"] = np.log(safe_prices / safe_prices.shift(1))

    if missing:
        print(f"Missing columns : {missing}")
    return out


def compute_trend_score(returns: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    if returns is None or returns.empty:
        return pd.DataFrame()
    score = returns.rolling(window).mean()
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



def compute_benchmark(df, capital):
    returns_df = compute_returns(df, cols=["adj_close"])
    if returns_df.empty or "adj_close_ret" not in returns_df.columns:
        return capital
    log_returns = returns_df["adj_close_ret"].dropna()
    return capital * np.exp(log_returns.sum())


def set_global_seed(seed: int = 1, deterministic_torch: bool = False) -> None:
    """Set reproducible seeds for Python, NumPy, and optionally Torch."""
    random.seed(seed)
    np.random.seed(seed)
    if deterministic_torch:
        try:
            import torch  # local import to keep utils lightweight if torch is absent
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def ensure_datetime_sorted(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Ensure date column is datetime and dataframe is sorted by date."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Missing `{date_col}` column.")
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return out


def add_row_and_split_columns(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    row_col: str = "_row",
    split_col: str = "_split",
) -> pd.DataFrame:
    """Add row index and chronological split labels: train/val/test."""
    if df is None or df.empty:
        raise ValueError("`df` is empty.")
    if not (0 < train_ratio < 1):
        raise ValueError("`train_ratio` must be in (0, 1).")
    if not (0 < val_ratio < 1):
        raise ValueError("`val_ratio` must be in (0, 1).")
    if train_ratio + val_ratio >= 1:
        raise ValueError("`train_ratio + val_ratio` must be < 1.")

    out = df.copy().reset_index(drop=True)
    n = len(out)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))

    out[row_col] = np.arange(n)
    out[split_col] = "test"
    out.loc[out[row_col] < i1, split_col] = "train"
    out.loc[(out[row_col] >= i1) & (out[row_col] < i2), split_col] = "val"
    return out


def build_forward_targets(
    df: pd.DataFrame,
    horizon: int = 5,
    price_col: str = "adj_close",
    vol_col: str = "vol_20",
    ret_target_col: str | None = None,
    z_target_col: str | None = None,
) -> pd.DataFrame:
    """Build forward return target and optional volatility-normalized target."""
    if df is None or df.empty:
        raise ValueError("`df` is empty.")
    if horizon < 1:
        raise ValueError("`horizon` must be >= 1.")
    if price_col not in df.columns:
        raise ValueError(f"Missing `{price_col}` column.")

    out = df.copy()
    ret_col = ret_target_col or f"target_ret_fwd_{horizon}"
    z_col = z_target_col or f"target_z_fwd_{horizon}"

    out[ret_col] = out[price_col].pct_change(horizon).shift(-horizon)
    if vol_col in out.columns:
        out[z_col] = out[ret_col] / (out[vol_col].astype(float) + 1e-8)
    return out


def clean_feature_target_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str | None = None,
    keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Coerce features/target to numeric and drop rows with NaNs on required columns."""
    if df is None or df.empty:
        raise ValueError("`df` is empty.")
    if not feature_cols:
        raise ValueError("`feature_cols` is empty.")

    required = [*feature_cols]
    if target_col is not None:
        required.append(target_col)
    if keep_cols:
        required = [*keep_cols, *required]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    out = df[required].copy()
    out[feature_cols] = out[feature_cols].apply(pd.to_numeric, errors="coerce")
    if target_col is not None:
        out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out = out.dropna(subset=[*feature_cols, *([target_col] if target_col else [])]).reset_index(drop=True)
    return out


def forecast_state_probs(
    current_state_probs: np.ndarray,
    transition_matrix: np.ndarray,
    horizon: int = 1,
) -> np.ndarray:
    """Forecast HMM state probabilities k-steps ahead: p_t @ A^k."""
    p = np.asarray(current_state_probs, dtype=float).reshape(-1)
    A = np.asarray(transition_matrix, dtype=float)

    if p.ndim != 1:
        raise ValueError("`current_state_probs` must be 1D.")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("`transition_matrix` must be square.")
    if A.shape[0] != p.shape[0]:
        raise ValueError("Mismatch between probs length and transition matrix size.")
    if horizon < 1:
        raise ValueError("`horizon` must be >= 1.")

    Ak = np.linalg.matrix_power(A, horizon)
    p_next = p @ Ak
    s = p_next.sum()
    if s <= 0:
        raise ValueError("Invalid forecast probabilities (sum <= 0).")
    return p_next / s


def expected_state_durations(transition_matrix: np.ndarray) -> np.ndarray:
    """Expected duration per state for Markov chain: 1 / (1 - A_ii)."""
    A = np.asarray(transition_matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("`transition_matrix` must be square.")
    diag = np.clip(np.diag(A), 0.0, 0.999999)
    return 1.0 / (1.0 - diag)
