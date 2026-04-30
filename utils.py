import numpy as np
import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds


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
