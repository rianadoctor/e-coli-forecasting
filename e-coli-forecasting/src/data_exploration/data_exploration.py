# GPT GENERATED
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_DIR = Path("data")
FIG_DIR = Path("figures")

def _find_date_column(columns: List[str]) -> Optional[str]:
    for c in columns:
        if "date" in c.lower():
            return c
    return None

def _find_ecoli_columns(columns: List[str]) -> List[str]:
    ecoli_cols = []
    for c in columns:
        lc = c.lower()
        if "ecoli" in lc:
            ecoli_cols.append(c)
    return ecoli_cols

def _read_csv(path: Path) -> pd.DataFrame:
    # Handle potential BOM with utf-8-sig
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df

def _ensure_datetime(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    if date_col and date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
    return df


def _get_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=["number"]).copy()


def _maybe_add_ecoli_linear(df: pd.DataFrame, ecoli_col: str) -> Tuple[pd.DataFrame, Optional[str]]:
    name_lower = ecoli_col.lower()
    if "log" in name_lower:
        linear_col = f"{ecoli_col}_linear"
        df = df.copy()
        # 10 ** log10 value to get CFU estimate
        df[linear_col] = (10 ** df[ecoli_col]).astype(float)
        return df, linear_col
    return df, None

def _safe_save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def visualize_csv(csv_path: Path) -> List[Path]:
    outputs: List[Path] = []
    df = _read_csv(csv_path)

    date_col = _find_date_column(df.columns.tolist())
    df = _ensure_datetime(df, date_col)

    ecoli_cols = _find_ecoli_columns(df.columns.tolist())
    target_col: Optional[str] = ecoli_cols[0] if ecoli_cols else None

    # If we have a log target, also add linear for visualization
    linear_col: Optional[str] = None
    if target_col:
        df, linear_col = _maybe_add_ecoli_linear(df, target_col)

    # Choose a column for target visualizations
    vis_target = linear_col or target_col

    # 1) Time series of target if date present
    if vis_target and date_col and df[date_col].notna().any():
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=df.sort_values(date_col), x=date_col, y=vis_target, ax=ax)
        ax.set_title(f"{csv_path.stem}: {vis_target} over time")
        ax.set_xlabel("Date")
        ax.set_ylabel(vis_target)
        out = FIG_DIR / csv_path.stem / f"timeseries_{vis_target}.png"
        _safe_save(fig, out)
        outputs.append(out)

    # 2) Distribution of target
    if vis_target:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[vis_target].dropna(), kde=True, ax=ax)
        ax.set_title(f"{csv_path.stem}: {vis_target} distribution")
        ax.set_xlabel(vis_target)
        out = FIG_DIR / csv_path.stem / f"hist_{vis_target}.png"
        _safe_save(fig, out)
        outputs.append(out)

    # 3) Correlation heatmap of numeric columns
    num = _get_numeric(df)
    if num.shape[1] >= 2:
        corr = num.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(min(10, 1 + 0.5 * corr.shape[1]), min(8, 1 + 0.5 * corr.shape[0])))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
        ax.set_title(f"{csv_path.stem}: numeric correlation heatmap")
        out = FIG_DIR / csv_path.stem / "corr_heatmap.png"
        _safe_save(fig, out)
        outputs.append(out)

    # 4) Top scatter plots: target vs top-k predictors by |corr|
    if vis_target and vis_target in num.columns:
        corrs = num.corr(numeric_only=True)[vis_target].drop(labels=[vis_target]).abs().sort_values(ascending=False)
        top_predictors = corrs.head(6).index.tolist()
        for pred in top_predictors:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.scatterplot(data=df, x=pred, y=vis_target, ax=ax)
            ax.set_title(f"{csv_path.stem}: {vis_target} vs {pred}")
            ax.set_xlabel(pred)
            ax.set_ylabel(vis_target)
            out = FIG_DIR / csv_path.stem / f"scatter_{vis_target}_vs_{pred}.png"
            _safe_save(fig, out)
            outputs.append(out)

    return outputs


def main() -> None:
    csvs = sorted([p for p in DATA_DIR.glob("*.csv") if p.is_file()])
    if not csvs:
        print(f"No CSV files found in {DATA_DIR}")
        return

    all_outputs: List[Path] = []
    for csv_path in csvs:
        print(f"Processing {csv_path}...")
        try:
            outs = visualize_csv(csv_path)
            all_outputs.extend(outs)
        except Exception as e:
            print(f"Failed to process {csv_path.name}: {e}")

    if all_outputs:
        print("Saved figures:")
        for p in all_outputs:
            print(f" - {p}")
    else:
        print("No figures were generated. Check data and column detection.")


if __name__ == "__main__":
    # Use a clean, readable style
    sns.set_theme(style="whitegrid")
    main()
