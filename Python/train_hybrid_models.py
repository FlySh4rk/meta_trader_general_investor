#!/usr/bin/env python3
"""
Train a hybrid MT5 gatekeeper model (RandomForest -> ONNX label-only output).

Expected files in project root (NOT_FOREX_TRADER):
  - Dataset.csv      : feature rows exported by EA
  - ReportTester.csv : MT5 report/deals used to derive labels
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


FEATURE_COLUMNS = [
    "SlopeNorm",
    "RSI",
    "Hour",
    "Day",
    "ATR_Norm",
    "DistEMA",
    "BB_Pos",
    "isTrend",
    "isLong",
]


def _read_csv_auto(path: Path) -> pd.DataFrame:
    """Read CSV trying common separators used by MT5 exports."""
    separators = [",", ";", "\t"]
    for sep in separators:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path)


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None


def _extract_profit_series(report_df: pd.DataFrame) -> pd.Series:
    candidate_names = [
        "Profit",
        "profit",
        "P/L",
        "PL",
        "Result",
        "NetProfit",
        "Net Profit",
    ]
    col = _find_column(report_df, candidate_names)
    if not col:
        raise ValueError(
            "Impossibile trovare la colonna profitto in ReportTester.csv. "
            f"Colonne disponibili: {list(report_df.columns)}"
        )

    numeric_profit = pd.to_numeric(report_df[col], errors="coerce").dropna()
    # Ignore pure zero rows (fees-only/headers/no-trade noise).
    numeric_profit = numeric_profit[numeric_profit != 0.0]
    if numeric_profit.empty:
        raise ValueError("Nessun trade valido trovato nel report MT5.")
    return numeric_profit.reset_index(drop=True)


def _build_training_frame(dataset_df: pd.DataFrame, deal_profits: pd.Series) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in dataset_df.columns]
    if missing:
        raise ValueError(
            "Dataset.csv non contiene tutte le feature richieste. Mancano: "
            + ", ".join(missing)
        )

    ds = dataset_df[FEATURE_COLUMNS].copy()
    n = min(len(ds), len(deal_profits))
    if n == 0:
        raise ValueError("Dataset o report vuoto dopo il preprocessing.")

    if len(ds) != len(deal_profits):
        print(
            f"[WARN] Righe non allineate (Dataset={len(ds)}, Deals={len(deal_profits)}). "
            f"Uso prime {n} righe per allineamento sequenziale."
        )

    ds = ds.iloc[:n].copy()
    ds["target"] = (deal_profits.iloc[:n] > 0.0).astype(np.int64).values
    return ds


def _build_sample_weights(train_df: pd.DataFrame) -> np.ndarray:
    """
    Equity-friendly bias:
    boost long pullbacks in trend when BB_Pos is near lower band.
    """
    w = np.ones(len(train_df), dtype=np.float32)
    mask_long_pullback = (
        (train_df["isTrend"] > 0.5)
        & (train_df["isLong"] > 0.5)
        & (train_df["BB_Pos"] <= 0.20)
    )
    w[mask_long_pullback.values] = 2.5
    return w


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "Dataset.csv"
    report_path = project_root / "ReportTester.csv"
    output_path = project_root / "hybrid_gatekeeper_rf.onnx"

    if not dataset_path.exists():
        raise FileNotFoundError(f"File non trovato: {dataset_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"File non trovato: {report_path}")

    dataset_df = _read_csv_auto(dataset_path)
    report_df = _read_csv_auto(report_path)
    profits = _extract_profit_series(report_df)
    train_df = _build_training_frame(dataset_df, profits)

    x = train_df[FEATURE_COLUMNS].astype(np.float32).values
    y = train_df["target"].astype(np.int64).values
    sample_weight = _build_sample_weights(train_df)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(x, y, sample_weight=sample_weight)

    # Critical for MT5 stability: fixed input shape [1, 9]
    initial_type = [("float_input", FloatTensorType([1, 9]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        options={id(model): {"zipmap": False}},
    )

    # "Nuclear option": remove probability tensor so MT5 reads a strict label output.
    if len(onnx_model.graph.output) > 1:
        onnx_model.graph.output.pop()

    with output_path.open("wb") as f:
        f.write(onnx_model.SerializeToString())

    positive_rate = float(np.mean(y))
    print("Training completato.")
    print(f"- Righe usate: {len(train_df)}")
    print(f"- Hit-rate positivo: {positive_rate:.2%}")
    print(f"- ONNX salvato in: {output_path}")


if __name__ == "__main__":
    main()
