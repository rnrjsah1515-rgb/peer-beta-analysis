"""원시 베타 → Blume → 언레버 → Peer 중앙값 → 대상회사 재레버.

같은 계산을 (시장지수 × 측정조건) 모든 조합에 적용해 베타의 범위를 만든다.
엑셀 '베타산정' 시트는 기본 조합만 수식으로 다시 계산하고, 결과가 이 모듈과 같은지 대조한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import beta as B


def debt_of(cap: pd.DataFrame, include_lease: bool, basis: str) -> pd.Series:
    d = cap["borrowings"] + (cap["lease"] if include_lease else 0.0)
    if basis == "net":
        d = (d - cap["cash"]).clip(lower=0.0)
    return d


@dataclass
class Chain:
    peers: pd.DataFrame        # 회사별 raw → Blume → D/E → βu
    median_unlevered: float
    mean_unlevered: float
    target_de: float
    relevered: float
    relevered_hp: float        # Harris-Pringle 비교값


def run_chain(raw: pd.Series, de: pd.Series, tax: float, target_de: float | None = None) -> Chain:
    df = pd.DataFrame({"raw_beta": raw, "de": de})
    df["blume"] = df["raw_beta"].map(B.blume)
    df["unlevered"] = [B.unlever_hamada(b, d, tax) for b, d in zip(df["blume"], df["de"])]
    df["unlevered_hp"] = [B.unlever_harris_pringle(b, d) for b, d in zip(df["blume"], df["de"])]
    tde = float(df["de"].median()) if target_de is None else target_de
    bu = float(df["unlevered"].median())
    return Chain(df, bu, float(df["unlevered"].mean()), tde, B.relever_hamada(bu, tde, tax),
                 B.relever_harris_pringle(float(df["unlevered_hp"].median()), tde))


def range_table(grid: pd.DataFrame, de: pd.Series, tax: float, index_labels: dict[str, str]) -> pd.DataFrame:
    """(지수, 측정조건) 조합별 Peer 중앙값과 재레버 베타."""
    rows = []
    for (choice, setting), g in grid.groupby(["index_choice", "setting"], sort=False):
        g = g.set_index("code")
        c = run_chain(g["raw_beta"], de.loc[g.index], tax)
        rows.append({
            "index_choice": choice, "index_label": index_labels.get(choice, choice), "setting": setting,
            "median_raw": float(g["raw_beta"].median()), "median_r2": float(g["r2"].median()),
            "significant": int((g["ci_low"] > 0).sum()), "n_peers": len(g),
            "median_unlevered": c.median_unlevered, "relevered": c.relevered,
        })
    return pd.DataFrame(rows)
