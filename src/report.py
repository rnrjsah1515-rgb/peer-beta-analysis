"""실행 결과를 output/summary.md 로 기록한다 (README 가 인용하는 수치의 출처)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pipeline import INDEX_CHOICES

EOK = 1e8


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write(path: Path, cfg: dict, meas, cap: pd.DataFrame, chain, ranges: pd.DataFrame, excel_result: str) -> None:
    b = meas.base
    p = chain.peers
    peers = pd.DataFrame({
        "회사": b["name"], "시장": b["listed"],
        "원시 β": p["raw_beta"].map("{:.2f}".format), "95% 구간": [f"{l:.2f} ~ {h:.2f}" for l, h in zip(b["ci_low"], b["ci_high"])],
        "R²": b["r2"].map("{:.1%}".format), "Dimson β": b["dimson_beta"].map("{:.2f}".format),
        "무거래일": b["zero_ret"].map("{:.1%}".format), "Blume β": p["blume"].map("{:.2f}".format),
        "D (억원)": (cap["debt"] / EOK).map("{:,.0f}".format), "E (억원)": (cap["market_cap"] / EOK).map("{:,.0f}".format),
        "D/E": cap["de"].map("{:.1%}".format), "βu": p["unlevered"].map("{:.2f}".format),
    })
    rng = ranges.pivot(index="index_label", columns="setting", values="relevered").map("{:.2f}".format)
    sig = ranges.pivot(index="index_label", columns="setting", values="median_r2").map("{:.1%}".format)
    w = meas.semi_weight
    text = f"""# 실행 결과 요약

- 대상: {cfg['case']['target_name']} · 평가기준일 {meas.valuation_date.date()}
- 채택 측정: {INDEX_CHOICES[cfg['measurement']['market_index']]} 대비 {cfg['measurement']['base']} · 법인세율 {cfg['capital']['tax_rate']:.1%}
- 엑셀 대조: {excel_result}

## 채택 조건 결과

{_md(peers.reset_index(drop=True))}

- 무부채 베타 중앙값 **{chain.median_unlevered:.3f}** (평균 {chain.mean_unlevered:.3f})
- 목표 D/E (Peer 중앙값) **{chain.target_de:.1%}**
- 재레버 베타 **{chain.relevered:.3f}** (Harris-Pringle 비교 {chain.relevered_hp:.3f})

## 시장지수 × 측정조건별 재레버 베타

{_md(rng.reset_index().rename(columns={'index_label': '시장지수'}))}

R² 중앙값

{_md(sig.reset_index().rename(columns={'index_label': '시장지수'}))}

## KOSPI 내 반도체 3종목 비중

2년 전 {w[w.index <= meas.valuation_date - pd.DateOffset(years=2)].iloc[-1]:.1%} → 1년 전 {w[w.index <= meas.valuation_date - pd.DateOffset(years=1)].iloc[-1]:.1%} → 평가기준일 {w.iloc[-1]:.1%}

## 데이터 대조

{_md(meas.data_check.fillna(''))}

{chr(10).join('- ' + n for n in meas.notes)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
