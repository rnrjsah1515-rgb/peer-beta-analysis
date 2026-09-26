"""베타 추정의 정밀도.

R² 가 낮다는 사실 자체는 CAPM 을 부정하지 않는다(설명되지 않는 부분은 분산 가능한 개별위험).
판단에 필요한 것은 기울기가 얼마나 정확히 측정됐는가, 즉 표준오차다.

Peer 를 동일가중 포트폴리오로 묶어 회귀하면 개별 잡음이 상쇄된다. 다만 잔차가 서로
상관되어 있으면(업종 공통요인) 상쇄 효과가 √N 에 못 미친다. 그 차이를 함께 보고한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import beta as B


@dataclass
class Precision:
    portfolio: B.OLS                 # Peer 동일가중 포트폴리오 회귀
    mean_single_se: float            # 개별 Peer 표준오차 평균
    se_if_independent: float         # 잔차가 독립이라면 기대되는 평균 베타의 표준오차
    mean_residual_corr: float        # Peer 잔차 간 평균 상관계수
    beta_dispersion: float           # 개별 베타의 표준편차 (회사 간 실제 차이)
    blume_low: float                 # 포트폴리오 베타 95% 구간에 Blume 조정을 적용한 값
    blume_high: float
    portfolio_de: float = 0.0        # 포트폴리오(동일가중)의 D/E — 언레버 기준


def analyze(returns_by_code: dict[str, pd.DataFrame], de: pd.Series | None = None) -> Precision:
    """returns_by_code: 코드 → DataFrame[s, m] (같은 지수·같은 측정조건). de: 코드별 D/E."""
    codes = list(returns_by_code)
    fits = {c: B.ols(returns_by_code[c]) for c in codes}
    market = returns_by_code[codes[0]]["m"]
    port = pd.DataFrame({
        "s": pd.DataFrame({c: returns_by_code[c]["s"] for c in codes}).mean(axis=1),
        "m": market,
    }).dropna()
    o = B.ols(port)

    resid = pd.DataFrame({
        c: returns_by_code[c]["s"] - fits[c].alpha - fits[c].beta * returns_by_code[c]["m"] for c in codes
    }).dropna()
    corr = resid.corr().to_numpy()
    n = len(codes)
    mean_corr = float((corr.sum() - n) / (n * (n - 1)))
    mean_se = float(np.mean([f.se for f in fits.values()]))
    return Precision(
        portfolio=o, mean_single_se=mean_se, se_if_independent=mean_se / np.sqrt(n),
        mean_residual_corr=mean_corr, beta_dispersion=float(np.std([f.beta for f in fits.values()], ddof=1)),
        blume_low=B.blume(o.ci_low), blume_high=B.blume(o.ci_high),
        portfolio_de=float(de.reindex(codes).mean()) if de is not None else 0.0,
    )


def relevered_band(p: Precision, target_de: float, tax: float) -> tuple[float, float]:
    """포트폴리오 베타 구간을 포트폴리오 D/E 로 언레버해 목표 D/E 로 재레버한 구간."""
    return tuple(
        B.relever_hamada(B.unlever_hamada(b, p.portfolio_de, tax), target_de, tax)
        for b in (p.blume_low, p.blume_high)
    )
