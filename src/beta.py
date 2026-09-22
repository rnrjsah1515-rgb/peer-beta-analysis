"""베타 추정과 조정.

측정: 단순수익률 OLS (Bloomberg 방식과 같은 가격수익률 기준, 배당 미포함)
보정: Dimson (전·후 1기 시장수익률 포함) — 거래가 뜸한 소형주의 베타 과소추정 점검용
조정: Blume (0.67 × 원시 + 0.33)
레버리지: Hamada (부채 베타 0) 기본, Harris-Pringle 은 비교용
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

RULE = {"D": None, "W": "W-FRI", "M": "ME"}


def returns(stock: pd.Series, index: pd.Series, freq: str, n: int, end: pd.Timestamp) -> pd.DataFrame:
    """기준일 이전 n개의 (종목, 지수) 단순수익률. 두 시계열 모두 가격이 있는 날만 쓴다."""
    px = pd.concat([stock.rename("s"), index.rename("m")], axis=1)
    px = px[px.index <= end].dropna()
    if RULE[freq]:
        px = px.resample(RULE[freq]).last().dropna()
    return px.pct_change().dropna().tail(n)


@dataclass
class OLS:
    beta: float
    alpha: float
    se: float
    t: float
    r2: float
    n: int
    ci_low: float
    ci_high: float


def ols(r: pd.DataFrame) -> OLS:
    x, y = r["m"].to_numpy(), r["s"].to_numpy()
    n = len(x)
    if n < 10:
        raise ValueError(f"관측치 {n}개로는 회귀할 수 없습니다.")
    xd = x - x.mean()
    beta = float((xd * (y - y.mean())).sum() / (xd ** 2).sum())
    alpha = float(y.mean() - beta * x.mean())
    resid = y - alpha - beta * x
    s2 = (resid ** 2).sum() / (n - 2)
    se = float(np.sqrt(s2 / (xd ** 2).sum()))
    r2 = float(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum())
    tc = float(stats.t.ppf(0.975, n - 2))
    return OLS(beta, alpha, se, beta / se, r2, n, beta - tc * se, beta + tc * se)


def dimson(r: pd.DataFrame) -> float:
    """β = b(t-1) + b(t) + b(t+1). 동시에 반영되지 못한 시장 정보가 다음 기에 반영되는 효과를 합산한다."""
    df = pd.DataFrame({"s": r["s"], "m0": r["m"], "m_lag": r["m"].shift(1), "m_lead": r["m"].shift(-1)}).dropna()
    X = np.column_stack([np.ones(len(df)), df["m_lag"], df["m0"], df["m_lead"]])
    coef, *_ = np.linalg.lstsq(X, df["s"].to_numpy(), rcond=None)
    return float(coef[1:].sum())


def rolling_beta(stock: pd.Series, index: pd.Series, end: pd.Timestamp, window: int = 52, span: int = 156) -> pd.Series:
    """주간 수익률 window 주 이동 베타 (최근 span 주)."""
    r = returns(stock, index, "W", span + window - 1, end)
    cov = r["s"].rolling(window).cov(r["m"])
    return (cov / r["m"].rolling(window).var()).dropna()


def zero_return_ratio(stock: pd.Series, end: pd.Timestamp, days: int = 250) -> float:
    """최근 days 거래일 중 종가가 전일과 같은 날의 비율 — 유동성(거래 빈도) 지표."""
    s = stock[stock.index <= end].dropna().tail(days + 1)
    return float((s.diff().dropna() == 0).mean())


def blume(beta: float) -> float:
    return 0.67 * beta + 0.33


def unlever_hamada(beta_l: float, de: float, tax: float) -> float:
    return beta_l / (1 + (1 - tax) * de)


def relever_hamada(beta_u: float, de: float, tax: float) -> float:
    return beta_u * (1 + (1 - tax) * de)


def unlever_harris_pringle(beta_l: float, de: float) -> float:
    return beta_l / (1 + de)


def relever_harris_pringle(beta_u: float, de: float) -> float:
    return beta_u * (1 + de)
