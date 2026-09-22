"""반도체 제외 KOSPI (대체 시장지수).

KOSPI 시가총액에서 삼성전자(보통·우선)와 SK하이닉스가 차지하는 비중을 w 라 하면,
    R_ex(t) = (R_KOSPI(t) − w(t−1) · R_semi(t)) / (1 − w(t−1))
R_semi 는 세 종목의 시가총액가중 수익률. 기간 초 비중을 쓰므로 지수 산식(시가총액가중)과 같은 구조다.

한계: 신규상장·유상증자 등 지수 산식의 시가총액 조정은 반영하지 못한다.
상장주식수는 공공데이터포털 일별 값이 있으면 쓰고, 없으면 현재 주식수를 과거에도 적용한다.
"""
from __future__ import annotations

import pandas as pd

SEMIS = {"005930": "삼성전자", "005935": "삼성전자우", "000660": "SK하이닉스"}


def ex_semis_index(kospi_close: pd.Series, kospi_mcap: pd.Series, semi_close: dict[str, pd.Series],
                   semi_shares: dict[str, pd.Series | float]) -> tuple[pd.Series, pd.Series]:
    """(반도체 제외 지수 수준[기준일=100], 반도체 비중 w) — 일별."""
    px = pd.DataFrame(semi_close).dropna()
    caps = pd.DataFrame({
        c: px[c] * (s.reindex(px.index).ffill().bfill() if isinstance(s, pd.Series) else s)
        for c, s in semi_shares.items()
    })
    df = pd.concat([kospi_close.rename("k"), kospi_mcap.rename("kcap"), caps.sum(axis=1).rename("scap")], axis=1).dropna()
    px, caps = px.loc[df.index], caps.loc[df.index]
    w = (df["scap"] / df["kcap"]).rename("semi_weight")
    r_k = df["k"].pct_change()
    # 반도체 묶음 수익률: 전일 시가총액 비중으로 가중한 가격수익률 (주식수 변동은 수익률에서 제외)
    r_s = (px.pct_change() * caps.shift(1).div(caps.shift(1).sum(axis=1), axis=0)).sum(axis=1, min_count=1)
    r_ex = ((r_k - w.shift(1) * r_s) / (1 - w.shift(1))).fillna(0.0)
    level = 100 * (1 + r_ex).cumprod()
    return level.rename("KOSPI_ex_semis"), w
