"""수집 → 대조 → 베타 측정을 한 번에 실행한다."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import beta as B
from .index_ex import SEMIS, ex_semis_index
from .prices import close_fdr, close_pykrx, cross_check, current_shares, index_close, kospi_marcap
from .sources import datagokr

# 시장지수 선택지: 라벨 → 설명
INDEX_CHOICES = {
    "KOSPI": "KOSPI (기본)",
    "LISTED": "상장시장 지수 (KOSPI/KOSDAQ)",
    "EX_SEMIS": "KOSPI 반도체 제외",
    "KOSDAQ": "KOSDAQ",
}


@dataclass
class Measurement:
    valuation_date: pd.Timestamp
    stocks: dict[str, pd.Series]        # 코드 → 일별 수정종가 (pykrx)
    indices: dict[str, pd.Series]       # KOSPI / KOSDAQ / EX_SEMIS → 일별 종가(수준)
    semi_weight: pd.Series              # KOSPI 내 반도체 3종목 시가총액 비중
    data_check: pd.DataFrame            # 소스 대조 결과
    grid: pd.DataFrame                  # 회사 × 측정조건 × 지수 별 회귀 통계
    base: pd.DataFrame                  # 기본 측정조건의 회사별 결과
    weekly_returns: pd.DataFrame        # 기본 조건의 주간 수익률 (엑셀 SLOPE 검증용)
    rolling: pd.DataFrame               # 52주 이동 베타
    official: dict[str, pd.DataFrame] = field(default_factory=dict)  # 공공데이터포털 주가 (코드 → close/shares/market_cap)
    notes: list[str] = field(default_factory=list)


def _index_for(choice: str, listed: str, indices: dict[str, pd.Series]) -> pd.Series:
    return indices[listed] if choice == "LISTED" else indices[choice]


def measure(cfg: dict, data_key: str | None = None, refresh: bool = False) -> Measurement:
    val = pd.Timestamp(cfg["case"]["valuation_date"])
    m = cfg["measurement"]
    alts = {k: (v[0], int(v[1])) for k, v in m["alternatives"].items()}
    peers = cfg["peers"]
    notes: list[str] = []

    # 가장 긴 측정기간(월간 5년 + 여유) 만큼 한 번에 수집
    start = (val - pd.DateOffset(months=66)).strftime("%Y-%m-%d")
    end = val.strftime("%Y-%m-%d")

    # ── 지수: 공식(공공데이터포털)을 기본으로, 없으면 FinanceDataReader. 다른 쪽은 대조용 ──
    fdr_idx = {mk: index_close(mk, start, end, refresh) for mk in ("KOSPI", "KOSDAQ")}
    semi_px = {c: close_pykrx(c, start, end, refresh) for c in SEMIS}
    official_idx = {}
    if data_key:
        try:
            semi_sh = {c: datagokr.stock_prices(data_key, c, start, end, refresh)["shares"] for c in SEMIS}
            official_idx = {mk: datagokr.index_prices(data_key, mk, start, end, refresh) for mk in ("KOSPI", "KOSDAQ")}
        except datagokr.DataGoKrError as e:
            notes.append(f"공공데이터포털 조회 실패로 공식 출처 대조를 생략했습니다: {str(e).splitlines()[0][:80]}")
            data_key = None
    if data_key:
        indices = {mk: official_idx[mk]["close"].rename(mk) for mk in official_idx}
        kospi_cap = official_idx["KOSPI"]["market_cap"]
    else:
        indices = dict(fdr_idx)
        kospi_cap = kospi_marcap(start, end, refresh)
        semi_sh = current_shares(list(SEMIS))
        notes.append("반도체 제외 지수: 공공데이터포털 키가 없어 현재 상장주식수를 과거에도 적용했습니다.")
    indices["EX_SEMIS"], semi_weight = ex_semis_index(indices["KOSPI"], kospi_cap, semi_px, semi_sh)

    # ── 종목 주가 + 대조 ──
    stocks, official, checks = {}, {}, []
    for code, (name, _) in peers.items():
        a = close_pykrx(code, start, end, refresh)
        stocks[code] = a
        row = {"code": code, "name": name}
        row.update({f"fdr_{k}": v for k, v in cross_check(a, close_fdr(code, start, end, refresh)).items()})
        if data_key:
            official[code] = datagokr.stock_prices(data_key, code, start, end, refresh)
            row.update({f"gov_{k}": v for k, v in cross_check(a, official[code]["close"]).items()})
        checks.append(row)
    for mk, df in official_idx.items():   # 지수: 공식(기본) vs FDR
        checks.append({"code": mk, "name": f"{mk} 지수 (공식 vs FDR)",
                       **{f"gov_{k}": v for k, v in cross_check(df["close"], fdr_idx[mk], tol=0.0005).items()}})
    data_check = pd.DataFrame(checks)

    # ── 회귀 ──
    rows = []
    for code, (name, listed) in peers.items():
        for choice in INDEX_CHOICES:
            idx = _index_for(choice, listed, indices)
            for label, (freq, n) in alts.items():
                r = B.returns(stocks[code], idx, freq, n, val)
                o = B.ols(r)
                rows.append({
                    "code": code, "name": name, "listed": listed, "index_choice": choice,
                    "setting": label, "freq": freq, "target_n": n, "n": o.n,
                    "raw_beta": o.beta, "se": o.se, "t": o.t, "r2": o.r2, "ci_low": o.ci_low, "ci_high": o.ci_high,
                    "dimson_beta": B.dimson(r), "enough_obs": o.n >= m["min_obs_ratio"] * n,
                })
    grid = pd.DataFrame(rows)

    base_index = m["market_index"]
    base = grid[(grid["setting"] == m["base"]) & (grid["index_choice"] == base_index)].set_index("code")
    base = base.assign(zero_ret=[B.zero_return_ratio(stocks[c], val) for c in base.index])

    # 엑셀 SLOPE 검증용: 모든 종목과 지수가 가격을 가진 주만 사용
    bf, bn = alts[m["base"]]
    px = pd.DataFrame({**{c: stocks[c] for c in peers}, base_index: indices[base_index]})
    px = px[px.index <= val].dropna()
    weekly = px.resample(B.RULE[bf]).last().dropna().pct_change().dropna().tail(bn)
    if any(len(B.returns(stocks[c], indices[base_index], bf, bn, val).index.difference(weekly.index)) for c in peers):
        notes.append("일부 종목의 거래정지 주가 있어 엑셀 SLOPE 검증용 공통 표본이 회사별 회귀 표본과 다를 수 있습니다.")

    rolling = pd.DataFrame({peers[c][0]: B.rolling_beta(stocks[c], indices[base_index], val) for c in peers})
    rolling["Peer 중앙값"] = rolling.median(axis=1)
    principle = m.get("principle_index", "KOSPI")
    if principle != base_index:
        rolling[f"Peer 중앙값 ({INDEX_CHOICES[principle]})"] = pd.DataFrame(
            {c: B.rolling_beta(stocks[c], indices[principle], val) for c in peers}).median(axis=1)
    return Measurement(val, stocks, indices, semi_weight, data_check, grid, base, weekly, rolling, official, notes)
