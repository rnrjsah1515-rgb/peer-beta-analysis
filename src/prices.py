"""주가·지수 수집과 소스 간 대조.

- 주가: pykrx(네이버 기반, 수정주가) 를 기본으로, FinanceDataReader 와 일별 종가를 대조한다.
- 지수: FinanceDataReader (KS11 = KOSPI, KQ11 = KOSDAQ). pykrx 지수 함수는 KRX 로그인이 필요해 쓰지 않는다.
- 수집 결과는 data/prices/*.csv 로 저장해 같은 요청을 반복하지 않는다.
"""
from __future__ import annotations

import contextlib
import io

import pandas as pd

from .config import ROOT

CACHE = ROOT / "data" / "prices"
INDEX_CODES = {"KOSPI": "KS11", "KOSDAQ": "KQ11"}


def _cached(name: str, fetch, refresh: bool) -> pd.Series:
    path = CACHE / f"{name}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s = fetch().astype(float)
    s.index = pd.to_datetime(s.index)
    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_csv(path)
    return s


def _quiet(fn, *args):
    """pykrx 가 import·호출 시 출력하는 로그인 안내 문구를 숨긴다."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args)


def close_pykrx(code: str, start: str, end: str, refresh: bool = False) -> pd.Series:
    def fetch():
        from pykrx import stock
        df = _quiet(stock.get_market_ohlcv, start.replace("-", ""), end.replace("-", ""), code)
        return df.iloc[:, 3].rename(code)          # 4번째 열 = 종가 (한글 열 이름 인코딩 문제 회피)
    return _cached(f"pykrx_{code}_{start}_{end}", fetch, refresh).rename(code)


def close_fdr(code: str, start: str, end: str, refresh: bool = False) -> pd.Series:
    def fetch():
        import FinanceDataReader as fdr
        return _quiet(fdr.DataReader, code, start, end)["Close"].rename(code)
    return _cached(f"fdr_{code}_{start}_{end}", fetch, refresh).rename(code)


def index_close(market: str, start: str, end: str, refresh: bool = False) -> pd.Series:
    return close_fdr(INDEX_CODES[market], start, end, refresh).rename(market)


def kospi_marcap(start: str, end: str, refresh: bool = False) -> pd.Series:
    """KOSPI 전체 시가총액(원) — 반도체 비중 계산용."""
    def fetch():
        import FinanceDataReader as fdr
        return _quiet(fdr.DataReader, "KS11", start, end)["MarCap"].rename("KOSPI_MarCap")
    return _cached(f"fdr_KS11_marcap_{start}_{end}", fetch, refresh)


def current_shares(codes: list[str]) -> dict[str, float]:
    """현재 상장주식수 (공공데이터포털 키가 없을 때 과거에도 같은 주식수를 쓰는 근사)."""
    import FinanceDataReader as fdr
    df = _quiet(fdr.StockListing, "KRX").set_index("Code")
    return {c: float(df.loc[c, "Stocks"]) for c in codes}


def cross_check(a: pd.Series, b: pd.Series, tol: float = 0.005) -> dict:
    """두 소스의 일별 종가 비교. 차이가 tol(0.5%) 을 넘는 날을 찾는다."""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1)
    both = df.dropna()
    gap = (both["a"] / both["b"] - 1).abs()
    bad = gap[gap > tol]
    return {
        "days_a": int(df["a"].notna().sum()),
        "days_b": int(df["b"].notna().sum()),
        "only_a": int((df["a"].notna() & df["b"].isna()).sum()),
        "only_b": int((df["b"].notna() & df["a"].isna()).sum()),
        "max_gap": float(gap.max()) if len(gap) else float("nan"),
        "bad_days": int(len(bad)),
        "first_bad": bad.index[0].date().isoformat() if len(bad) else "",
    }


def fx_rate(pair: str, date: pd.Timestamp, refresh: bool = False) -> float:
    """평가기준일(또는 직전 영업일) 환율. pair 예: 'USD/KRW'."""
    start = (date - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    s = close_fdr(pair, start, date.strftime("%Y-%m-%d"), refresh)
    return float(s[s.index <= date].dropna().iloc[-1])
