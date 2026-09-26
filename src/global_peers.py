"""해외 유사기업 (확장표본).

국내 Peer 만으로는 업종 공통요인 때문에 표준오차가 0.14 아래로 내려가지 않는다(precision.py).
공통요인이 다른 해외 ODM 을 넣으면 그 한계를 낮출 수 있다.

측정은 각 사의 상장 시장 지수 대비로 하고(현지 시장이 그 주주의 시장 포트폴리오), 현지 법인세율로
언레버해 무부채 베타로 비교한다. 자료는 yfinance(주가·지수·Total Debt·주식수).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import pandas as pd

from . import beta as B
from .config import ROOT
from .prices import fx_rate

CACHE = ROOT / "data" / "global"


def _history(ticker: str, start: str, end: str, refresh: bool = False) -> pd.Series:
    path = CACHE / f"{ticker.replace('^', '_')}_{start}_{end}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    import yfinance as yf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)["Close"]
    s.index = pd.to_datetime(s.index.date)
    path.parent.mkdir(parents=True, exist_ok=True)
    s.rename(ticker).to_frame().to_csv(path)
    return s.rename(ticker)


def _fundamentals(ticker: str, refresh: bool = False) -> dict:
    """Total Debt(차입금+사채+리스), 주식수, 표시통화, 기준일."""
    path = CACHE / f"{ticker.replace('^', '_')}_fund.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path).iloc[0].to_dict()
    import yfinance as yf
    tk = yf.Ticker(ticker)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bs, info = tk.balance_sheet, tk.info
    row = {
        "total_debt": float(bs.loc["Total Debt"].iloc[0]),
        "cash": float(bs.loc["Cash And Cash Equivalents"].iloc[0]) if "Cash And Cash Equivalents" in bs.index else 0.0,
        "shares": float(info["sharesOutstanding"]),
        "currency": info.get("currency", ""),
        "asof": str(bs.columns[0].date()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
    return row


def krw_rate(currency: str, date: pd.Timestamp, refresh: bool = False) -> float:
    """1 통화단위당 원화. 직접 고시가 없으면 달러를 거쳐 환산한다."""
    if currency == "KRW":
        return 1.0
    try:
        return fx_rate(f"{currency}/KRW", date, refresh)
    except Exception:
        usd_krw = fx_rate("USD/KRW", date, refresh)
        return usd_krw / fx_rate(f"USD/{currency}", date, refresh)


def screen_reason(de: float, market_cap_krw: float, screen: dict) -> str:
    """사전 스크리닝 판정 — 결과(베타)가 아니라 D/E 와 규모로만 적용한다."""
    why = []
    if "max_de" in screen and de > screen["max_de"]:
        why.append(f"D/E {de:.0%} > 기준 {screen['max_de']:.0%}")
    if "min_market_cap_eok" in screen and market_cap_krw / 1e8 < screen["min_market_cap_eok"]:
        why.append(f"시가총액 {market_cap_krw / 1e8:,.0f}억원 < 기준 {screen['min_market_cap_eok']:,.0f}억원")
    return "; ".join(why)


@dataclass
class GlobalResult:
    table: pd.DataFrame                     # 채택된 해외 Peer
    excluded: pd.DataFrame                  # 스크리닝 제외
    residuals: dict[str, pd.Series] = field(default_factory=dict)
    ses: dict[str, float] = field(default_factory=dict)


def measure_global(cfg: dict, val: pd.Timestamp, refresh: bool = False) -> GlobalResult:
    gp = dict(cfg.get("global_peers") or {})
    screen = gp.pop("screen", {})
    if not gp:
        return GlobalResult(pd.DataFrame(), pd.DataFrame())
    freq, nobs = cfg["measurement"]["alternatives"][cfg["measurement"]["base"]]
    start = (val - pd.DateOffset(months=30)).strftime("%Y-%m-%d")
    end = (val + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    rows, resid, ses = [], {}, {}
    for ticker, spec in gp.items():
        name, country, index_ticker, tax = spec[0], spec[1], spec[2], float(spec[3])
        px, ix = _history(ticker, start, end, refresh), _history(index_ticker, start, end, refresh)
        r = B.returns(px, ix, freq, int(nobs), val)
        o = B.ols(r)
        f = _fundamentals(ticker, refresh)
        price = float(px[px.index <= val].dropna().iloc[-1])
        mcap = price * f["shares"]
        rate = krw_rate(f["currency"], val, refresh)
        de = f["total_debt"] / mcap
        blume = B.blume(o.beta)
        rows.append({
            "ticker": ticker, "name": name, "country": country, "index": index_ticker, "tax": tax,
            "raw_beta": o.beta, "se": o.se, "t": o.t, "r2": o.r2, "n": o.n,
            "ci_low": o.ci_low, "ci_high": o.ci_high, "blume": blume,
            "debt": f["total_debt"] * rate, "market_cap": mcap * rate, "de": de,
            "unlevered": B.unlever_hamada(blume, de, tax), "currency": f["currency"], "asof": f["asof"],
        })
        resid[name] = r["s"] - o.alpha - o.beta * r["m"]
        ses[name] = o.se

    df = pd.DataFrame(rows).set_index("ticker")
    df["excluded_reason"] = [screen_reason(r["de"], r["market_cap"], screen) for _, r in df.iterrows()]
    keep = df[df["excluded_reason"] == ""]
    return GlobalResult(keep, df[df["excluded_reason"] != ""],
                        {k: v for k, v in resid.items() if k in set(keep["name"])},
                        {k: v for k, v in ses.items() if k in set(keep["name"])})
