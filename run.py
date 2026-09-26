"""실행: python run.py            (.env 에 DART_API_KEY 필요, DATA_GO_KR_API_KEY 는 있으면 공식 출처 대조)
       python run.py --refresh  (캐시를 무시하고 다시 수집)
"""
from __future__ import annotations

import argparse

import numpy as np

from src import report
from src.capital import peer_capital
from src.config import ROOT, api_key, load_config, load_env
from src.excel import build
from src.excel_check import check
from src.pipeline import INDEX_CHOICES, measure
from src.precision import analyze, relevered_band
from src.prices import fx_rate
from src import beta as B
from src.selection import debt_of, range_table, run_chain


def main():
    ap = argparse.ArgumentParser(description="유사기업 베타 분석")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--skip-excel-check", action="store_true")
    args = ap.parse_args()

    load_env()
    cfg = load_config()
    dart_key, data_key = api_key("DART_API_KEY"), api_key("DATA_GO_KR_API_KEY")
    if not dart_key:
        raise SystemExit(".env 에 DART_API_KEY 를 입력하세요 (Peer 차입금·주식수 수집에 필요).")

    print("[1/4] 주가·지수 수집, 소스 대조, 회귀")
    meas = measure(cfg, data_key, args.refresh)
    for note in meas.notes:
        print("  ·", note)

    print("[2/4] DART 차입금·주식수")
    capc = cfg["capital"]
    fx = {"USD": fx_rate("USD/KRW", meas.valuation_date, args.refresh)}
    cap, debt_log = peer_capital(dart_key, cfg["peers"], meas.stocks, meas.valuation_date, fx,
                                 capc.get("overrides"), args.refresh)
    for code, row in cap[cap["flags"] != ""].iterrows():
        print(f"  · {row['name']}: {row['flags']}")
    cap["debt"] = debt_of(cap, capc["include_lease"], capc["debt_basis"])
    cap["de"] = cap["debt"] / cap["market_cap"]
    if meas.official:   # 공식 시가총액(상장주식수 기준)과 비교
        cap["gov_market_cap"] = [
            float(meas.official[c]["market_cap"].loc[:meas.valuation_date].iloc[-1]) for c in cap.index]

    print("[3/4] 베타 산정")
    tax = capc["tax_rate"]
    idx = cfg["measurement"]["market_index"]
    w = meas.weekly_returns
    raw = w[list(cfg["peers"])].apply(lambda s: np.cov(s, w[idx], ddof=1)[0, 1] / w[idx].var())
    chain = run_chain(raw, cap["de"], tax)
    ranges = range_table(meas.grid, cap["de"], tax, INDEX_CHOICES)
    freq, nobs = cfg["measurement"]["alternatives"][cfg["measurement"]["base"]]
    prec = analyze({c: B.returns(meas.stocks[c], meas.indices[idx], freq, int(nobs), meas.valuation_date)
                    for c in cfg["peers"]}, cap["de"])
    band = relevered_band(prec, chain.target_de, tax)
    coe = cfg["cost_of_equity"]
    print(f"  추정오차: 포트폴리오 β {prec.portfolio.beta:.3f} (SE {prec.portfolio.se:.3f}, R² {prec.portfolio.r2:.1%})"
          f" → 재레버 구간 {band[0]:.2f}~{band[1]:.2f}"
          f" · Ke {coe['risk_free'] + band[0] * coe['erp']:.1%}~{coe['risk_free'] + band[1] * coe['erp']:.1%}")
    print(f"  βu 중앙값 {chain.median_unlevered:.3f} · 목표 D/E {chain.target_de:.1%} → 재레버 β {chain.relevered:.3f}")

    print("[4/4] 엑셀·보고서")
    xlsx = ROOT / "output" / "peer_beta.xlsx"
    refs = build(xlsx, cfg, meas, cap, debt_log, ranges, INDEX_CHOICES, prec, band)
    expected = {refs["median_unlevered"]: chain.median_unlevered, refs["relevered"]: chain.relevered,
                **{refs[f"raw_{c}"]: float(raw[c]) for c in cfg["peers"]}}
    excel_result = "생략" if args.skip_excel_check else check(xlsx, expected)
    print("  엑셀 대조:", excel_result)
    report.write(ROOT / "output" / "summary.md", cfg, meas, cap, chain, ranges, excel_result, prec, band)
    print("완료:", xlsx)


if __name__ == "__main__":
    main()
