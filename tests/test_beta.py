import numpy as np
import pandas as pd
import pytest

from src import beta as B
from src.capital import common_shares, extract_debt, latest_report
from src.index_ex import ex_semis_index
from src.selection import run_chain


def _synthetic(beta=1.2, n=300, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 0.02, n)
    s = 0.001 + beta * m + rng.normal(0, 0.01, n)
    return pd.DataFrame({"s": s, "m": m})


def test_ols_matches_polyfit():
    r = _synthetic()
    o = B.ols(r)
    slope, intercept = np.polyfit(r["m"], r["s"], 1)
    assert o.beta == pytest.approx(slope)
    assert o.alpha == pytest.approx(intercept)
    assert o.ci_low < 1.2 < o.ci_high


def test_dimson_recovers_lagged_response():
    """시장 움직임의 절반이 한 기 늦게 반영되는 종목: OLS 는 절반만, Dimson 은 전체를 잡는다."""
    rng = np.random.default_rng(1)
    m = pd.Series(rng.normal(0, 0.02, 2000))
    s = 0.5 * m + 0.5 * m.shift(1).fillna(0) + rng.normal(0, 0.002, 2000)
    r = pd.DataFrame({"s": s, "m": m})
    assert B.ols(r).beta == pytest.approx(0.5, abs=0.02)
    assert B.dimson(r) == pytest.approx(1.0, abs=0.02)


def test_hamada_round_trip():
    bu = B.unlever_hamada(1.1, 0.5, 0.275)
    assert bu == pytest.approx(1.1 / (1 + 0.725 * 0.5))
    assert B.relever_hamada(bu, 0.5, 0.275) == pytest.approx(1.1)
    assert B.blume(1.0) == pytest.approx(1.0)


def test_chain_uses_median():
    raw = pd.Series([0.5, 1.0, 1.5], index=list("abc"))
    de = pd.Series([0.0, 0.2, 0.4], index=list("abc"))
    c = run_chain(raw, de, 0.25)
    assert c.target_de == pytest.approx(0.2)
    assert c.median_unlevered == pytest.approx(float(c.peers["unlevered"].median()))
    assert c.relevered == pytest.approx(c.median_unlevered * (1 + 0.75 * 0.2))


def test_ex_semis_equals_kospi_when_semis_move_with_market():
    """반도체가 지수와 똑같이 움직이면 나머지 종목도 같은 수익률 → 반도체 제외 지수 = KOSPI."""
    d = pd.bdate_range("2025-01-01", periods=50)
    k = pd.Series(100 * np.cumprod(1 + np.random.default_rng(2).normal(0, 0.01, 50)), index=d)
    ex, w = ex_semis_index(k, k * 1e10, {"x": k * 10}, {"x": 3e8})
    assert (ex / ex.iloc[0]).to_numpy() == pytest.approx((k / k.iloc[0]).to_numpy())
    assert w.iloc[0] == pytest.approx(0.3)


@pytest.mark.parametrize("date,expected", [
    ("2026-09-18", (2026, "11012")), ("2026-12-01", (2026, "11014")),
    ("2026-06-01", (2026, "11013")), ("2026-04-10", (2025, "11011")), ("2026-02-01", (2025, "11014")),
])
def test_latest_report(date, expected):
    assert latest_report(pd.Timestamp(date)) == expected


def test_extract_debt_excludes_discounts():
    rows = [
        {"sj_div": "BS", "account_nm": "단기차입금", "thstrm_amount": "100"},
        {"sj_div": "BS", "account_nm": "유동성장기차입금", "thstrm_amount": "50"},
        {"sj_div": "BS", "account_nm": "사채", "thstrm_amount": "200"},
        {"sj_div": "BS", "account_nm": "사채할인발행차금", "thstrm_amount": "-5"},
        {"sj_div": "BS", "account_nm": "비유동 리스부채", "thstrm_amount": "30"},
        {"sj_div": "BS", "account_nm": "현금및현금성자산", "thstrm_amount": "70"},
        {"sj_div": "IS", "account_nm": "단기차입금", "thstrm_amount": "999"},
    ]
    d = extract_debt(rows)
    assert (d.borrowings, d.lease, d.cash) == (350, 30, 70)


def test_common_shares():
    rows = [{"se": "보통주", "istc_totqy": "1,000", "tesstk_co": "50"}, {"se": "합계", "istc_totqy": "1,100", "tesstk_co": "50"}]
    assert common_shares(rows) == (1000, 50)


def test_portfolio_beta_equals_average_of_betas():
    """동일가중 포트폴리오의 베타 = 개별 베타의 평균. 달라지는 것은 표준오차뿐이다."""
    from src.precision import analyze
    rng = np.random.default_rng(3)
    m = pd.Series(rng.normal(0, 0.02, 400))
    rets = {f"c{i}": pd.DataFrame({"s": b * m + rng.normal(0, 0.02, 400), "m": m})
            for i, b in enumerate([0.6, 0.9, 1.2])}
    p = analyze(rets)
    assert p.portfolio.beta == pytest.approx(np.mean([B.ols(r).beta for r in rets.values()]))
    assert p.portfolio.se < p.mean_single_se          # 묶으면 정밀해지고
    assert p.se_if_independent == pytest.approx(p.mean_single_se / np.sqrt(3))


def test_residual_correlation_blocks_the_sqrt_n_gain():
    """잔차에 공통요인이 있으면 표준오차가 독립 가정치까지 줄지 않는다."""
    from src.precision import analyze
    rng = np.random.default_rng(4)
    m = pd.Series(rng.normal(0, 0.02, 400))
    common = pd.Series(rng.normal(0, 0.02, 400))      # 업종 공통 충격
    rets = {f"c{i}": pd.DataFrame({"s": 0.9 * m + common + rng.normal(0, 0.005, 400), "m": m})
            for i in range(4)}
    p = analyze(rets)
    assert p.mean_residual_corr > 0.8
    assert p.portfolio.se > p.se_if_independent * 1.5


def test_average_beta_se_matches_portfolio_regression():
    """일반화한 식 (1/N²)ΣΣρ·SE·SE 는 같은 지수일 때 포트폴리오 회귀의 표준오차와 같다."""
    from src.precision import analyze, average_beta_se
    rng = np.random.default_rng(5)
    m = pd.Series(rng.normal(0, 0.02, 300))
    common = pd.Series(rng.normal(0, 0.01, 300))
    rets = {f"c{i}": pd.DataFrame({"s": (0.7 + 0.1 * i) * m + common + rng.normal(0, 0.01, 300), "m": m})
            for i in range(4)}
    p = analyze(rets)
    se, corr, nobs = average_beta_se(p.ses, {c: p.residuals[c] for c in rets})
    assert se == pytest.approx(p.portfolio.se, rel=1e-6)
    assert corr == pytest.approx(p.mean_residual_corr)
    assert nobs == 300


def test_screening_uses_leverage_and_size_only():
    """사전 스크리닝 기준은 베타 결과가 아니라 D/E·규모로만 적용된다."""
    from src.global_peers import screen_reason
    screen = {"max_de": 2.0, "min_market_cap_eok": 1000}
    assert screen_reason(0.2, 5000e8, screen) == ""
    assert "D/E" in screen_reason(4.15, 5000e8, screen)
    assert "시가총액" in screen_reason(0.2, 210e8, screen)
