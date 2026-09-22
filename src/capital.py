"""Peer 자본구조: DART 재무상태표의 이자부부채 + 평가기준일 시가총액 → D/E.

- 부채: 평가기준일 직전에 공시된 정기보고서(분·반기 포함)의 연결재무상태표. 장부가를 시장가치의 대용치로 쓴다.
- 자기자본: 평가기준일 종가 × (발행주식수 − 자기주식수), 보통주 기준.
어떤 계정을 부채에 넣었는지 로그로 남겨 검증할 수 있게 한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .sources import dart

REPORT_NAMES = {"11011": "사업보고서", "11012": "반기보고서", "11013": "1분기보고서", "11014": "3분기보고서"}

# (항목, 계정명 패턴, 제외 패턴) — 공백과 괄호 속 문구("(유동)" 등)를 없앤 계정명에 적용
DEBT_RULES = (
    ("borrowings", r"(차입금|차입부채|사채|유동성장기부채)$", r"할인|할증|이자|상환할증금|전환권|신주인수권"),
    ("lease", r"리스(유동|비유동)?부채$", None),
    ("cash", r"^현금및현금성자산$", None),
    # 차입금을 본문에 따로 표시하지 않고 '금융부채'로 묶는 회사용 — 차입금이 하나도 없을 때만 쓰고 표시한다
    ("financial_aggregate", r"^(단기|장기|유동|비유동)금융부채$", None),
)


def latest_report(val: pd.Timestamp) -> tuple[int, str]:
    """평가기준일 현재 공시가 끝났을 가장 최근 정기보고서 (법정 제출기한 기준)."""
    y, md = val.year, (val.month, val.day)
    if md >= (11, 15):
        return y, "11014"
    if md >= (8, 15):
        return y, "11012"
    if md >= (5, 16):
        return y, "11013"
    if md >= (3, 31):
        return y - 1, "11011"
    return y - 1, "11014"


def _amount(text: str | None) -> float:
    try:
        return float((text or "").replace(",", ""))
    except ValueError:
        return 0.0


@dataclass
class DebtResult:
    borrowings: float = 0.0
    lease: float = 0.0
    cash: float = 0.0
    financial_aggregate: float = 0.0
    currency: str = "KRW"
    log: list[dict] = field(default_factory=list)


def _clean(name: str) -> str:
    return re.sub(r"\(.*?\)|\s+", "", name or "")


def extract_debt(rows: list[dict]) -> DebtResult:
    out = DebtResult()
    bs = [r for r in rows if r.get("sj_div") == "BS"]
    out.currency = next((r["currency"] for r in bs if r.get("currency")), "KRW")
    for line, pattern, exclude in DEBT_RULES:
        for r in bs:
            name = _clean(r.get("account_nm", ""))
            if re.search(pattern, name) and not (exclude and re.search(exclude, name)):
                amt = _amount(r.get("thstrm_amount"))
                setattr(out, line, getattr(out, line) + amt)
                out.log.append({"line": line, "account_id": r.get("account_id"), "account_nm": r.get("account_nm"),
                                "amount": amt})
                if line == "cash":
                    break
    return out


def common_shares(rows: list[dict]) -> tuple[float, float]:
    """(발행주식총수, 자기주식수) — 보통주 행."""
    for r in rows:
        if "보통" in (r.get("se") or ""):
            return _amount(r.get("istc_totqy")), _amount(r.get("tesstk_co"))
    raise ValueError("주식총수 현황에서 보통주 행을 찾지 못했습니다.")


def peer_capital(key: str, peers: dict, closes: dict[str, pd.Series], val: pd.Timestamp,
                 fx: dict[str, float] | None = None, overrides: dict | None = None,
                 refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """회사별 부채·시가총액 표(원 단위)와 계정 매핑 로그.

    fx: 통화 → 원화 환율 (외화로 공시하는 해외 상장사용, 평가기준일 환율)
    overrides: 코드 → {borrowings, lease (억원), source} — 주석에서 직접 확인한 값
    """
    fx, overrides = {"KRW": 1.0, **(fx or {})}, overrides or {}
    year, rcode = latest_report(val)
    rows, logs = [], []
    for code, (name, _) in peers.items():
        corp, _ = dart.corp_code(key, code, refresh)
        try:
            fs, basis = dart.financial_statements(key, corp, year, "CFS", refresh, rcode), "연결"
        except dart.DartError:   # 종속회사가 없어 별도재무제표만 공시하는 회사
            fs, basis = dart.financial_statements(key, corp, year, "OFS", refresh, rcode), "별도"
        d = extract_debt(fs)
        if d.currency not in fx:
            raise ValueError(f"{name}: {d.currency} 환율이 없습니다.")
        rate = fx[d.currency]
        borrowings, lease, flags = d.borrowings * rate, d.lease * rate, []
        if d.currency != "KRW":
            flags.append(f"{d.currency} 공시 → {rate:,.1f}원 환산")
        if code in overrides:
            o = overrides[code]
            borrowings, lease = o["borrowings"] * 1e8, o["lease"] * 1e8
            flags.append(f"주석 수기입력: {o['source']}")
        elif d.borrowings == 0 and d.financial_aggregate > 0:
            borrowings = d.financial_aggregate * rate - lease
            flags.append("본문 '금융부채' 합계 사용 (차입금 미구분)")
        issued, treasury = common_shares(dart.share_counts(key, corp, year, rcode, refresh))
        px = closes[code][closes[code].index <= val].dropna()
        rows.append({
            "code": code, "name": name, "report": f"{year} {REPORT_NAMES[rcode]} ({basis})",
            "borrowings": borrowings, "lease": lease, "cash": d.cash * rate, "flags": "; ".join(flags),
            "issued": issued, "treasury": treasury, "price": float(px.iloc[-1]), "price_date": px.index[-1].date(),
        })
        logs.extend({"code": code, "name": name, "currency": d.currency, **x} for x in d.log)
    df = pd.DataFrame(rows).set_index("code")
    df["market_cap"] = df["price"] * (df["issued"] - df["treasury"])
    return df, pd.DataFrame(logs)
