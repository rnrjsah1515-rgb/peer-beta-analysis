"""금융감독원 Open DART API.

- corpCode.xml : 종목코드 → DART 고유번호(corp_code) 변환
- fnlttSinglAcntAll.json : 단일회사 전체 재무제표 (사업보고서 = reprt_code 11011)
"""
from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile

import time

import requests

from .cache import CACHE_DIR, cached_json


def get(url: str, params: dict, timeout: int = 30, tries: int = 3):
    """일시적인 네트워크·SSL 오류는 잠깐 쉬었다 다시 시도한다."""
    for attempt in range(tries):
        try:
            return requests.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))

BASE = "https://opendart.fss.or.kr/api"
ANNUAL_REPORT = "11011"


class DartError(RuntimeError):
    pass


def corp_code(key: str, stock_code: str, refresh: bool = False) -> tuple[str, str]:
    """종목코드로 (corp_code, corp_name) 을 찾는다."""
    path = CACHE_DIR / "dart" / "CORPCODE.xml"
    if refresh or not path.exists():
        resp = get(f"{BASE}/corpCode.xml", {"crtfc_key": key}, timeout=60)
        if not resp.content.startswith(b"PK"):  # 오류 시 zip 이 아닌 JSON/XML 이 온다
            raise DartError(f"corpCode.xml 다운로드 실패: {resp.text[:200]}")
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(zf.read(zf.namelist()[0]))
    for node in ET.parse(path).getroot().iter("list"):
        if (node.findtext("stock_code") or "").strip() == stock_code:
            return node.findtext("corp_code"), node.findtext("corp_name")
    raise DartError(f"종목코드 {stock_code} 에 해당하는 DART 고유번호가 없습니다.")


def financial_statements(key: str, corp: str, year: int, fs_div: str = "CFS", refresh: bool = False,
                         report_code: str = ANNUAL_REPORT) -> list[dict]:
    """전체 재무제표 계정 목록. 각 행은 account_id, account_nm, sj_div, thstrm_amount 등을 가진다.

    report_code: 11011 사업보고서 / 11012 반기보고서 / 11013 1분기보고서 / 11014 3분기보고서
    """
    params = {"crtfc_key": key, "corp_code": corp, "bsns_year": str(year), "reprt_code": report_code, "fs_div": fs_div}

    def fetch():
        body = get(f"{BASE}/fnlttSinglAcntAll.json", params).json()
        if body.get("status") != "000":
            raise DartError(f"DART {year} {fs_div}: {body.get('status')} {body.get('message')}")
        return body["list"]

    cache_key = {k: v for k, v in params.items() if k != "crtfc_key"}
    return cached_json("dart", cache_key, fetch, refresh)["data"]


def share_counts(key: str, corp: str, year: int, report_code: str, refresh: bool = False) -> list[dict]:
    """주식의 총수 현황 (발행주식총수 istc_totqy, 자기주식수 tesstk_co, 유통주식수 distb_stock_co)."""
    params = {"crtfc_key": key, "corp_code": corp, "bsns_year": str(year), "reprt_code": report_code}

    def fetch():
        body = get(f"{BASE}/stockTotqySttus.json", params).json()
        if body.get("status") != "000":
            raise DartError(f"DART 주식총수 {year}: {body.get('status')} {body.get('message')}")
        return body["list"]

    cache_key = {k: v for k, v in params.items() if k != "crtfc_key"}
    return cached_json("dart_shares", cache_key, fetch, refresh)["data"]
