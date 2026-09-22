"""공공데이터포털 금융위원회 시세정보 — 공식 출처 대조용.

- 주식시세: 일별 종가(수정 전), 상장주식수, 시가총액
- 지수시세: KOSPI·KOSDAQ 일별 종가
(2026년 현재 V2 주소. 구 주소(/service/...)는 신규 키에 대해 SERVICE_KEY_IS_NOT_REGISTERED 를 돌려준다.)
"""
from __future__ import annotations

import pandas as pd
import requests

from .cache import cached_json

STOCK_URL = "https://apis.data.go.kr/1160100/GetStockSecuritiesInfoService_V2/getStockPriceInfo_V2"
INDEX_URL = "https://apis.data.go.kr/1160100/GetMarketIndexInfoService_V2/getStockMarketIndex_V2"
INDEX_NAMES = {"KOSPI": "코스피", "KOSDAQ": "코스닥"}
PAGE = 1000


class DataGoKrError(RuntimeError):
    pass


def _range(start: str, end: str) -> dict:
    """beginBasDt 는 해당일 포함, endBasDt 는 해당일 미포함(<) — 종료일을 하루 늘려 end 당일까지 받는다."""
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    return {"beginBasDt": start.replace("-", ""), "endBasDt": end_excl}


def _fetch_all(url: str, key: str, params: dict) -> list[dict]:
    items: list[dict] = []
    page = 1
    while True:
        q = {"serviceKey": key, "resultType": "json", "numOfRows": PAGE, "pageNo": page, **params}
        resp = requests.get(url, params=q, timeout=30)
        try:
            body = resp.json()["response"]
        except (ValueError, KeyError):
            import re
            code = re.search(r'"errMsg"\s*:\s*"([^"]+)"', resp.text)
            raise DataGoKrError(f"공공데이터포털 응답 오류: {code.group(1) if code else resp.text[:200]}") from None
        if body["header"]["resultCode"] != "00":
            raise DataGoKrError(f"공공데이터포털: {body['header']['resultMsg']}")
        batch = body["body"]["items"]["item"] if body["body"]["items"] else []
        items.extend(batch)
        if len(items) >= int(body["body"]["totalCount"]) or not batch:
            return items
        page += 1


def stock_prices(key: str, code: str, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """index=날짜, columns=close(수정 전 종가), shares(상장주식수), market_cap(원)."""
    params = {"likeSrtnCd": code, **_range(start, end)}
    rows = cached_json("datagokr_stock", params, lambda: _fetch_all(STOCK_URL, key, params), refresh)["data"]
    rows = [r for r in rows if r["srtnCd"] == code]   # likeSrtnCd 는 부분일치
    if not rows:
        raise DataGoKrError(f"{code} 주가 데이터가 없습니다.")
    df = pd.DataFrame({
        "close": [float(r["clpr"]) for r in rows],
        "shares": [float(r["lstgStCnt"]) for r in rows],
        "market_cap": [float(r["mrktTotAmt"]) for r in rows],
    }, index=pd.to_datetime([r["basDt"] for r in rows], format="%Y%m%d"))
    return df.sort_index()


def index_prices(key: str, market: str, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """index=날짜, columns=close(지수), market_cap(상장시가총액, 원)."""
    params = {"idxNm": INDEX_NAMES[market], **_range(start, end)}
    rows = cached_json("datagokr_index", params, lambda: _fetch_all(INDEX_URL, key, params), refresh)["data"]
    rows = [r for r in rows if r["idxNm"] == INDEX_NAMES[market]]
    if not rows:
        raise DataGoKrError(f"{market} 지수 데이터가 없습니다.")
    df = pd.DataFrame({
        "close": [float(r["clpr"]) for r in rows],
        "market_cap": [float(r["lstgMrktTotAmt"]) for r in rows],
    }, index=pd.to_datetime([r["basDt"] for r in rows], format="%Y%m%d"))
    return df.sort_index()
