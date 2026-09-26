"""엑셀 워크북. '베타산정' 시트는 주간수익률 원자료에서 SLOPE 부터 재레버까지 수식으로 연결한다."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

EOK = 1e8
HEAD = PatternFill("solid", fgColor="1F3864")
INPUT = PatternFill("solid", fgColor="FFF2CC")
KEY = PatternFill("solid", fgColor="E2EFDA")
WHITE = Font(color="FFFFFF", bold=True)
BOLD = Font(bold=True)
THIN = Border(bottom=Side(style="thin", color="999999"))
F3, PCT, NUM = "0.000", "0.0%", "#,##0"


def _header(ws, row: int, labels: list[str], widths: list[int] | None = None):
    for i, text in enumerate(labels, 1):
        c = ws.cell(row, i, text)
        c.fill, c.font = HEAD, WHITE
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        if widths:
            ws.column_dimensions[get_column_letter(i)].width = widths[i - 1]


def _frame(ws, df: pd.DataFrame, row: int = 1, formats: dict[str, str] | None = None, index: bool = False):
    df = df.reset_index() if index else df
    _header(ws, row, [str(c) for c in df.columns])
    for r, rec in enumerate(df.itertuples(index=False), row + 1):
        for c, v in enumerate(rec, 1):
            if isinstance(v, pd.Timestamp):
                v = v.to_pydatetime()
            cell = ws.cell(r, c, v)
            fmt = (formats or {}).get(df.columns[c - 1])
            if fmt:
                cell.number_format = fmt
    for i, col in enumerate(df.columns, 1):
        ws.column_dimensions[get_column_letter(i)].width = max(10, min(40, len(str(col)) * 2 + 2))


def build(path: Path, cfg: dict, meas, cap: pd.DataFrame, debt_log: pd.DataFrame, ranges: pd.DataFrame,
          index_labels: dict[str, str], prec=None, band=None, gl=None, ext=None) -> dict[str, str]:
    """워크북을 저장하고, 파이썬 결과와 대조할 셀 주소를 돌려준다."""
    wb = Workbook()
    peers = list(cfg["peers"])
    n = len(peers)
    capc = cfg["capital"]
    idx_name = cfg["measurement"]["market_index"]

    # ── 주간수익률 (원자료) ──
    wr = wb.create_sheet("주간수익률")
    w = meas.weekly_returns
    _header(wr, 1, ["주(금요일)"] + [cfg["peers"][c][0] for c in peers] + [index_labels.get(idx_name, idx_name)])
    for r, (d, rec) in enumerate(w.iterrows(), 2):
        wr.cell(r, 1, d.to_pydatetime()).number_format = "yyyy-mm-dd"
        for c, col in enumerate(peers + [idx_name], 2):
            wr.cell(r, c, float(rec[col])).number_format = "0.00%"
    wr.column_dimensions["A"].width = 12
    last = len(w) + 1
    idx_col = get_column_letter(n + 2)
    x_rng = f"주간수익률!${idx_col}$2:${idx_col}${last}"

    # ── 베타산정 (수식) ──
    ws = wb["Sheet"]
    ws.title = "베타산정"
    ws["A1"] = f"{cfg['case']['target_name']} — 유사기업 베타 산정"
    ws["A1"].font = Font(bold=True, size=14)
    inputs = [
        ("평가기준일", meas.valuation_date.to_pydatetime(), "yyyy-mm-dd"),
        ("시장지수", index_labels.get(idx_name, idx_name), None),
        ("측정조건", f"주간 수익률 {len(w)}주 (금요일 종가)", None),
        ("법인세율 (한계)", capc["tax_rate"], PCT),
        ("Blume 가중치 (원시)", 0.67, "0.00"),
        ("Blume 상수 (시장=1)", 0.33, "0.00"),
        ("리스부채 포함 (1/0)", 1 if capc["include_lease"] else 0, "0"),
        ("순차입금 기준 (1/0)", 1 if capc["debt_basis"] == "net" else 0, "0"),
    ]
    for i, (label, v, fmt) in enumerate(inputs, 3):
        ws.cell(i, 1, label)
        c = ws.cell(i, 2, v)
        c.fill = INPUT
        if fmt:
            c.number_format = fmt
    T, BW, BC, LF, NF = "$B$6", "$B$7", "$B$8", "$B$9", "$B$10"

    hr = 12
    cols = ["코드", "회사", "상장시장", "원시 β\n(SLOPE)", "R²\n(RSQ)", "Blume β", "차입금·사채\n(억원)", "리스부채\n(억원)",
            "현금\n(억원)", "부채 D\n(억원)", "시가총액 E\n(억원)", "D/E", "βu\n(Hamada)", "βu\n(Harris-Pringle)", "부채 출처"]
    _header(ws, hr, cols, [9, 16, 9, 10, 8, 9, 12, 11, 11, 11, 12, 8, 10, 12, 18])
    ws.row_dimensions[hr].height = 32
    first = hr + 1
    for i, code in enumerate(peers):
        r = first + i
        name, listed = cfg["peers"][code]
        y_col = get_column_letter(i + 2)
        y_rng = f"주간수익률!${y_col}$2:${y_col}${last}"
        k = cap.loc[code]
        vals = [code, name, listed, f"=SLOPE({y_rng},{x_rng})", f"=RSQ({y_rng},{x_rng})", f"={BW}*D{r}+{BC}",
                k["borrowings"] / EOK, k["lease"] / EOK, k["cash"] / EOK,
                f"=MAX(G{r}+{LF}*H{r}-{NF}*I{r},0)", k["market_cap"] / EOK, f"=J{r}/K{r}",
                f"=F{r}/(1+(1-{T})*L{r})", f"=F{r}/(1+L{r})", k["report"]]
        fmts = [None, None, None, F3, PCT, F3, NUM, NUM, NUM, NUM, NUM, PCT, F3, F3, None]
        for c, (v, fmt) in enumerate(zip(vals, fmts), 1):
            cell = ws.cell(r, c, v)
            if fmt:
                cell.number_format = fmt
    lastp = first + n - 1
    for j, (label, fn) in enumerate([("중앙값", "MEDIAN"), ("평균", "AVERAGE")]):
        r = lastp + 1 + j
        ws.cell(r, 2, label).font = BOLD
        for col in "DEFL" + "MN":
            cell = ws.cell(r, ord(col) - 64, f"={fn}({col}{first}:{col}{lastp})")
            cell.number_format = PCT if col in "EL" else F3
            cell.font = BOLD
    med = lastp + 1

    tr = med + 3
    ws.cell(tr, 1, "대상회사 재레버").font = Font(bold=True, size=12)
    target = [
        ("목표 D/E (Peer 중앙값)", f"=L{med}", PCT),
        ("목표 D/(D+E)", f"=B{tr + 1}/(1+B{tr + 1})", PCT),
        ("무부채 베타 βu (Peer 중앙값, Hamada)", f"=M{med}", F3),
        ("재레버 베타 βL = βu × (1 + (1−t) × D/E)", f"=B{tr + 3}*(1+(1-{T})*B{tr + 1})", F3),
        ("[비교] Harris-Pringle βL = βu × (1 + D/E)", f"=N{med}*(1+B{tr + 1})", F3),
    ]
    for i, (label, f, fmt) in enumerate(target, tr + 1):
        ws.cell(i, 1, label)
        c = ws.cell(i, 2, f)
        c.number_format = fmt
    ws.cell(tr + 4, 2).fill = KEY
    ws.cell(tr + 4, 2).font = BOLD
    ws.column_dimensions["A"].width = 40
    note = tr + 7
    ws.cell(note, 1, "노란 셀은 입력값. 원시 β 는 '주간수익률' 시트의 원자료로 계산되며 파이썬 결과와 자동 대조한다.")
    ws.cell(note + 1, 1, "부채는 장부가(시장가치 대용), 시가총액은 평가기준일 종가 × (발행주식수 − 자기주식수).")

    # ── 범위 (지수 × 측정조건) ──
    rs = wb.create_sheet("범위")
    rs["A1"] = "시장지수 · 측정조건별 재레버 베타 (목표 D/E 는 모든 조합에서 동일)"
    rs["A1"].font = BOLD
    view = ranges[["index_label", "setting", "median_raw", "median_r2", "significant", "n_peers",
                   "median_unlevered", "relevered"]].rename(columns={
        "index_label": "시장지수", "setting": "측정조건", "median_raw": "원시 β 중앙값", "median_r2": "R² 중앙값",
        "significant": "β>0 유의 (95%)", "n_peers": "Peer 수", "median_unlevered": "βu 중앙값", "relevered": "재레버 β"})
    _frame(rs, view, 3, {"원시 β 중앙값": F3, "R² 중앙값": PCT, "βu 중앙값": F3, "재레버 β": F3})

    # ── 추정오차 ──
    if prec is not None:
        ps = wb.create_sheet("추정오차")
        coe = cfg["cost_of_equity"]
        rows = [
            ("Peer 동일가중 포트폴리오 베타", prec.portfolio.beta, F3),
            ("표준오차", prec.portfolio.se, F3),
            ("t값", prec.portfolio.t, "0.0"),
            ("R²", prec.portfolio.r2, PCT),
            ("95% 신뢰구간 하한 (원시)", prec.portfolio.ci_low, F3),
            ("95% 신뢰구간 상한 (원시)", prec.portfolio.ci_high, F3),
            ("개별 Peer 표준오차 평균", prec.mean_single_se, F3),
            ("잔차 독립 가정 시 기대 표준오차", prec.se_if_independent, F3),
            ("Peer 잔차 간 평균 상관", prec.mean_residual_corr, "0.00"),
            ("개별 베타의 표준편차", prec.beta_dispersion, F3),
            ("재레버 베타 구간 하한", band[0], F3),
            ("재레버 베타 구간 상한", band[1], F3),
            ("무위험이자율 (가정)", coe["risk_free"], PCT),
            ("시장위험프리미엄 (가정)", coe["erp"], PCT),
            ("자기자본비용 하한", coe["risk_free"] + band[0] * coe["erp"], PCT),
            ("자기자본비용 상한", coe["risk_free"] + band[1] * coe["erp"], PCT),
        ]
        ps["A1"] = "베타 추정의 정밀도 — R² 가 아니라 표준오차로 판단한다"
        ps["A1"].font = BOLD
        for i, (label, v, fmt) in enumerate(rows, 3):
            ps.cell(i, 1, label)
            c = ps.cell(i, 2, float(v))
            c.number_format = fmt
        ps.cell(len(rows) + 4, 1, "Peer 를 묶어도 표준오차가 '독립 가정' 값까지 줄지 않는 이유는 잔차가 서로 상관되어 있기 때문이다(업종 공통요인).")
        ps.column_dimensions["A"].width = 44
        ps.column_dimensions["B"].width = 14

    # ── 확장표본 ──
    if ext:
        gs2 = wb.create_sheet("확장표본")
        gs2["A1"] = "해외 ODM 을 더한 확장표본 — 공통요인이 다른 표본을 넣어야 추정오차가 실제로 줄어든다"
        gs2["A1"].font = BOLD
        cols = ["회사", "국가", "시장지수", "원시 β", "SE", "R²", "D/E", "βu", "통화", "재무제표 기준일"]
        view = gl.table.reset_index()[["name", "country", "index", "raw_beta", "se", "r2", "de", "unlevered",
                                       "currency", "asof"]]
        view.columns = cols
        _frame(gs2, view, 3, {"원시 β": F3, "SE": F3, "R²": PCT, "D/E": PCT, "βu": F3})
        r0 = len(view) + 6
        for i, (label, dom, e) in enumerate([
            ("평균 베타의 표준오차", prec.portfolio.se, ext["se"]),
            ("잔차 간 평균 상관", prec.mean_residual_corr, ext["mean_corr"]),
            ("βu 중앙값", prec.portfolio.beta and float(ext["median_unlevered"]), ext["median_unlevered"]),
            ("재레버 베타 구간 하한", band[0], ext["band"][0]),
            ("재레버 베타 구간 상한", band[1], ext["band"][1]),
        ], start=0):
            gs2.cell(r0 + i, 1, label)
            gs2.cell(r0 + i, 2, float(dom)).number_format = F3
            gs2.cell(r0 + i, 3, float(e)).number_format = F3
        gs2.cell(r0 - 1, 2, "국내 6사").font = BOLD
        gs2.cell(r0 - 1, 3, f"확장표본 {ext['n']}사").font = BOLD
        if not gl.excluded.empty:
            gs2.cell(r0 + 6, 1, "사전 스크리닝 제외: " + "; ".join(
                f"{r['name']} ({r['excluded_reason']})" for _, r in gl.excluded.iterrows()))
        gs2.column_dimensions["A"].width = 30

    # ── 회귀통계 ──
    gs = wb.create_sheet("회귀통계")
    g = meas.grid.assign(index_choice=meas.grid["index_choice"].map(index_labels))
    _frame(gs, g, 1, {c: F3 for c in ("raw_beta", "se", "t", "ci_low", "ci_high", "dimson_beta")} | {"r2": PCT})

    # ── 이동베타 ──
    rb = wb.create_sheet("이동베타")
    rb["A1"] = f"52주 이동 베타 (vs {index_labels.get(idx_name, idx_name)})"
    _frame(rb, meas.rolling.rename_axis("주").reset_index(), 3, {c: F3 for c in meas.rolling.columns} | {"주": "yyyy-mm-dd"})

    # ── 반도체비중 ──
    sw = wb.create_sheet("반도체비중")
    s = meas.semi_weight.resample("W-FRI").last().dropna()
    _frame(sw, s.rename_axis("주").rename("삼성전자·우·SK하이닉스 / KOSPI 시가총액").reset_index(), 1,
           {"주": "yyyy-mm-dd", "삼성전자·우·SK하이닉스 / KOSPI 시가총액": PCT})

    # ── 데이터대조 / 부채매핑 ──
    dc = wb.create_sheet("데이터대조")
    _frame(dc, meas.data_check, 1, {c: "0.00%" for c in meas.data_check.columns if c.endswith("max_gap")})
    for i, note in enumerate(meas.notes, len(meas.data_check) + 3):
        dc.cell(i, 1, note)
    if not debt_log.empty:
        dl = wb.create_sheet("부채매핑")
        _frame(dl, debt_log.assign(amount=debt_log["amount"] / EOK).rename(columns={"amount": "금액(억원)"}), 1,
               {"금액(억원)": NUM})

    wb.move_sheet("범위", offset=-(len(wb.sheetnames) - 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return {"median_unlevered": f"베타산정!B{tr + 3}", "relevered": f"베타산정!B{tr + 4}",
            **{f"raw_{c}": f"베타산정!D{first + i}" for i, c in enumerate(peers)}}
