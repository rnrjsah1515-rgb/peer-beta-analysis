"""설치된 Excel 로 수식을 재계산해 파이썬 결과와 같은지 확인한다 (Windows 전용, 없으면 건너뜀)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PS = r"""
$ErrorActionPreference = 'Stop'
$x = New-Object -ComObject Excel.Application
$x.Visible = $false; $x.DisplayAlerts = $false
try {
  $wb = $x.Workbooks.Open('{path}', 0, $true)
  $x.CalculateFull()
  foreach ($ref in @({refs})) {
    $sheet, $cell = $ref.Split('!')
    Write-Output ([string]$wb.Worksheets.Item($sheet).Range($cell).Value2)
  }
  $wb.Close($false)
} finally { $x.Quit() }
"""


def excel_values(path: Path, refs: list[str]) -> list[float] | None:
    if sys.platform != "win32":
        return None
    script = PS.replace("{path}", str(path.resolve()).replace("'", "''")).replace(
        "{refs}", ",".join(f"'{r}'" for r in refs))
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                             capture_output=True, timeout=120, encoding="utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
    if out.returncode != 0 or len(lines) != len(refs):
        return None
    try:
        return [float(v) for v in lines]
    except ValueError:
        return None


def check(path: Path, expected: dict[str, float], tol: float = 1e-6) -> str:
    got = excel_values(path, list(expected))
    if got is None:
        return "생략 (Excel 을 실행할 수 없음)"
    bad = [f"{r}: 엑셀 {g:,.4f} / 파이썬 {e:,.4f}" for (r, e), g in zip(expected.items(), got)
           if abs(g - e) > tol * max(1.0, abs(e))]
    return "통과 — 엑셀 수식 결과가 파이썬 계산과 일치" if not bad else "불일치: " + "; ".join(bad)
