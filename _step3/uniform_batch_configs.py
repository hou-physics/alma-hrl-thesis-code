"""Parse every canonical {gal}_analyse_code/step3_analyze.py into the config
table for the uniform-prescription batch (mainline B stage 1).

Canonical = step3_analyze.py only; variants (_adaptive, _all_components,
_deep*, _aca, _v1_legacy) are historical modes, skipped. NGC 5253 canonical
= X1c survey tier (user decision 2026-08-01).

Run standalone to print + write the table:
  conda run -n casa_env python _step3/uniform_batch_configs.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

MY_CODE = Path("/Volumes/HouAstro/master/master_thesis/my_code")
OUT_CSV = Path("/Volumes/HouAstro/master/result_v2/_uniform_batch/batch_configs.csv")

FIELDS = ["weak_line", "strong_line", "hrl_pbcor_path", "co_nonpbcor_path",
          "co_pbcor_path", "z", "strong_line_window_kms"]


def parse_config(path: Path) -> dict | None:
    text = path.read_text()
    m = re.search(r'WORKDIR\s*=\s*["\']([^"\']+)["\']', text)
    workdir = m.group(1) if m else ""
    # module-level string variables (COMBINED_PBCOR = f"{WORKDIR}/...", etc.)
    variables = {"WORKDIR": workdir}
    for vm in re.finditer(
            r'^([A-Z_][A-Z0-9_]*)\s*=\s*f?["\']([^"\']+)["\']', text, re.M):
        variables[vm.group(1)] = vm.group(2).replace("{WORKDIR}", workdir)
    # restrict field search to the AnalysisConfig(...) call to skip docstrings
    ci = text.find("AnalysisConfig(")
    body = text[ci:] if ci != -1 else text
    out = {"config_file": str(path), "workdir": workdir}
    for f in FIELDS:
        m = re.search(
            rf'\b{f}\s*=\s*(?:f?["\']([^"\']*)["\']|([A-Z_][A-Z0-9_]*)'
            rf'|([0-9.eE+-]+))\s*,', body)
        if not m:
            out[f] = ""
            continue
        if m.group(1) is not None:
            val = m.group(1).replace("{WORKDIR}", workdir)
        elif m.group(2) is not None:
            val = variables.get(m.group(2), "")
        else:
            val = m.group(3)
        out[f] = val
    return out


def build_table():
    rows = []
    for cfg_path in sorted(MY_CODE.glob("*_analyse_code/step3_analyze.py")):
        gal_folder = cfg_path.parent.name.replace("_analyse_code", "")
        c = parse_config(cfg_path)
        strong = c["co_nonpbcor_path"] or c["co_pbcor_path"]
        strong_kind = ("nonpbcor" if c["co_nonpbcor_path"] else
                       "pbcor" if c["co_pbcor_path"] else "MISSING")
        exists = Path(strong).exists() if strong else False
        rows.append(dict(
            galaxy=gal_folder,
            z=c["z"], weak_line=c["weak_line"], strong_line=c["strong_line"],
            strong_path=strong, strong_kind=strong_kind,
            strong_exists=exists,
            hrl_path=c["hrl_pbcor_path"],
            hrl_exists=Path(c["hrl_pbcor_path"]).exists()
            if c["hrl_pbcor_path"] else False,
            window_kms=c["strong_line_window_kms"] or "300",
        ))
    # wave-1+ onboarded targets (2026-08-04): rows from the onboard CSV,
    # same schema + per-row pb_mult (pbcor × PB-plane flattening)
    wave_csv = Path("/Volumes/HouAstro/master/master_thesis/work_dir"
                    "/_wave1/wave1_configs.csv")
    if wave_csv.exists():
        import csv as _csv
        for r in _csv.DictReader(open(wave_csv)):
            rows.append(dict(
                galaxy=r["galaxy"], z=r["z"], weak_line=r["weak_line"],
                strong_line=r["strong_line"], strong_path=r["strong_path"],
                strong_kind=r["strong_kind"],
                strong_exists=Path(r["strong_path"]).exists(),
                hrl_path=r["hrl_path"],
                hrl_exists=Path(r["hrl_path"]).exists(),
                window_kms=r["window_kms"] or "300",
                pb_mult=r.get("pb_mult", ""),
            ))
    return rows


if __name__ == "__main__":
    OUT_CSV.parent.mkdir(exist_ok=True)
    rows = build_table()
    fields = list(rows[0].keys())
    for r in rows:                       # wave rows add pb_mult etc.
        fields += [k for k in r if k not in fields]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader(); w.writerows(rows)
    print(f"{'galaxy':<18}{'z':>9}{'weak':>6}{'strong':>8}{'kind':>10}"
          f"{'S✓':>4}{'H✓':>4}{'win':>6}")
    for r in rows:
        print(f"{r['galaxy']:<18}{r['z']:>9}{r['weak_line']:>6}"
              f"{r['strong_line']:>8}{r['strong_kind']:>10}"
              f"{'Y' if r['strong_exists'] else 'N':>4}"
              f"{'Y' if r['hrl_exists'] else 'N':>4}{r['window_kms']:>6}")
    print(f"\n{len(rows)} configs → {OUT_CSV}")
