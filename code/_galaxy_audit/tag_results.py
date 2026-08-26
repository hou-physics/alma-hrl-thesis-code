"""Apply macOS Finder color tags to results/{galaxy}/ dirs based on Status verdict.

Color scheme (per docs/progress.md memory entry 2026-04-29):
- Green   = clean detection (✅, S/N >= 5 + clean diagnostics)
- Yellow  = borderline / marginal (⚠️ marginal flag)
- Grey    = non-detection (❌, sub-threshold or selector-bias non-det)
- Purple  = methodology-failure (e.g. NGC 7793 full_field/all_components/adaptive
            high-S/N spurious results from baseline-residual accumulation)

Reads:
- results/{galaxy}/summary.md "Status:" line for verdict
- progress.md Status table for canonical thesis verdict (overrides summary.md
  when summary.md says "Detection ✅" but progress.md flags as non-detection
  via joint diagnostic; e.g. NGC 6810, NGC 7793, NGC 5253 X1c)

Manual overrides (methodology-failure cases) hard-coded below.
"""
from __future__ import annotations

import plistlib
import re
import subprocess
from pathlib import Path

RESULTS = Path("/Volumes/HouAstro/master/results")

# Manual override map: directory name -> color tag
# Used for NGC 7793 spurious-high-S/N variants (selector-bias amplifications)
# and for cases where summary.md headline disagrees with the thesis verdict.
MANUAL_OVERRIDES: dict[str, str] = {
    # NGC 7793 spurious-high-S/N variants (caveats-draft.md "Sample selection limit")
    "NGC7793_full_field": "Purple",
    "NGC7793_all_components": "Purple",
    "NGC7793_adaptive": "Purple",
    # NGC 7793 canonical = non-detection per joint diagnostic
    "NGC7793": "Gray",
    # NGC 6810 selector-bias non-det (S/N 3.0 looks marginal but Nbeam=2504 fails)
    "NGC6810": "Gray",
    # NGC 5253 X1c reclassified non-det after footprint-smoothing fix
    "NGC5253_x1c": "Gray",
    # NGC 5253 main detections
    "NGC5253_deep": "Green",
    "NGC5253_deep_native": "Green",
    "NGC5253": "Green",  # X1a clean
    "NGC5253_aca": "Gray",  # ACA non-det
    # Adaptive variants
    "NGC3627_adaptive": "Green",
    "NGC3627_all_components": "Green",
    "NGC3627": "Green",
    # NGC 5248 variants — all non-detections
    "NGC5248": "Gray",
    "NGC5248_adaptive": "Gray",
    "NGC5248_all_components": "Gray",
    # NGC 1313 / NGC 2903 / NGC 4569 / NGC 5236 _all_components variants → non-det
    "NGC1313_all_components": "Gray",
    "NGC2903_all_components": "Gray",
    "NGC4569_all_components": "Gray",
    "NGC5236_all_components": "Gray",
    # 2026-05-02 mode=all refinement batch (4 multi-line audit non-detections)
    # All remain non-detections; IC4687 morphology shifted from 6→137 beams
    # (IoU 0.04→0.98) — interacting LIRG pair envelope captured but S/N
    # still 2.43 < 3.27 null threshold. Documented in caveats as a
    # center-anchored mask failure in interacting pairs.
    "IC4687_all_components": "Gray",
    "IC5179_all_components": "Gray",
    "NGC5408_all_components": "Gray",
    "NGC5728_all_components": "Gray",
    # NGC 4945 detection
    "NGC4945": "Green",
    "NGC4945_adaptive": "Green",
    # NGC 7582 marginal (Sy 2 reproduction batch)
    "NGC7582": "Yellow",
    # IRAS F18293-3413 marginal (S/N 3.7)
    "IRASF18293-3413": "Yellow",
    # New 2026-05-01 detections
    "He2-10": "Green",
    # New 2026-05-01 non-detections
    "NGC5408": "Gray",
    "NGC5728": "Gray",
    "NGC1377": "Gray",
    "IC5179": "Gray",
    "IC4687": "Gray",
    "NGC7172": "Gray",
    "IRAS17208-0014": "Gray",
    # NGC 5135 reclassified 2026-05-02 from detection → non-detection
    # Per-galaxy null test (16 offsets, run_null_test.py CONTROL_GROUP):
    # p95 = 6.18 > headline 5.64; 2/16 off-line offsets exceed real-line.
    # SPW contains compact continuum residual + bandpass drift that mimic
    # 5-6σ scan-best signal at off-line offsets — disqualifies the 5.64 result.
    "NGC5135": "Gray",
}

VALID_COLORS = {"Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Gray"}


def parse_status_from_summary(summary_path: Path) -> str | None:
    """Return 'Detection' / 'Non-detection' / 'Marginal' or None if not parseable."""
    if not summary_path.exists():
        return None
    text = summary_path.read_text(errors="replace")
    m = re.search(r"^\*\*Status:\*\*\s*(.+?)$", text, re.MULTILINE)
    if not m:
        return None
    raw = m.group(1).strip()
    if "Non-detection" in raw or "non-detection" in raw:
        return "Non-detection"
    if "Marginal" in raw or "marginal" in raw:
        return "Marginal"
    if "Detection" in raw:
        return "Detection"
    return None


def status_to_color(status: str | None) -> str | None:
    if status == "Detection":
        return "Green"
    if status == "Marginal":
        return "Yellow"
    if status == "Non-detection":
        return "Gray"
    return None


def set_finder_tag(path: Path, color: str) -> None:
    if color not in VALID_COLORS:
        raise ValueError(f"invalid color {color!r}")
    plist_data = plistlib.dumps([color], fmt=plistlib.FMT_BINARY)
    hex_str = plist_data.hex()
    subprocess.run(
        ["xattr", "-wx", "com.apple.metadata:_kMDItemUserTags", hex_str, str(path)],
        check=True,
    )


def main() -> None:
    rows: list[tuple[str, str, str]] = []
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        # Decide color
        if d.name in MANUAL_OVERRIDES:
            color = MANUAL_OVERRIDES[d.name]
            source = "manual"
        else:
            status = parse_status_from_summary(d / "summary.md")
            color = status_to_color(status)
            source = f"summary.md ({status})" if status else "no-summary"
        if color is None:
            rows.append((d.name, "—", source))
            continue
        try:
            set_finder_tag(d, color)
            rows.append((d.name, color, source))
        except subprocess.CalledProcessError as e:
            rows.append((d.name, f"FAILED ({e})", source))

    # Print summary
    by_color: dict[str, list[str]] = {}
    for name, color, _ in rows:
        by_color.setdefault(color, []).append(name)
    print(f"{'Galaxy':<32} {'Color':<10} {'Source'}")
    print("-" * 70)
    for name, color, source in rows:
        print(f"{name:<32} {color:<10} {source}")
    print("\nSummary:")
    for color, names in sorted(by_color.items()):
        print(f"  {color}: {len(names)}")


if __name__ == "__main__":
    main()
