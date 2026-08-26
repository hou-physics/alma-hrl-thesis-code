"""CLI: regenerate plots from a cached step3 analysis (no cube re-load).

Usage:
    python replot.py <output_dir> [<output_dir> ...]

Example:
    python replot.py /Volumes/HouAstro/master/results/NGC1313
    python replot.py /Volumes/HouAstro/master/results/NGC*

Requires that `_plot_cache.pkl` exists in each output_dir (written
automatically by `analyze()` on the previous run).
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1
    sys.path.insert(0, str(Path(__file__).parent))
    from analyze import replot_from_cache
    errors = 0
    for d in sys.argv[1:]:
        try:
            replot_from_cache(d)
        except Exception as e:  # noqa: BLE001 — surface any failure as per-galaxy
            print(f"[{d}] FAILED: {e}", file=sys.stderr)
            errors += 1
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
