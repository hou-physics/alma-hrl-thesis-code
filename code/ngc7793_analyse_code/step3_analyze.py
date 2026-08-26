"""NGC 7793 step3 config (Phase Y dual-cube pipeline, off-pointing observation).

NGC 7793 is a SA(s)d spiral at d ~ 3.6 Mpc. The ALMA pointing center is offset
~26.7" from the NED nucleus (~HPBW at 230 GHz on 12m), so the PB response at
the nucleus is degraded relative to an on-axis pointing (but still ~0.98 at
the nucleus pixel in this cube — the field of view is padded well beyond the
HPBW by tclean). R_pred = -2.48 places this galaxy below the R_pred >= -2.0
null-test detection cutoff, so a non-detection is the expected outcome.

Signal region: centered on NED NGC 7793 nucleus (RA=359.457210, Dec=-32.590990).
Size 45" covers the nuclear region generously.

Noise region: offset dRA=-60", dDec=+10" from nucleus (RA=359.437428,
Dec=-32.588212). CO(2-1) mom-0 is at the noise floor there and PB_min=0.975
over a 20" box — well above the 0.3 threshold required for off-pointing cases.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC7793"

config = AnalysisConfig(
    galaxy="NGC 7793",
    z=0.000749,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC7793_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC7793_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC7793_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC7793_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC7793_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC7793_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        ra_deg=359.457210, dec_deg=-32.590990,
        size_arcsec=45.0,
    ),
    noise_region=SkyRegion(
        # dRA=-60", dDec=+10" from NED nucleus; PB_min=0.975 over 20" box,
        # |CO mom-0| ~ 0.0001 (at noise floor).
        ra_deg=359.437428, dec_deg=-32.588212,
        size_arcsec=20.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC7793",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
