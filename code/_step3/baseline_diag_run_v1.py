"""Run baseline_diag on the v1 6-galaxy validation set.

Hardcoded configs match PoC for direct regression check against spec §6.
"""
from pathlib import Path

from baseline_diag_runner import GalaxyJob, run_jobs, write_overview_and_master_csv


OUT_ROOT = Path("/Volumes/HouAstro/master/result_v2")
WORKDIR  = Path("/Volumes/HouAstro/master/master_thesis/work_dir")


JOBS = [
    GalaxyJob(
        name="NGC5253",
        cube_path=str(WORKDIR / "NGC5253/NGC5253_H30a_pbcor.fits"),
        seed_path=str(OUT_ROOT / "NGC5253/phase2_raw_step3/best_mask_hrl.npy"),
        z=0.0014, line="H30a", W_kms=135.0,
        out_dir=OUT_ROOT / "NGC5253/baseline_diag",
    ),
    GalaxyJob(
        name="NGC6810",
        cube_path=str(WORKDIR / "NGC6810/NGC6810_H30a_pbcor.fits"),
        seed_path=str(OUT_ROOT / "NGC6810/phase2_raw_step3/best_mask_hrl.npy"),
        z=0.006775, line="H30a", W_kms=435.0,
        out_dir=OUT_ROOT / "NGC6810/baseline_diag",
    ),
    GalaxyJob(
        name="NGC4945",
        cube_path=str(WORKDIR / "NGC4945/NGC4945_H30a_spw1_v1_contsub.fits"),
        seed_path=str(OUT_ROOT / "NGC4945/phase2_raw_step3/best_mask_hrl.npy"),
        z=0.00188, line="H30a", W_kms=315.0,
        out_dir=OUT_ROOT / "NGC4945/baseline_diag",
    ),
    GalaxyJob(
        name="NGC5135",
        cube_path=str(WORKDIR / "NGC5135/NGC5135_H30a_pbcor.fits"),
        seed_path=str(OUT_ROOT / "NGC5135/phase2_raw_step3/best_mask_hrl.npy"),
        z=0.013693, line="H30a", W_kms=345.0,
        out_dir=OUT_ROOT / "NGC5135/baseline_diag",
    ),
    GalaxyJob(
        name="He2-10",
        cube_path=str(WORKDIR / "He2-10/He2-10_H40a_pbcor.fits"),
        seed_path=str(OUT_ROOT / "He2-10/phase2_raw_step3/best_mask_hrl.npy"),
        z=0.002912, line="H40a", W_kms=75.0,
        out_dir=OUT_ROOT / "He2-10/baseline_diag",
    ),
    GalaxyJob(
        name="NGC3628",
        cube_path=str(WORKDIR / "NGC3628/NGC3628_H30a_pbcor.fits"),
        seed_path=str(OUT_ROOT / "NGC3628/phase2_raw_step3/best_mask_hrl.npy"),
        z=0.002772, line="H30a", W_kms=345.0,
        out_dir=OUT_ROOT / "NGC3628/baseline_diag",
    ),
]


def main():
    results = run_jobs(JOBS)
    write_overview_and_master_csv(
        results,
        OUT_ROOT / "_baseline_diag_master.csv",
        OUT_ROOT / "_baseline_diag_overview.png",
    )
    for r in results:
        print(f"{r.name:10s} flag={r.flag:7s} "
              f"max|F_res|/σ={r.F_res_nsigma_max:.2f}")


if __name__ == "__main__":
    main()
