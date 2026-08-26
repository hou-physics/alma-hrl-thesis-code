"""Remove stage3_ir.csv rows left unusable by IR-service outages
(cog_flag == 'no-image' or note == 'no-iras') so a stage-3 resume
re-fetches exactly those galaxies. Prints 'BAD=<n> <names>'."""
import csv

P = "/Volumes/HouAstro/master/result_v2/_uniform_batch/stage3_ir.csv"
rows = list(csv.DictReader(open(P)))
hdr = list(rows[0].keys())
bad = [r["galaxy"] for r in rows
       if r.get("cog_flag") == "no-image"
       or "no-iras" in (r.get("note", ""), r.get("dist_flag", ""))
       or not r.get("iras_f60", "").strip()]
keep = [r for r in rows if r["galaxy"] not in bad]
with open(P, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr)
    w.writeheader()
    w.writerows(keep)
print(f"BAD={len(bad)} {' '.join(bad)}", flush=True)
