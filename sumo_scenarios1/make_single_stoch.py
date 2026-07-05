"""STOCHASTIC demand around a MEAN (reviewer suggestion):
   Obafemi Awolowo Way = 1400 +/- 200 veh/h,  Allen Avenue = 280 +/- 100 veh/h.
   The rate is resampled every 120s window (Gaussian around the mean, clamped >=50).
   Fixed seed -> reproducible. Vehicle ARRIVAL timing within each window is already
   Poisson via SUMO's vehsPerHour insertion."""
import random

random.seed(11)

ROUTES = {
    "awolowo_a": ("610011092#2", "135617573#1", 1400, 200),
    "awolowo_b": ("134404792#3", "135195291#2", 1400, 200),
    "allen_a":   ("136338240#2", "134404770#4", 280, 100),
    "allen_b":   ("135617577#1", "134404795#1", 280, 100),
}

lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<routes>",
         '  <!-- Stochastic demand: rate ~ N(mean, sd) resampled every 120s (seed=11). -->',
         '  <vType id="car" accel="2.6" decel="4.5" length="5.0" minGap="2.5" '
         'maxSpeed="13.9" departLane="random" lcImpatience="0.8"/>']

fid = 0
for begin in range(0, 3600, 120):
    end = begin + 120
    for name, (frm, to, mean, sd) in ROUTES.items():
        rate = max(50, int(random.gauss(mean, sd)))
        lines.append(
            f'  <flow id="{name}_{fid}" type="car" from="{frm}" to="{to}" '
            f'begin="{begin}" end="{end}" vehsPerHour="{rate}" departSpeed="max"/>')
        fid += 1
lines.append("</routes>")

with open("lagos_single_stoch.rou.xml", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"Wrote lagos_single_stoch.rou.xml with {fid} flows (mean +/- noise).")
