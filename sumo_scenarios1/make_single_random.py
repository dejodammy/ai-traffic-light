"""Generate a RANDOM/variable demand scenario routed through the faithful single
Ikeja junction (works for both the 2-phase and 4-phase nets, same edges).
Demand on each of the 4 approaches swings randomly over time. Fixed seed = reproducible."""
import random

random.seed(7)  # fixed -> the random scenario is stable/reproducible

ROUTES = {
    "art_a":  ("610011092#2", "135617573#1"),
    "art_b":  ("134404792#3", "135195291#2"),
    "cross_a":("136338240#2", "134404770#4"),
    "cross_b":("135617577#1", "134404795#1"),
}

lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<routes>",
         '  <!-- RANDOM variable demand on the faithful Ikeja junction (seed=7). -->',
         '  <vType id="car" accel="2.6" decel="4.5" length="5.0" minGap="2.5" '
         'maxSpeed="13.9" departLane="random" lcImpatience="0.8"/>']

fid = 0
for begin in range(0, 1800, 90):                 # 90-second windows
    end = begin + 90
    for name, (frm, to) in ROUTES.items():
        # each approach independently swings from light to heavy each window
        rate = random.choice([100, 250, 450, 700, 1000, 1400])
        if random.random() < 0.2:                # occasional surge
            rate = random.randint(1200, 1800)
        lines.append(
            f'  <flow id="r{fid}" type="car" from="{frm}" to="{to}" '
            f'begin="{begin}" end="{end}" vehsPerHour="{rate}" departSpeed="max"/>')
        fid += 1
lines.append("</routes>")

with open("lagos_single_random.rou.xml", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"Wrote lagos_single_random.rou.xml with {fid} random flows.")
