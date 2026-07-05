"""Generate an EXTREMELY random/chaotic traffic scenario on the Lagos intersection.

Random surges hit random directions at random times, with rates swinging from
near-empty to gridlock. This is a robustness test: the model never trained on
anything like this. Run from sumo_scenarios1/:
    ..\\venv\\Scripts\\python.exe make_chaos.py
"""
import random

random.seed()  # different chaos every run; set a number here for reproducibility

ROUTES = {
    "N_S": "n2c c2s", "S_N": "s2c c2n", "E_W": "e2c c2w", "W_E": "w2c c2e",
    "N_W": "n2c c2w", "S_E": "s2c c2e", "E_N": "e2c c2n", "W_S": "w2c c2s",
}

lines = []
lines.append('<?xml version="1.0" encoding="UTF-8"?>')
lines.append("<routes>")
lines.append('  <!-- EXTREMELY RANDOM chaos scenario: random surges, random directions. -->')
lines.append('  <vType id="car" accel="2.6" decel="4.5" length="5.0" minGap="2.5" '
             'maxSpeed="13.9" departLane="random" lcImpatience="0.9" lcPushy="0.7"/>')
for rid, edges in ROUTES.items():
    lines.append(f'  <route id="{rid}" edges="{edges}"/>')

# Build randomized flows in 60-second windows, sorted by begin time (SUMO requires this).
flow_id = 0
for begin in range(0, 3000, 60):
    end = begin + 60
    # each window randomly slams a few directions with wildly varying demand
    for rid in random.sample(list(ROUTES), k=random.randint(1, len(ROUTES))):
        # rate swings from a trickle (50) to total gridlock (1800) veh/hour
        rate = random.choice([50, 100, 200, 400, 800, 1200, 1800])
        if random.random() < 0.25:          # 25% chance of an extreme spike
            rate = random.randint(1500, 2500)
        lines.append(
            f'  <flow id="f{flow_id}" type="car" route="{rid}" '
            f'begin="{begin}" end="{end}" vehsPerHour="{rate}" '
            f'departLane="random" departSpeed="max"/>'
        )
        flow_id += 1

lines.append("</routes>")

with open("intersection_chaos.rou.xml", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")

print(f"Wrote intersection_chaos.rou.xml with {flow_id} random flows.")
