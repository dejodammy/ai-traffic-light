"""Generate REALISTIC demand for the bigger Ikeja junction:
  - vehicles spawn on the FAR-UPSTREAM edge of each approach (up to ~1.1 km away)
  - each approach splits across its THREE exits (left / through / right)
Usage:
  ..\\venv\\Scripts\\python.exe make_turning_demand.py --art 1400 --cross 280 --out lagos_big_peak.rou.xml
"""
import argparse

# approach -> (far-upstream spawn edge, [exit edges], is_arterial)
APPROACHES = {
    "artA":   ("610011092#1",  ["135617573#1", "134404770#4", "134404795#1"], True),
    "artB":   ("134404792#0",  ["134404795#1", "135195291#2", "135617573#1"], True),
    "crossA": ("136338240#0",  ["135195291#2", "135617573#1", "134404770#4"], False),
    "crossB": ("135617577#0",  ["134404770#4", "134404795#1", "135195291#2"], False),
}
SPLIT = [0.25, 0.55, 0.20]   # left / through / right proportions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--art", type=float, default=1400.0, help="veh/h per arterial approach")
    p.add_argument("--cross", type=float, default=280.0, help="veh/h per cross approach")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<routes>",
             f'  <!-- Realistic turning demand: arterial {args.art:.0f}, cross {args.cross:.0f} veh/h/approach;'
             ' cars spawn far upstream and split left/through/right. -->',
             '  <vType id="car" accel="2.6" decel="4.5" length="5.0" minGap="2.5" '
             'maxSpeed="13.9" departLane="random" lcImpatience="0.8"/>']
    for name, (origin, exits, is_art) in APPROACHES.items():
        total = args.art if is_art else args.cross
        for k, (exit_edge, frac) in enumerate(zip(exits, SPLIT)):
            rate = round(total * frac)
            if rate <= 0:
                continue
            lines.append(
                f'  <flow id="{name}_{k}" type="car" from="{origin}" to="{exit_edge}" '
                f'begin="0" end="3600" vehsPerHour="{rate}" departSpeed="max"/>')
    lines.append("</routes>")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
