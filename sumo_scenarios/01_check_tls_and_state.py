import traci

SUMOCFG = "ideal.sumocfg"

def pick_target_tls(tls_ids):
    # Prefer junction id "c" if it exists (since our center node is c)
    if "c" in tls_ids:
        return "c"
    return tls_ids[0]

def main():
    traci.start(["sumo-gui", "-c", SUMOCFG, "--start", "--quit-on-end"])

    tls_ids = traci.trafficlight.getIDList()
    print("\n=== TLS IDs FOUND ===")
    for t in tls_ids:
        print(" -", t)

    if not tls_ids:
        print("\nNo traffic lights detected. Check node type is traffic_light and rebuild net.")
        traci.close()
        return

    tls_id = pick_target_tls(tls_ids)
    print(f"\nTarget TLS: {tls_id}")

    controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
    # remove duplicates while keeping order
    unique_lanes = list(dict.fromkeys(controlled_lanes))

    print("\n=== Controlled Lanes (State Inputs) ===")
    for ln in unique_lanes:
        print(" -", ln)

    print("\n=== Initial Queue (Halting Vehicles) per Lane ===")
    for ln in unique_lanes:
        q = traci.lane.getLastStepHaltingNumber(ln)
        print(f"{ln}: {q}")

    print("\n=== Initial Waiting Time (seconds) per Lane ===")
    for ln in unique_lanes:
        w = traci.lane.getWaitingTime(ln)
        print(f"{ln}: {w:.2f}")

    traci.close()

if __name__ == "__main__":
    main()
