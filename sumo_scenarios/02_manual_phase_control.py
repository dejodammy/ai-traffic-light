import traci
import time

SUMOCFG = "ideal.sumocfg"

def pick_target_tls(tls_ids):
    if "c" in tls_ids:
        return "c"
    return tls_ids[0]

def total_queue(lanes):
    return sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)

def main():
    traci.start(["sumo-gui", "-c", SUMOCFG, "--start", "--quit-on-end"])

    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids:
        print("No TLS found.")
        traci.close()
        return

    tls_id = pick_target_tls(tls_ids)
    lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))

    print(f"Controlling TLS: {tls_id}")
    print(f"Controlled lanes: {len(lanes)} lanes")

    # Inspect available phases
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    num_phases = len(logic.phases)
    print(f"Phases available in current program: {num_phases}")

    # Simple action loop: alternate between phase 0 and phase 2 if possible
    # (Many default programs follow 0: NS green, 1: yellow, 2: EW green, 3: yellow)
    action_phases = [0, 2] if num_phases >= 3 else [0]

    step = 0
    action_index = 0
    hold_steps = 30  # hold each green for 30 seconds

    # run for 600 seconds (10 minutes)
    SIM_STEPS = 600

    prev_q = None

    while step < SIM_STEPS:
        # change phase every hold_steps
        if step % hold_steps == 0:
            phase = action_phases[action_index % len(action_phases)]
            traci.trafficlight.setPhase(tls_id, phase)
            print(f"\n[step={step}] setPhase -> {phase}")
            action_index += 1

        traci.simulationStep()

        # compute queue-based reward example
        q = total_queue(lanes)
        if prev_q is None:
            reward = 0
        else:
            reward = prev_q - q  # positive if queue decreases
        prev_q = q

        if step % 10 == 0:
            print(f"[step={step}] total_queue={q}, reward={reward}")

        step += 1

    traci.close()

if __name__ == "__main__":
    main()
