import numpy as np
from rl_env import SumoRLEnv, EnvConfig

def main():
    cfg = EnvConfig(
        sumocfg="ideal.sumocfg",
        tls_id="c",
        decision_interval=10,
        episode_steps=200,
        use_gui=True,     # set False for speed
        seed=42,
        use_yellow=True,
        yellow_duration=3,
    )

    env = SumoRLEnv(cfg)
    state = env.reset()
    print("Initial state shape:", state.shape)  # should be (12,)

    done = False
    total_reward = 0.0

    while not done:
        action = np.random.randint(0, 2)  # random 0 or 1
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if info["step"] % 50 == 0:
            print(f"step={info['step']}, action={action}, phase={info['phase']}, total_queue={info['total_queue']:.0f}, reward={reward:.2f}")

    env.close()
    print("Episode finished. Total reward:", total_reward)

if __name__ == "__main__":
    main()
