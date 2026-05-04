import numpy as np
import matplotlib.pyplot as plt

data = np.load("training_logs.npz")
rewards = data["rewards"]
avg_q = data["avg_queue"]
avg_w = data["avg_wait"]

plt.figure()
plt.plot(rewards)
plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

plt.figure()
plt.plot(avg_q)
plt.title("Average Queue per Episode")
plt.xlabel("Episode")
plt.ylabel("Vehicles (halting)")
plt.show()

plt.figure()
plt.plot(avg_w)
plt.title("Average Waiting Time per Episode")
plt.xlabel("Episode")
plt.ylabel("Seconds")
plt.show()
