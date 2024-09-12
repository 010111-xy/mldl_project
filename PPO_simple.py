import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create the custom environment
env = gym.make('CustomHopper-v0')

# Wrap the environment to log episode rewards
env = Monitor(env, filename="./ppo_customhopper_logs")

# Define the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the PPO model
model.learn(total_timesteps=1_000_000)  # Adjust timesteps as needed

# Save the trained model
model.save("ppo_customhopper_model")

# Load the log data
log_data = pd.read_csv("./ppo_customhopper_logs.monitor.csv", comment="#")
episode_rewards = log_data['r']  # The 'r' column contains episode rewards

# Calculate moving average with a window size
window_size = 100
moving_avg = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')

# Plotting the episode rewards and moving average
plt.figure(figsize=(10, 6))

# Plot the raw episode rewards with light color for shading
plt.plot(episode_rewards, color='lightblue', label='Episode Rewards', alpha=0.5)

# Plot the moving average with a darker color
plt.plot(np.arange(window_size - 1, len(episode_rewards)), moving_avg, color='blue', label=f'{window_size}-Episode Moving Average')

# Labels and title
plt.title('Training Performance of PPO on CustomHopper-v0')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.legend()

# Show the plot
plt.show()

# Calculate the final average reward and standard deviation
final_avg_reward = np.mean(episode_rewards[-window_size:])
final_std_reward = np.std(episode_rewards[-window_size:])

print(f"Final Average Reward: {final_avg_reward:.2f}")
print(f"Standard Deviation of Final Rewards: {final_std_reward:.2f}")
