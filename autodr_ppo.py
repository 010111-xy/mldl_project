import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
import gym

# Custom function to evaluate the performance of the model
def evaluate_performance(env, model):
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward

# Initialize the environment and model
env = gym.make('CustomHopper-v0')  # Replace with your custom environment
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1)

# ADR Parameters
phi_i_L = np.array([0.1] * env.observation_space.shape[0])  # Example lower bound
phi_i_H = np.array([0.9] * env.observation_space.shape[0])  # Example upper bound
phi = np.random.uniform(phi_i_L, phi_i_H)
delta = 0.05
t_L, t_H = 0.1, 0.9
m = 10

# Buffers for performance data
D_L = [[] for _ in range(env.observation_space.shape[0])]
D_H = [[] for _ in range(env.observation_space.shape[0])]

# Track performance
performance_history = []

# Training loop with ADR
for episode in range(10000):  # Loop over training iterations
    # Sample environment parameters according to current phi
    lambda_i = np.random.choice(len(phi))
    x = np.random.rand()

    if x < 0.5:
        env_param = phi_i_L[lambda_i]  # Sample at the lower bound
        D_i = D_L
    else:
        env_param = phi_i_H[lambda_i]  # Sample at the upper bound
        D_i = D_H

    # Update the environment parameter
    env.set_env_parameter(lambda_i, env_param)  # Custom method to update environment parameter

    # Train the model on this environment instance
    model.learn(total_timesteps=1000)

    # Evaluate the performance on the environment with the current setting
    p = evaluate_performance(env, model)

    # Update the buffer with the new performance
    D_i[lambda_i].append(p)
    
    # Track the performance
    performance_history.append(p)

    # Check if the buffer length has reached m
    if len(D_i[lambda_i]) >= m:
        avg_p = np.mean(D_i[lambda_i])
        D_i[lambda_i] = []  # Clear the buffer

        # Adjust the environment parameter bounds based on performance
        if avg_p >= t_H:
            phi[lambda_i] += delta
        elif avg_p <= t_L:
            phi[lambda_i] -= delta

        # Ensure the parameters remain within a valid range
        phi[lambda_i] = np.clip(phi[lambda_i], phi_i_L[lambda_i], phi_i_H[lambda_i])

    # Update the environment with the new parameters
    env.set_env_parameters(phi)

    # Optional: Print progress
    if episode % 100 == 0:
        print(f"Episode {episode}: Performance = {p}")

# Plotting the performance over episodes
plt.figure(figsize=(10, 6))
plt.plot(performance_history)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Performance of the Model Throughout Training Episodes')
plt.show()
