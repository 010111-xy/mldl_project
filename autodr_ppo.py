import gym
from gym import Wrapper
import numpy as np
import matplotlib.pyplot as plt
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class CustomEnvWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def set_env_parameter(self, index, value):
    # Modify the environment based on the index and value
    # Here, you would implement the logic to change the desired parameter in the environment
    # For example, if 'index' corresponds to a specific physical property like leg mass:
        if index == 0:  # Assuming index 0 controls leg mass
            self.env.leg_mass = value  # Update the leg_mass attribute of the environment
        else:
            raise NotImplementedError("Invalid parameter index")

    def set_env_parameters(self, parameters):
        for index, value in enumerate(parameters):
            self.set_env_parameter(index, value)

# Training script
def evaluate_performance(env, model):
    obs = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward

original_env = gym.make('CustomHopper-v0')
wrapped_env = CustomEnvWrapper(original_env)
env = DummyVecEnv([lambda: wrapped_env])
model = PPO('MlpPolicy', env, verbose=1)

# ADR Parameters
phi_i_L = np.array([0.1] * env.observation_space.shape[0])
phi_i_H = np.array([0.9] * env.observation_space.shape[0])
phi = np.random.uniform(phi_i_L, phi_i_H)
delta = 0.05
t_L, t_H = 0.1, 0.9
m = 10

D_L = [[] for _ in range(env.observation_space.shape[0])]
D_H = [[] for _ in range(env.observation_space.shape[0])]
performance_history = []

for episode in range(10000):
    lambda_i = np.random.choice(len(phi))
    x = np.random.rand()
    if x < 0.5:
        env_param = phi_i_L[lambda_i]
        D_i = D_L
    else:
        env_param = phi_i_H[lambda_i]
        D_i = D_H
    
    wrapped_env.set_env_parameter(lambda_i, env_param)
    model.learn(total_timesteps=1000)
    p = evaluate_performance(env, model)
    D_i[lambda_i].append(p)
    performance_history.append(p)
    
    if len(D_i[lambda_i]) >= m:
        avg_p = np.mean(D_i[lambda_i])
        D_i[lambda_i] = []
        if avg_p >= t_H:
            phi[lambda_i] += delta
        elif avg_p <= t_L:
            phi[lambda_i] -= delta
        phi[lambda_i] = np.clip(phi[lambda_i], phi_i_L[lambda_i], phi_i_H[lambda_i])
    wrapped_env.set_env_parameters(phi)
    if episode % 100 == 0:
        print(f"Episode {episode}: Performance = {p}")

plt.figure(figsize=(10, 6))
plt.plot(performance_history)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Performance of the Model Throughout Training Episodes')
plt.show()
