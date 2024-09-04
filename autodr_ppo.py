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
        if index == 0:  # Thigh mass
            self.env.model.body_mass[self.env.model.body_name2id('thigh')] = value
        elif index == 1:  # Leg mass
            self.env.model.body_mass[self.env.model.body_name2id('leg')] = value
        elif index == 2:  # Foot mass
            self.env.model.body_mass[self.env.model.body_name2id('foot')] = value
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
model = PPO('MlpPolicy', env, verbose=0)

# ADR Parameters: Define custom ranges for thigh, leg, and foot masses
phi_i_L = np.array([3.9, 2.7, 4.9])  # Lower bounds for thigh, leg, and foot masses
phi_i_H = np.array([4.1, 2.9, 5.1])  # Upper bounds for thigh, leg, and foot masses
phi = np.random.uniform(phi_i_L, phi_i_H)
#step size for adjusting phi_i
delta = 0.05
#performance thresholds for adjusting bounds
t_L, t_H = 0.1, 0.9
# num samples required before adjusting bounds
m = 10

D_L = [[] for _ in range(len(phi))]
D_H = [[] for _ in range(len(phi))]
performance_history = []

for episode in range(10000):
    lambda_i = np.random.choice(len(phi)) #here i am only randomizing one parameter at once
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
