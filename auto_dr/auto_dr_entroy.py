import gym
from gym import Wrapper
import numpy as np
import matplotlib.pyplot as plt
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
from collections import deque

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
        elif index == 3:
            self.env.model.dof_frictionloss[self.env.model.joint_name2id('thigh_joint')] = value
        elif index == 4:
            self.env.model.dof_frictionloss[self.env.model.joint_name2id('leg_joint')] = value
        elif index == 5:
            self.env.model.dof_frictionloss[self.env.model.joint_name2id('foot_joint')] = value
        else:
            raise NotImplementedError("Invalid parameter index")
        
    def set_env_parameters(self, parameters):
        for index, value in enumerate(parameters):
            self.set_env_parameter(index, value)

# Training script
def evaluate_performance(env, model):
    obs = env.reset()
    total_reward = 0
    total_speed = 0
    total_stability = 0
    total_balance = 0
    done = False
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Extract the dictionary from the list
        info_dict = info[0] if isinstance(info, list) and len(info) > 0 else {}
        
        # Safely extract metrics from the dictionary
        total_reward += reward
        total_speed += info_dict.get('speed', 0)
        total_stability += info_dict.get('stability', 0)
        total_balance += info_dict.get('balance', 0)
        steps += 1
    

    avg_speed = total_speed / steps if steps > 0 else 0
    avg_stability = total_stability / steps if steps > 0 else 0
    avg_balance = total_balance / steps if steps > 0 else 0
    return total_reward, avg_speed, avg_stability, avg_balance

original_env = gym.make('CustomHopper-v0')
wrapped_env = CustomEnvWrapper(original_env)
env = DummyVecEnv([lambda: wrapped_env])
model = PPO('MlpPolicy', env, verbose=0)

parameters_num = 6
# ADR Parameters: Define custom ranges for thigh, leg, and foot masses
phi_i_L = np.array([3.5, 2.3, 4.6, 0.001, 0.001, 0.001])  # Lower bounds for masses and frictions
phi_i_H = np.array([4.5, 3.3, 5.5, 0.5, 0.5, 0.5])  # Upper bounds for masses and frictions

phi_min = np.array([0.1, 0.1, 0.1, 0, 0, 0])

#step size for adjusting phi_i1
delta = 0.1
#performance thresholds for adjusting bounds
t_L, t_H = 0.6, 1.0
# num samples required before adjusting bounds
m = 10

D_L = [[] for _ in range(parameters_num)]
D_H = [[] for _ in range(parameters_num)]
D_i = [[] for _ in range(parameters_num)]

performance_history = []
speed_history = []
stability_history = []
balance_history = []
entropy_ADR_history = []

best_param = [] # print best params
best_performance = -np.inf

print(f'Start AutoDR Training: {datetime.now()}')
writer = SummaryWriter('auto_dr/tensor_board6/')


for episode in range(10000):
    phi = np.random.uniform(phi_i_L, phi_i_H)
    # select parameter
    lambda_i = np.random.choice(parameters_num)
    # decide from lower or higher bound
    x = np.random.rand()
    if x < 0.5:
        phi[lambda_i] = phi_i_L[lambda_i]
        D_i[lambda_i] = D_L[lambda_i]
    else:
        phi[lambda_i] = phi_i_H[lambda_i]
        D_i[lambda_i] = D_H[lambda_i]
    
    wrapped_env.set_env_parameters(phi)
    model.learn(total_timesteps=1000)
    reward, avg_speed, avg_stability, avg_balance = evaluate_performance(env, model)

    with open("env_param_log3.txt", "a") as log_file:
       log_file.write(f"Episode {episode}: Reward = {reward}, Parameters = {phi}\n")
    
    performance_history.append(reward)
    speed_history.append(avg_speed)
    stability_history.append(avg_stability)
    balance_history.append(avg_balance)

    epsilon = 1e-8
    entropy = np.mean(np.log(np.maximum(phi_i_H - phi_i_L, epsilon)))
    D_i[lambda_i].append(entropy)

    writer.add_scalar(f'Episode Entropy ADR', entropy, episode)
    if reward > best_performance:
        best_performance = reward
        best_param = phi

    with open("lambda_log.txt3", "a") as log_file:
        log_file.write(f"Episode {episode}: Reward = {reward}, Entropy = {entropy}, lambda_i = {lambda_i}, phi = {phi[lambda_i]}, phi_i_L = {phi_i_L[lambda_i]}, phi_i_H = {phi_i_H[lambda_i]}\n")

    if len(D_i[lambda_i]) >= m:
        avg_entropy = np.mean(D_i[lambda_i])
        D_i[lambda_i] = []
        if x < 0.5:
            D_L[lambda_i] = []
        else:
            D_H[lambda_i] = []

        if avg_entropy >= t_H:
            # want to be more stable
            if x < 0.5:
                # increase lower bound
                phi_i_L[lambda_i] += delta
                action = f"phi_i_L[{lambda_i}] increased by {delta}"
            else:
                # decrease upper bound
                phi_i_H[lambda_i] = max(phi_i_H[lambda_i] - delta, phi_i_L[lambda_i] + epsilon)
                action = f"phi_i_H[{lambda_i}] decreased by {delta}"
            
        elif avg_entropy <= t_L:
            # need to explore enough
            if x < 0.5:
                # decrease lower bound
                phi_i_L[lambda_i] = max(phi_i_L[lambda_i] - delta, phi_min[lambda_i])
                action = f"phi_i_L[{lambda_i}] decreased by {delta}"
            else:
                # increase upper bound
                phi_i_H[lambda_i] += delta
                action = f"phi_i_H[{lambda_i}] increased by {delta}"

        with open("update_bound_log3.txt", "a") as log_file:
            log_file.write(f"Episode {episode}: lambda_i = {lambda_i}, phi_i_L = {phi_i_L[lambda_i]}, phi_i_H = {phi_i_H[lambda_i]}, Action: {action}\n")
 

    writer.add_scalar(f'Episode Performance', reward, episode)
    writer.add_scalar(f'Episode Average Speed', avg_speed, episode)
    writer.add_scalar(f'Episode Average Stability', avg_stability, episode)
    writer.add_scalar(f'Episode Average Balance', avg_balance, episode)
    
    if episode % 10 == 0:
        print(f"Episode {episode}: Reward = {reward}, Parameters = {phi}")

print(f'Best performance: {best_performance}')
print(f'Best parameters: {best_param}')

writer.close()