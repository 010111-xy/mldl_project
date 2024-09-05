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

# ADR Parameters: Define custom ranges for thigh, leg, and foot masses
phi_i_L = np.array([3.8, 2.7, 5, 0.01, 0.01, 0.01])  # Lower bounds for masses and frictions
phi_i_H = np.array([4.9, 3.6, 6.5, 0.5, 0.5, 0.5])  # Upper bounds for masses and frictions
phi = np.random.uniform(phi_i_L, phi_i_H)


# phi_min = np.array([3.0, 2.0, 4.0, 0, 0, 0])
phi_max = np.array([12.5, 12.5, 12.5, 8, 8, 8])

#step size for adjusting phi_i1
delta = 0.1
#performance thresholds for adjusting bounds
t_L, t_H = 0.4, 0.6
# num samples required before adjusting bounds
m = 10

D_L = [[] for _ in range(len(phi))]
D_H = [[] for _ in range(len(phi))]
performance_history = []
speed_history = []
stability_history = []
balance_history = []

best_phi = phi.copy() # print best params
best_performance = -np.inf

print(f'Start AutoDR Training: {datetime.now()}')
writer = SummaryWriter('auto_dr/tensor_board3/')


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
    reward, avg_speed, avg_stability, avg_balance = evaluate_performance(env, model)

    D_i[lambda_i].append(reward)
    performance_history.append(reward)
    speed_history.append(avg_speed)
    stability_history.append(avg_stability)
    balance_history.append(avg_balance)

    if reward > best_performance:
        best_performance = reward
        best_phi = phi.copy()

    if len(D_i[lambda_i]) >= m:
        avg_p = np.mean(D_i[lambda_i])
        D_i[lambda_i] = []

        if avg_p < best_performance * 0.9:
            delta *= 0.5
        else:
            delta = min(delta * 1.05, 0.1)

        if avg_p >= t_H:
            phi[lambda_i] += delta
            phi_i_H[lambda_i] += delta
        elif avg_p <= t_L:
            phi[lambda_i] -= delta
            phi_i_L[lambda_i] -= delta
        
        # Ensure phi stays within the current bounds
        # phi[lambda_i] = np.clip(phi[lambda_i], phi_i_L[lambda_i], phi_i_H[lambda_i])
        phi[lambda_i] = np.clip(phi[lambda_i], phi_i_L[lambda_i], min(phi_i_H[lambda_i], phi_max[lambda_i]))


    wrapped_env.set_env_parameters(phi)

    writer.add_scalar(f'Episode Performance', reward, episode)
    writer.add_scalar(f'Episode Average Speed', avg_speed, episode)
    writer.add_scalar(f'Episode Average Stability', avg_stability, episode)
    writer.add_scalar(f'Episode Average Balance', avg_balance, episode)

    with open("phi_log.txt", "a") as log_file:
        log_file.write(f"Episode {episode}: Reward = {reward}, lambda_i = {lambda_i}, phi = {phi[lambda_i]}, phi_i_L = {phi_i_L[lambda_i]}, phi_i_H = {phi_i_H[lambda_i]}\n")
    
    with open("param_log.txt", "a") as log_file:
        log_file.write(f"Episode {episode}: Reward = {reward}, Parameters = {wrapped_env.env.model.body_mass.tolist() + wrapped_env.env.model.dof_frictionloss.tolist()}\n")
    
    if episode % 10 == 0:
        print(f"Episode {episode}: Reward = {reward}, Parameters = {wrapped_env.env.model.body_mass.tolist() + wrapped_env.env.model.dof_frictionloss.tolist()}")

print(f'Best performance: {best_performance}')
print(f'Best parameters: {best_phi}')

writer.close()