import gym
from gym import Wrapper
import numpy as np
import matplotlib.pyplot as plt
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import torch
import torch.optim as optim
from rnd_network import RNDNetwork
import pandas as pd

class CustomEnvWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def set_env_parameter(self, index, value):
        if index == 0:  # Thigh mass
            self.env.model.body_mass[self.env.model.body_name2id('thigh')] = value
        elif index == 1:  # Leg mass
            self.env.model.body_mass[self.env.model.body_name2id('leg')] = value
        elif index == 2:  # Foot mass
            self.env.model.body_mass[self.env.model.body_name2id('foot')] = value
        elif index == 3:  # Thigh joint friction
            self.env.model.dof_frictionloss[self.env.model.joint_name2id('thigh_joint')] = value
        elif index == 4:  # Leg joint friction
            self.env.model.dof_frictionloss[self.env.model.joint_name2id('leg_joint')] = value
        elif index == 5:  # Foot joint friction
            self.env.model.dof_frictionloss[self.env.model.joint_name2id('foot_joint')] = value
        else:
            raise NotImplementedError("Invalid parameter index")
        
    def set_env_parameters(self, parameters):
        for index, value in enumerate(parameters):
            self.set_env_parameter(index, value)

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
        
        info_dict = info[0] if isinstance(info, list) and len(info) > 0 else {}
        
        total_reward += reward
        total_speed += info_dict.get('speed', 0)
        total_stability += info_dict.get('stability', 0)
        total_balance += info_dict.get('balance', 0)
        steps += 1
    
    avg_speed = total_speed / steps if steps > 0 else 0
    avg_stability = total_stability / steps if steps > 0 else 0
    avg_balance = total_balance / steps if steps > 0 else 0
    return total_reward, avg_speed, avg_stability, avg_balance


def compute_intrinsic_reward(state, target_network, predictor_network):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    
    # Compute target and predictor outputs
    target_output = target_network(state_tensor).detach()  # Detach to avoid gradient updates
    predictor_output = predictor_network(state_tensor)
    
    # Compute the intrinsic reward (RND prediction error)
    intrinsic_reward = torch.mean((target_output - predictor_output) ** 2).item()
    
    return intrinsic_reward


def update_predictor_network(state, target_network, predictor_network, optimizer):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    target_output = target_network(state_tensor)
    predictor_output = predictor_network(state_tensor)
    
    loss = torch.mean((target_output - predictor_output) ** 2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Set up the environment
original_env = gym.make('CustomHopper-v0')
wrapped_env = CustomEnvWrapper(original_env)
env = DummyVecEnv([lambda: wrapped_env])
model = PPO('MlpPolicy', env, verbose=0)

# Initialize RND networks
obs_space = original_env.observation_space.shape[0]
target_network = RNDNetwork(obs_space)
for param in target_network.parameters():
    param.requires_grad = False

predictor_network = RNDNetwork(obs_space)
optimizer = optim.Adam(predictor_network.parameters(), lr=1e-4)


# ADR Parameters
phi_L = np.array([3.5, 2.3, 4.6, 0.001, 0.001, 0.001])
phi_H = np.array([4.5, 3.3, 5.5, 0.5, 0.5, 0.5])
phi_min = np.array([0.1, 0.1, 0.1, 0, 0, 0])
delta = 0.1 
t_L, t_H = 80, 175   # Adjust these thresholds as needed
m = 10

D_L = [[] for _ in range(6)]
D_H = [[] for _ in range(6)]
performance_history = []
speed_history = []
stability_history = []
balance_history = []
entropy_ADR_history = []
intrinsic_reward_history = []
intrinsic_reward = 0

# Open file to save reward and parameters
with open('reward_and_parameters_intrinsic_80175.txt', 'w') as f:
    for episode in range(10000):
        lambda_vec = np.random.uniform(phi_L, phi_H)
        i = np.random.choice(6)
        x = np.random.rand()
        if x < 0.5:
            lambda_vec[i] = phi_L[i]
        else:
            lambda_vec[i] = phi_H[i]
        
        wrapped_env.set_env_parameters(lambda_vec)

        obs = wrapped_env.reset()
        total_intrinsic_reward = 0
        
        for step in range(1000):  # Rollout for 1000 timesteps per episode
            action, _ = model.predict(obs, deterministic=True)
            new_obs, extrinsic_reward, done, info = wrapped_env.step(action)
            
            # Compute intrinsic reward using RND
            timestep_intrinsic_reward = compute_intrinsic_reward(new_obs, target_network, predictor_network)
            intrinsic_reward += timestep_intrinsic_reward
            # Update predictor network with the new state
            update_predictor_network(new_obs, target_network, predictor_network, optimizer)
            
            obs = new_obs
            if done:
                break
        
        model.learn(total_timesteps=1000)
        reward, speed, stability, balance = evaluate_performance(env, model)
        ## Entropy calculation with a small epsilon to avoid NaN
        epsilon = 1e-8
        entropy = np.mean(np.log(np.maximum(phi_H - phi_L, epsilon)))

        intrinsic_reward_history.append(intrinsic_reward)                  
        performance_history.append(reward)
        speed_history.append(speed)
        stability_history.append(stability)
        balance_history.append(balance)
        entropy_ADR_history.append(entropy)
            
        if x < 0.5: 
            D_L[i].append(intrinsic_reward)
            if len(D_L[i]) >= m:
                avg_intrinsic_r = np.mean(D_L[i])
                D_L[i] = []
                if avg_intrinsic_r >= t_H:
                    phi_L[i] += delta 
                elif avg_intrinsic_r <= t_L:
                    phi_L[i] = max(phi_L[i] - delta, phi_min[i])
        else:
            D_H[i].append(intrinsic_reward)
            if len(D_H[i]) >= m:
                avg_intrinsic_r= np.mean(D_H[i])
                D_H[i] = []
                if avg_intrinsic_r >= t_H:
                    phi_H[i] = max(phi_H[i] - delta, phi_L[i] + epsilon)  # Ensure phi_H > phi_L
                elif avg_intrinsic_r <= t_L:
                    phi_H[i] += delta
        # Create the log message
        log_message = f"Episode {episode}: Reward = {reward}, Intrisic Reward = {intrinsic_reward}, Entropy = {entropy}, Parameters = {wrapped_env.env.model.body_mass.tolist() + wrapped_env.env.model.dof_frictionloss.tolist()}\n"
        
        # Write to the file
        f.write(log_message)
        
        # Print to the terminal
        print(log_message, end='')  # `end=''` prevents adding an extra newline

# Function to plot with shading
def plot_with_shading(x, y, color_line, color_fill, label, ylabel, title):
    y_mean = pd.Series(y).rolling(window=50).mean()
    y_std = pd.Series(y).rolling(window=50).std()
    plt.plot(x, y_mean, color=color_line, label=label)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=color_fill, alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

# Plot all metrics
episodes = np.arange(len(performance_history))

# Plot Total Reward
plt.figure()
plot_with_shading(episodes, performance_history, 'darkblue', 'lightblue', 'Total Reward', 'Reward', 'Performance of the Model')
plt.tight_layout()
plt.savefig('performance_intrinsic_80175.png')

# Plot Speed
plt.figure()
plot_with_shading(episodes, speed_history, 'darkgreen', 'lightgreen', 'Speed', 'Speed', 'Speed Throughout Training')
plt.tight_layout()
plt.savefig('speed_intrinsic_80175.png')

# Plot Stability
plt.figure()
plot_with_shading(episodes, stability_history, 'darkorange', 'navajowhite', 'Stability', 'Stability', 'Stability Throughout Training')
plt.tight_layout()
plt.savefig('stability_intrinsic_80175.png')

# Plot Balance
plt.figure()
plot_with_shading(episodes, balance_history, 'darkred', 'lightcoral', 'Balance', 'Balance', 'Balance Throughout Training')
plt.tight_layout()
plt.savefig('balance_intrinsic_80175.png')

# Plot ADR entropy
plt.figure()
plot_with_shading(episodes, entropy_ADR_history, 'purple', 'plum', 'ADR Entropy', 'Entropy', 'ADR Entropy Throughout Training')
plt.tight_layout()
plt.savefig('entropy_intrinsic_80175.png')

# Plot ADR entropy
plt.figure()
plot_with_shading(episodes, intrinsic_reward_history, 'teal', 'lightcyan', 'Intrinsic Reward', 'Intrinsic Reward', 'Intrinsic Reward Throughout Training')
plt.tight_layout()
plt.savefig('intrinsic_reward_intrinsic_80175.png')

plt.show()
