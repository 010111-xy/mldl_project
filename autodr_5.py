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



# Set up the environment
original_env = gym.make('CustomHopper-v0')
wrapped_env = CustomEnvWrapper(original_env)
env = DummyVecEnv([lambda: wrapped_env])
model = PPO('MlpPolicy', env, verbose=0)

# ADR Parameters: Define custom ranges for thigh, leg, and foot masses, and joint frictions
phi_L = np.array([3.5, 2.3, 4.6, 0.001, 0.001, 0.001])  # Initial lower bounds
phi_H = np.array([4.5, 3.3, 5.5, 0.5, 0.5, 0.5])  # Initial upper bounds
phi_min = np.array([0.1, 0.1, 0.1, 0, 0, 0])
delta = 0.1 #step size
t_L, t_H = 900, 1200
m = 10

D_L = [[] for _ in range(6)]
D_H = [[] for _ in range(6)]
performance_history = []
speed_history = []
stability_history = []
balance_history = []
entropy_ADR_history = []


for episode in range(10000):
    #Sample parameters
    lambda_vec = np.random.uniform(phi_L, phi_H)
    #choose a random index
    i = np.random.choice(6)  # Randomize one parameter at a time
    x = np.random.rand()
    if x < 0.5:
        lambda_vec[i] = phi_L[i]
    else:
        lambda_vec[i] = phi_H[i]
    
    wrapped_env.set_env_parameters(lambda_vec)
    
    model.learn(total_timesteps=1000)
    reward, speed, stability, balance = evaluate_performance(env, model)
    
    # Update history
    performance_history.append(reward)
    speed_history.append(speed)
    stability_history.append(stability)
    balance_history.append(balance)
    entropy_ADR_history.append(np.mean(np.log(phi_H-phi_L)))
        
    # Determine which parameter to adjust based on performance
    if x < 0.5: #lowerbound
        D_L[i].append(reward)
        if len(D_L[i]) >= m:
            avg_reward = np.mean(D_L[i])
            D_L[i] = []
            if avg_reward >= t_H:
                phi_L[i] = max(phi_L[i]-delta, phi_min[i])
            elif avg_reward <= t_L:
                phi_L[i] += delta        
    else:
        D_H[i].append(reward)
        if len(D_H[i]) >= m:
            avg_reward = np.mean(D_H[i])
            D_H[i] = []
            if avg_reward >= t_H:
                phi_H[i] += delta
            elif avg_reward <= t_L:
                phi_H[i] -= delta
            # Ensure lambda stays within the current bounds
         #   lambda_vec[i] = np.clip(lambda_vec[i], phi_L[i], phi_H[i])
        #wrapped_env.set_env_parameters(lambda_vec)

    if episode % 10 == 0:
        print(f"Episode {episode}: Reward = {reward}, Parameters = {wrapped_env.env.model.body_mass.tolist() + wrapped_env.env.model.dof_frictionloss.tolist()}")

# Plotting and saving plots as PNG files
plt.figure(figsize=(12, 15))

plt.subplot(2, 1, 1)
plt.plot(performance_history, label='Total Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Performance of the Model Throughout Training Episodes')
plt.legend()
plt.savefig('performance_plot5.png')  # Save the plot as a PNG

# Plot Speed
plt.figure()
plt.plot(speed_history, label='Speed', color='blue')
plt.xlabel('Episode')
plt.ylabel('Speed')
plt.title('Speed Throughout Training Episodes')
plt.legend()
plt.tight_layout()
plt.savefig('speed_plot.png')  # Save the Speed plot
plt.close()  # Close the current figure

# Plot Stability
plt.figure()
plt.plot(stability_history, label='Stability', color='green')
plt.xlabel('Episode')
plt.ylabel('Stability')
plt.title('Stability Throughout Training Episodes')
plt.legend()
plt.tight_layout()
plt.savefig('stability_plot.png')  # Save the Stability plot
plt.close()  # Close the current figure

# Plot Balance
plt.figure()
plt.plot(balance_history, label='Balance', color='red')
plt.xlabel('Episode')
plt.ylabel('Balance')
plt.title('Balance Throughout Training Episodes')
plt.legend()
plt.tight_layout()
plt.savefig('balance_plot.png')  # Save the Balance plot
plt.close()

plt.figure()
plt.plot(entropy_ADR_history, label='ADR Entropy', color='red')
plt.xlabel('Episode')
plt.ylabel('Entropy')
plt.title('ADR entropy Throughout Training Episodes')
plt.legend()
plt.tight_layout()
plt.savefig('entropy_plot.png')  # Save the Balance plot
plt.close()
plt.show()