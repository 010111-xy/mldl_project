import gym
import numpy as np

from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


def main():
  env = gym.make('CustomHopper-source-v0')
  print(env.sim.model.body_names)
  print(env.sim.model.body_mass[1])
  print(env.sim.model.body_mass[2]) 
  print(env.sim.model.body_mass[3])
  print(env.sim.model.body_mass[4])
  mass_bounds = {
    'thigh': {
      'low': 3.5,
      'high': 4.5
    },
    'leg': {
      'low': 2.25,
      'high': 3.25
    },
    'foot': {
      'low': 4.75,
      'high': 5.25
    }
  }

  target_torso_mass = env.sim.model.body_mass[1]
  source_torso_mass = target_torso_mass - 1.0

  source_env = gym.make('CustomHopper-source-v0')

  # Access the MuJoCo model
  model = source_env.unwrapped.sim.model

  # Randomize masses for thigh, leg, and foot
  thigh_mass = np.random.uniform(mass_bounds['thigh']['low'], mass_bounds['thigh']['high'])
  leg_mass = np.random.uniform(mass_bounds['leg']['low'], mass_bounds['leg']['high'])
  foot_mass = np.random.uniform(mass_bounds['foot']['low'], mass_bounds['foot']['high'])

  # Set the masses in the environment
  model.body_mass[2] = thigh_mass       # Thigh mass
  model.body_mass[3] = leg_mass         # Leg mass
  model.body_mass[4] = foot_mass        # Foot mass
  model.body_mass[1] = source_torso_mass  # Torso mass

  source_vec_env = make_vec_env('CustomHopper-source-v0', n_envs=4, monitor_dir="./logs/")

  # Create the PPO model
  model = PPO("MlpPolicy", source_vec_env, verbose=0)

  # Train the model
  model.learn(total_timesteps=1000000)

  # Evaluate the policy in the source environment
  mean_reward, std_reward = evaluate_policy(model, source_vec_env, n_eval_episodes=10)
  print(f"Mean reward in source environment: {mean_reward} +/- {std_reward}")

  model.save("task6_trained_model.zip")


def test():
  # Load the trained model
  trained_model = PPO.load("task6_trained_model.zip")

  # Create environments for source and target environments
  source_env = gym.make('CustomHopper-source-v0')
  target_env = gym.make('CustomHopper-target-v0')

  # Evaluate the trained policy on the source environment
  mean_reward_source, std_reward_source = evaluate_policy(trained_model, source_env, n_eval_episodes=10)
  print(f"Mean reward on source environment: {mean_reward_source} +/- {std_reward_source}")

  # Evaluate the trained policy on the target environment
  mean_reward_target, std_reward_target = evaluate_policy(trained_model, target_env, n_eval_episodes=10)
  print(f"Mean reward on target environment: {mean_reward_target} +/- {std_reward_target}")

if __name__ == '__main__':
  main()
