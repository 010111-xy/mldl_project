import gym
import numpy as np

from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def main():
  writer = SummaryWriter('basic_algorithm/tensor_board/domain_randomization')

  env = gym.make('CustomHopper-source-v0')
  # print(env.sim.model.body_names)
  # print(env.sim.model.body_mass[1])
  # print(env.sim.model.body_mass[2]) 
  # print(env.sim.model.body_mass[3])
  # print(env.sim.model.body_mass[4])
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
  thigh_mass = np.random.normal((mass_bounds['thigh']['low'] + mass_bounds['thigh']['high']) / 2, (mass_bounds['thigh']['high'] - mass_bounds['thigh']['low']) / 6)
  leg_mass = np.random.normal((mass_bounds['leg']['low'] + mass_bounds['leg']['high']) / 2, (mass_bounds['leg']['high'] + mass_bounds['leg']['low']) / 6)
  foot_mass = np.random.normal((mass_bounds['foot']['low'] + mass_bounds['foot']['high']) / 2, (mass_bounds['foot']['high'] + mass_bounds['foot']['low']) / 6)


  # Set the masses in the environment
  model.body_mass[2] = thigh_mass       # Thigh mass
  model.body_mass[3] = leg_mass         # Leg mass
  model.body_mass[4] = foot_mass        # Foot mass
  model.body_mass[1] = source_torso_mass  # Torso mass

  source_vec_env = make_vec_env('CustomHopper-source-v0', n_envs=4, monitor_dir="./logs/")

  # Create the PPO model
  model = PPO("MlpPolicy", source_vec_env, verbose=0)
  print('Train Start Time:', datetime.now())
  # Train the model
  model.learn(total_timesteps=1000000)
  print('Train End Time:', datetime.now())
  # Evaluate the policy in the source environment
  # mean_reward, std_reward = evaluate_policy(model, source_vec_env, n_eval_episodes=10)
  rewards = []

  print('Evaluation Start Time:', datetime.now())
  for episode in range(1000):
    episode_reward, _ = evaluate_policy(model, env, n_eval_episodes=1, return_episode_rewards=True)
    rewards.append(episode_reward[0])
    writer.add_scalar('GDR Training/Episode Reward', episode_reward[0], episode)

  print('Evaluation End Time:', datetime.now())
  print(f'Mean Reward: {np.mean(rewards)}')
  model.save("model/ppo_gdr.zip")
  writer.close()


def test():
  # Load the trained model
  trained_model = PPO.load("basic_algorithm/model/ppo_gdr.zip")

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
