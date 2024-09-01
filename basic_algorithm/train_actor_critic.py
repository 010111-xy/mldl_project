"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym

from datetime import datetime
from basic_algorithm.actor_critic_normal import Agent as AC_NORAML_AGENT, Policy as AC_NORAML_POLICY
from basic_algorithm.actor_critic_xavier import Agent as AC_XAVIER_AGENT, Policy as AC_XAVIER_POLICY
from env.custom_hopper import *
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--model-type', default='AC_XAVIER', type=str, help='model type [AC_NORMAL, AC_XAVIER]')

    return parser.parse_args()

args = parse_args()


def main():

	writer = SummaryWriter('basic_algorithm/tensor_board/actor-critic')
	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')
	print('Train Start Time:', datetime.now())
	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	print('Model type', args.model_type)

	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	if args.model_type == 'AC_NORMAL':
		policy = AC_NORAML_POLICY(observation_space_dim, action_space_dim)
		agent = AC_NORAML_AGENT(policy, device=args.device)
		print('----AC_NORMAL---')
	else:
		policy = AC_XAVIER_POLICY(observation_space_dim, action_space_dim)
		agent = AC_XAVIER_AGENT(policy, device=args.device)
		print('----AC_XAVIER---')


	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward
		
		# update the policy
		agent.update_policy()

		writer.add_scalar(f'{args.model_type} Training/Episode Reward Return', train_reward, episode)
		
		
		if (episode+1)%args.print_every == 0:
			print('Time:', datetime.now())
			print('Training episode:', episode + 1)
			print('Episode return:', train_reward)

	writer.close()
	torch.save(agent.policy.state_dict(), f'basic_algorithm/model/{args.model_type}_model.mdl')

	

if __name__ == '__main__':
	main()