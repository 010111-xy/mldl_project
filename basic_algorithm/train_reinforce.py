"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym

from datetime import datetime
from basic_algorithm.reinforce_normal import REINFORCE as REINFORCE_NORMAL, PolicyNetwork as REINFORCE_NORMAL_POLICY
from basic_algorithm.reinforce_xavier import REINFORCE as REINFORCE_XAVIER, PolicyNetwork as REINFORCE_XAVIER_POLICY
from env.custom_hopper import *
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--model-type', default='REINFORCE_NORMAL', type=str, help='model type [REINFORCE_NORMAL, REINFORCE_XAVIER]')

    return parser.parse_args()

args = parse_args()


def main():

	writer = SummaryWriter('basic_algorithm/tensor_board/reinforce')
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

	if args.model_type == 'REINFORCE_NORMAL':
		policy = REINFORCE_NORMAL_POLICY(observation_space_dim, action_space_dim)
		agent = REINFORCE_NORMAL(policy, device=args.device)
		print('----REINFORCE_NORMAL---')
	else:
		policy = REINFORCE_XAVIER_POLICY(observation_space_dim, action_space_dim)
		agent = REINFORCE_XAVIER(policy, device=args.device)
		print('----REINFORCE_XAVIER---')


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
		for name, param in agent.policy.named_parameters():
			writer.add_histogram(f'{args.model_type} Training/Policy', param, episode)
	
		for i, param_group in enumerate(agent.optimizer.param_groups):
			writer.add_scalar(f'{args.model_type} Training/Learning Rate/group_{i}', param_group['lr'], episode)
		
		if (episode+1)%args.print_every == 0:
			print('Time:', datetime.now())
			print('Training episode:', episode + 1)
			print('Episode return:', train_reward)

	writer.close()
	torch.save(agent.policy.state_dict(), f'basic_algorithm/model/{args.model_type}_model.mdl')

	

if __name__ == '__main__':
	main()