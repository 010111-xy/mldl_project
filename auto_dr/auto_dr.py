def train_with_autodr(env, agent, num_episodes, target_performance, expansion_rate):
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # agent.update(state, action, reward, next_state, done)
            agent.learn(total_timesteps=1)
            state = next_state
            episode_reward += reward
        
        # Update randomization boundaries
        env.envs[0].expand_boundaries(episode_reward, target_performance, expansion_rate)
        
        # Log progress
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")
            for param, range_dict in env.envs[0].randomization_params.items():
                print(f"  {param}: {range_dict['current_range']}")