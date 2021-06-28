import gym
from DQN_3layers import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt
import random

MAX_EP_STEPS = 1000

env = gym.make('LunarLander-v2')
env = env.unwrapped
env.seed(0)
random.seed(0)
print("action_space", env.action_space.n)
print("observation_space", env.observation_space.shape)

RL = DeepQNetwork(n_actions=env.action_space.n,
				  n_features=env.observation_space.shape[0],
				  learning_rate=5e-4,
				  e_greedy=0.9,
				  replace_target_iter=100,
				  memory_size=2000,
				  e_greedy_increment=0.001)

total_steps = 0
all_ep_r = []

for i_episode in range(1000):
	observation = env.reset()
	ep_r = 0
	t = 0

	while True:
		# if i_episode % 50 == 0:
		# env.render()

		action = RL.choose_action(observation)
		# print("action",action)

		observation_, reward, done, info = env.step(action)
		#
		# print("observation_", observation_)
		# print("reward", reward)
		# print("done", done)
		RL.store_transition(observation, action, reward, observation_)

		ep_r +=reward

		if total_steps > 1000:
			RL.learn()
		observation = observation_
		t += 1

		if done or t >= MAX_EP_STEPS:
			ep_rs_sum = ep_r

			all_ep_r.append(ep_r)
			print('episode: ', i_episode,
				  'ep_r: ', round(ep_r, 2))
			break

		total_steps += 1

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode')
plt.ylabel('Moving averaged episode reward')
plt.show()

# RL.plot_cost()