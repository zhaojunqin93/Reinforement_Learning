import gym
from Policy_Gradient import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('LunarLander-v2')
env = env.unwrapped

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.95
)
MAX_EP_STEPS = 200
all_ep_r = []

for i_episode in range(1000):

    observation = env.reset()
    t = 0

    while True:
        # env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        t += 1
        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(RL.ep_rs)

            # if 'running_reward' not in globals():
            #     running_reward = ep_rs_sum
            # else:
            #     running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

            all_ep_r.append(ep_rs_sum)

            print("episode:", i_episode, "  reward:", int(ep_rs_sum))

            vt = RL.learn()

            break

        observation = observation_

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode')
plt.ylabel('Moving averaged episode reward')
plt.show()

# RL.plot_cost()