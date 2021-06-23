import gym
import numpy as np
import tensorflow as tf
from policy_net import Policy_net
from ppo import PPOTrain
import matplotlib.pyplot as plt

ITERATION = int(2000)
GAMMA = 0.95


def main():
    env = gym.make('CartPole-v0')
    env.seed(0)
    S_DIMs = env.observation_space.shape[0] # S_DIMS 4
    A_DIMs = env.action_space.n # A_DIMS 2
    ob_space = env.observation_space # ob_space Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), float32)
    Policy = Policy_net('policy', S_DIMs, A_DIMs)
    Old_Policy = Policy_net('old_policy', S_DIMs, A_DIMs)
    PPO = PPOTrain(Policy, Old_Policy, gamma=GAMMA)
    all_ep_r = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        obs = env.reset()
        reward = 0

        for iteration in range(ITERATION):  # episode
            observations = []
            actions = []
            v_preds = []
            rewards = []

            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=True)
                act = np.ndarray.item(act) # Copy an element of an array to a standard Python scalar and return it.
                v_pred = np.ndarray.item(v_pred)

                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)

                next_obs, reward, done, info = env.step(act)

                if done:
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    obs = env.reset()
                    reward = -1
                    break
                else:
                    obs = next_obs

            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)
            rewards = np.array(rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()

            all_ep_r.append(sum(rewards))
            print("episode:", iteration, "   reward:", sum(rewards))
            PPO.assign_policy_parameters()

            inp = [observations, actions, rewards, v_preds_next, gaes]

            # train
            for epoch in range(4):
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=64)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          rewards=sampled_inp[2],
                          v_preds_next=sampled_inp[3],
                          gaes=sampled_inp[4])

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode');
    plt.ylabel('Moving averaged episode reward');
    plt.show()

if __name__ == '__main__':
    main()