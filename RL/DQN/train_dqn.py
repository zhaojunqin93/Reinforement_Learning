import numpy as np
import tensorflow as tf
import gym
import time

from dqn_model import DQN

DEFAULT_ENV_NAME = "LunarLander-v2"
MEAN_REWARD_GOAL = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
REPLAY_MIN_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000

EPSILON_START = 1.0
EPSILON_FINAL = 0.02
EPSILON_DECAY_FRAMES = 10 ** 5

import collections

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, sess, model, epsilon=0.0):
        done_reward = None
        while self.state.shape[0] == 1:
            print('reset')
            self.reset()
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.choose_best_action(sess, model, self.state)

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

    def choose_best_action(self, sess, model, state):
        # print(np.array(state).reshape([1, len(state)]))
        q = sess.run(model.output, feed_dict={model.input_data: state.reshape([1, len(state)])})
        return np.argmax(q[0])

    def fit_batch(self, sess, model, batch):
        """Do one deep Q learning iteration.
        - model: The DQN
        batch:
        - start_states: numpy array of starting states
        - actions: numpy array of one-hot encoded actions corresponding to the start states
        - rewards: numpy array of rewards corresponding to the start states and actions
        - next_states: numpy array of the resulting states corresponding to the start states and actions
        - is_terminal: numpy boolean array of whether the resulting state is terminal
          """
        start_states, actions, rewards, is_terminal, next_states = batch
        actions_oh = np.zeros((actions.shape[0], self.env.action_space.n))
        actions_oh[np.arange(actions.shape[0]), actions] = 1

        next_Q_values = sess.run(model.output, feed_dict={model.input_data: next_states})
        # The Q values of the terminal states is 0 by definition, so override them
        next_Q_values[is_terminal] = 0
        # The Q values of each start state is the reward + gamma * the max next state Q value
        Q_values = rewards + GAMMA * np.max(next_Q_values, axis=1)
        Q_values = actions_oh * Q_values[:, None]
        sess.run(model.train_opt,
                 feed_dict={model.input_data: start_states, model.actions: actions_oh, model.q_values: Q_values})


def train():
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    env = gym.make(DEFAULT_ENV_NAME)
    model = DQN(env.observation_space.shape, env.action_space.n, LEARNING_RATE)
    state = env.reset()
    c = collections.Counter()
    buffer = ExperienceBuffer(REPLAY_BUFFER_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START
    total_rewards = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            frame_idx += 1
            epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_FRAMES)
            reward = agent.play_step(sess, model, epsilon)
            if reward is not None:
                total_rewards.append(reward)
                speed = (frame_idx - ts_frame) / (time.time() - ts)
                ts_frame = frame_idx
                ts = time.time()
                mean_reward = np.mean(total_rewards[-100:])
                print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                    frame_idx, len(total_rewards), mean_reward, epsilon,
                    speed
                ))
                if len(buffer) < REPLAY_MIN_SIZE:
                    continue
                batch = buffer.sample(BATCH_SIZE)
                print(len(batch[0]))
                agent.fit_batch(sess, model, batch)

train()