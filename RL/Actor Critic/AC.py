import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

np.random.seed(2)
tf.set_random_seed(2)

OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200
MAX_EP_STEPS = 200
RENDER = False
GAMMA = 0.9

env = gym.make('CartPole-v1')
env.seed(1)
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_F_List = env.observation_space
N_A = env.action_space.n
N_A_List = env.action_space

class Actor(object):
	def __init__(self, sess, n_inputs, n_actions, a_lr = 0.001):
		self.sess = sess

		self.s = tf.placeholder(tf.float32, [1, n_inputs], name='state')
		self.a = tf.placeholder(tf.int32, None, name='action')
		self.td_error = tf.placeholder(tf.float32, None, name='td_error')

		with tf.variable_scope('Actor'):
			l1 = tf.layers.dense(
				inputs=self.s,
				units=32,
				activation=tf.nn.relu,
				kernel_initializer=tf.random_normal_initializer(0., .1),
				bias_initializer=tf.constant_initializer(0.1),
				name='l1'
			)

			self.acts_prob = tf.layers.dense(
				inputs=l1,
				units=n_actions,
				activation=tf.nn.softmax,
				kernel_initializer=tf.random_normal_initializer(0., .1),
				bias_initializer=tf.constant_initializer(0.1),
				name='acts_prob'
			)

		with tf.variable_scope('exp_v'):
			log_prob = tf.log(self.acts_prob[0, self.a])
			self.exp_v = tf.reduce_mean(log_prob * self.td_error)

		with tf.variable_scope('train'):
			self.train_op = tf.train.AdamOptimizer(a_lr).minimize(-self.exp_v)

	def choose_action(self, s):
		s = s[np.newaxis, :]
		probs = self.sess.run(self.acts_prob, {self.s: s})
		return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

	def learn(self, s, a, td_error):
		s = s[np.newaxis, :]
		feed_dict = {self.s: s, self.a: a, self.td_error: td_error}
		_, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
		return exp_v

class Critic(object):
	def __init__(self, sess, n_inputs, c_lr = 0.01):
		self.sess = sess

		self.s = tf.placeholder(tf.float32, [1, n_inputs], name='state')
		self.v_ = tf.placeholder(tf.float32, [1, 1], name='v_next')
		self.r = tf.placeholder(tf.float32, None, name='reward')

		with tf.variable_scope('Critic'):
			l1 = tf.layers.dense(
				inputs=self.s,
				units=64,
				activation=tf.nn.relu,
				kernel_initializer=tf.random_normal_initializer(0., .1),
				bias_initializer=tf.constant_initializer(0.1),
				name='l1'
			)

			self.v = tf.layers.dense(
				inputs=l1,
				units=1,
				activation=None,
				kernel_initializer=tf.random_normal_initializer(0., .1),
				bias_initializer=tf.constant_initializer(0.1),
				name='value'
			)

		with tf.variable_scope('squared_TD_error'):
			self.td_error = self.r + GAMMA * self.v_ - self.v
			self.loss = tf.square(self.td_error)

		with tf.variable_scope('train'):
			self.train_op = tf.train.AdamOptimizer(c_lr).minimize(self.loss)

	def learn(self, s, r, s_):
		s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

		v_ = self.sess.run(self.v, {self.s: s_})

		td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s, self.v_: v_, self.r: r})
		# print("td_error", td_error)
		return td_error

sess = tf.Session()

actor = Actor(sess, n_inputs=N_F, n_actions=N_A, a_lr=0.001)
critic = Critic(sess, n_inputs=N_F, c_lr=0.01)

sess.run(tf.global_variables_initializer())

all_ep_r = []

for i_episode in range(MAX_EPISODE):
	s = env.reset()
	ep_r = 0
	t = 0

	while True:
		a = actor.choose_action(s)
		# print("action", a)

		s_, r, done, info = env.step(a)

		if done:
			r = -20

		ep_r += r

		td_error = critic.learn(s, r, s_)
		actor.learn(s, a, td_error)

		s = s_
		t += 1

		if done or t >= MAX_EP_STEPS:
			ep_rs_sum = ep_r

			if 'running_reward' not in globals():
				running_reward = ep_rs_sum
			else:
				running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

			if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True
			print("episode:", i_episode, "  reward:", int(running_reward))
			break
	if i_episode == 0:
		all_ep_r.append(ep_r)
	else:
		all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()