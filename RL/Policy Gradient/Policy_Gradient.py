import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class PolicyGradient:
	def __init__(self, n_features, n_actions, learning_rate = 0.01, reward_decay = 0.95):
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = reward_decay

		self.ep_obs, self.ep_as, self.ep_rs = [], [], []

		self.cost_his = []

		self._build_net()

		self.sess = tf.Session()

		self.sess.run(tf.global_variables_initializer())

	def _build_net(self):
		with tf.variable_scope('inputs'):
			self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name='observation')
			self.tf_acts = tf.placeholder(tf.int32, [None, ], name='actions_num')
			self.tf_vt = tf.placeholder(tf.float32, [None, ], name='action_value')

		layer = tf.layers.dense(self.tf_obs,
								32,
								tf.nn.relu,
								kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
								bias_initializer=tf.constant_initializer(0.1),
								name='fc1')
		self.all_act = tf.layers.dense(layer,
								  self.n_actions,
								  tf.nn.softmax,
								  kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
								  bias_initializer=tf.constant_initializer(0.1),
								  name='fc2')

		with tf.variable_scope('loss'):
			log_prob = tf.reduce_sum(-tf.log(self.all_act) * tf.one_hot(self.tf_acts, self.n_actions), axis=1)
			self.loss = tf.reduce_mean(log_prob * self.tf_vt)

		with tf.variable_scope('train'):
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

	def choose_action(self, observation):
		prob_weights = self.sess.run(self.all_act, feed_dict={self.tf_obs: observation[np.newaxis, :]})
		action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
		return action

	def store_transition(self, s, a, r):
		self.ep_obs.append(s)
		self.ep_as.append(a)
		self.ep_rs.append(r)

	def learn(self):
		# discount and normalize episode reward
		discounted_ep_rs_norm = self._discount_and_norm_rewards()

		# train on episode
		_, cost = self.sess.run([self.train_op, self.loss], feed_dict={
			self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
			self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
			self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
		})

		self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
		self.cost_his.append(cost)
		return discounted_ep_rs_norm


	def _discount_and_norm_rewards(self):
		# discount episode rewards
		discounted_ep_rs = np.zeros_like(self.ep_rs)
		running_add = 0
		for t in reversed(range(0, len(self.ep_rs))):
			running_add = running_add * self.gamma + self.ep_rs[t]
			discounted_ep_rs[t] = running_add

		# normalize episode rewards
		discounted_ep_rs -= np.mean(discounted_ep_rs)
		discounted_ep_rs /= np.std(discounted_ep_rs)
		return discounted_ep_rs

	def plot_cost(self):
		plt.plot(np.arange(len(self.cost_his)), self.cost_his)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.show()