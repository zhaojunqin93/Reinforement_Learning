import tensorflow as tf


class Policy_net:
    def __init__(self, name: str, S_DIMs, A_DIMs, temp=0.1):
        """
        :param temp: temperature of boltzmann distribution
        """

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None, S_DIMs], name='obs')

            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=32, activation=tf.nn.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=32, activation=tf.nn.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=A_DIMs, activation=tf.nn.tanh)
                self.act_probs = tf.layers.dense(inputs=tf.divide(layer_3, temp), units=A_DIMs, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=32, activation=tf.nn.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=32, activation=tf.nn.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)

            self.act_stochastic = tf.random.categorical(tf.log(self.act_probs), num_samples=1) # Draws samples from a categorical distribution.
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
            feed_dict = {self.obs: obs}
            a = tf.get_default_session().run(self.act_stochastic, feed_dict)
            # print("a", a)
            b = tf.get_default_session().run(self.v_preds, feed_dict)
            # print("b", b)
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict)
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)