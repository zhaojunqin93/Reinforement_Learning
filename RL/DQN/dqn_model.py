import tensorflow as tf

class DQN:
    def __init__(self, input_shape, n_actions, learning_rate):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.create_model(input_shape[0], n_actions)
        # Setup TensorBoard Writer
        writer = tf.summary.FileWriter("/tensorboard/pg/1")
        ## Losses
        tf.summary.scalar("Loss", self.loss)
        write_op = tf.summary.merge_all()

    def create_model(self, input_size, n_actions):
        with tf.name_scope("inputs"):
            self.input_data = tf.placeholder(tf.float32, [None, input_size], name="input_data")
            self.actions = tf.placeholder(tf.float32, [None, n_actions], name="actions")
            with tf.name_scope("fc1"):
                self.fc1 = tf.contrib.layers.fully_connected(inputs=self.input_data,
                                                             num_outputs=16,
                                                             activation_fn=tf.nn.relu,
                                                             weights_initializer=tf.contrib.layers.xavier_initializer())
            with tf.name_scope("output"):
                self.output = tf.contrib.layers.fully_connected(inputs=self.fc1,
                                                                num_outputs=n_actions,
                                                                activation_fn=tf.nn.relu,
                                                                weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("filtered_output"):
                # filtered_output = tf.multiply(self.output,actions)
                self.filtered_output = self.output * self.actions

            with tf.name_scope("loss"):
                self.q_values = tf.placeholder(tf.float32, [None, n_actions], name="q_values")
                self.loss = tf.reduce_mean(tf.squared_difference(self.filtered_output, self.q_values))

            with tf.name_scope("train"):
                self.train_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)