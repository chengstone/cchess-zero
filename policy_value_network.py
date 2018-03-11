#coding:utf-8
import tensorflow as tf
import numpy as np

import os


class policy_value_network(object):
    def __init__(self, res_block_nums = 7):
        #         self.ckpt = os.path.join(os.getcwd(), 'models/best_model.ckpt-13999')    # TODO
        self.save_dir = "./models"
        self.is_logging = True

        """reset TF Graph"""
        tf.reset_default_graph()
        """Creat a new graph for the network"""
        # g = tf.Graph()

        self.sess = tf.Session()
        # self.sess = tf.InteractiveSession()

        # Variables
        self.filters_size = 128    # or 256
        self.prob_size = 2086
        self.digest = None
        self.training = tf.placeholder(tf.bool, name='training')
        self.inputs_ = tf.placeholder(tf.float32, [None, 9, 10, 14], name='inputs')  # + 2    # TODO C plain x 2
        self.c_l2 = 0.0001
        self.momentum = 0.9
        self.global_norm = 100
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')    #0.001    #5e-3    #0.05    #
        tf.summary.scalar('learning_rate', self.learning_rate)

        # First block
        self.pi_ = tf.placeholder(tf.float32, [None, self.prob_size], name='pi')
        self.z_ = tf.placeholder(tf.float32, [None, 1], name='z')

        # NWHC format
        # batch, 9 * 10, 14 channels
        # inputs_ = tf.reshape(self.inputs_, [-1, 9, 10, 14])
        #   data_format: A string, one of `channels_last` (default) or `channels_first`.
        #     The ordering of the dimensions in the inputs.
        #     `channels_last` corresponds to inputs with shape `(batch, width, height, channels)`
        #   while `channels_first` corresponds to inputs with shape `(batch, channels, width, height)`.
        self.layer = tf.layers.conv2d(self.inputs_, self.filters_size, 3, padding='SAME')  # filters 128(or 256)

        self.layer = tf.contrib.layers.batch_norm(self.layer, center=False, epsilon=1e-5, fused=True,
                                                  is_training=self.training, activation_fn=tf.nn.relu)    # epsilon = 0.25

        # residual_block
        with tf.name_scope("residual_block"):
            for _ in range(res_block_nums):
                self.layer = self.residual_block(self.layer)

        # policy_head
        with tf.name_scope("policy_head"):
            self.policy_head = tf.layers.conv2d(self.layer, 2, 1, padding='SAME')
            self.policy_head = tf.contrib.layers.batch_norm(self.policy_head, center=False, epsilon=1e-5, fused=True,
                                                            is_training=self.training, activation_fn=tf.nn.relu)

            # print(self.policy_head.shape)  # (?, 9, 10, 2)
            self.policy_head = tf.reshape(self.policy_head, [-1, 9 * 10 * 2])
            self.policy_head = tf.contrib.layers.fully_connected(self.policy_head, self.prob_size, activation_fn=None)
            # self.prediction = tf.nn.softmax(self.policy_head)

        # value_head
        with tf.name_scope("value_head"):
            self.value_head = tf.layers.conv2d(self.layer, 1, 1, padding='SAME')
            self.value_head = tf.contrib.layers.batch_norm(self.value_head, center=False, epsilon=1e-5, fused=True,
                                                           is_training=self.training, activation_fn=tf.nn.relu)
            # print(self.value_head.shape)  # (?, 9, 10, 1)
            self.value_head = tf.reshape(self.value_head, [-1, 9 * 10 * 1])
            self.value_head = tf.contrib.layers.fully_connected(self.value_head, 256, activation_fn=tf.nn.relu)
            self.value_head = tf.contrib.layers.fully_connected(self.value_head, 1, activation_fn=tf.nn.tanh)

        # loss
        with tf.name_scope("loss"):
            self.policy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.pi_, logits=self.policy_head)
            self.policy_loss = tf.reduce_mean(self.policy_loss)

            #             self.value_loss = tf.squared_difference(self.z_, self.value_head)
            self.value_loss = tf.losses.mean_squared_error(labels=self.z_, predictions=self.value_head)
            self.value_loss = tf.reduce_mean(self.value_loss)
            tf.summary.scalar('mse_loss', self.value_loss)

            regularizer = tf.contrib.layers.l2_regularizer(scale=self.c_l2)
            regular_variables = tf.trainable_variables()
            self.l2_loss = tf.contrib.layers.apply_regularization(regularizer, regular_variables)

            #             self.loss = self.value_loss - self.policy_loss + self.l2_loss
            self.loss = self.value_loss + self.policy_loss + self.l2_loss
            tf.summary.scalar('loss', self.loss)

            #     train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        #         optimizer = tf.train.AdamOptimizer(self.learning_rate)
        #         gradients = optimizer.compute_gradients(self.loss)
        #         train_op = optimizer.apply_gradients(gradients, global_step=global_step)

        # 优化损失
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate, momentum=self.momentum, use_nesterov=True)

        # self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(self.update_ops):
        #     self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        # Accuracy
        correct_prediction = tf.equal(tf.argmax(self.policy_head, 1), tf.argmax(self.pi_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction, name='accuracy')
        tf.summary.scalar('move_accuracy', self.accuracy)

        # grads = self.average_gradients(tower_grads)
        grads = optimizer.compute_gradients(self.loss)
        # defensive step 2 to clip norm
        clipped_grads, self.norm = tf.clip_by_global_norm(
            [g for g, _ in grads], self.global_norm)

        # defensive step 3 check NaN
        # See: https://stackoverflow.com/questions/40701712/how-to-check-nan-in-gradients-in-tensorflow-when-updating
        grad_check = [tf.check_numerics(g, message='NaN Found!') for g in clipped_grads]
        with tf.control_dependencies(grad_check):
            self.train_op = optimizer.apply_gradients(
                zip(clipped_grads, [v for _, v in grads]),
                global_step=self.global_step, name='train_step')

        if self.is_logging:
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

        self.summaries_op = tf.summary.merge_all()

        # Train Summaries
        self.train_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), "cchesslogs/train"), self.sess.graph)

        # Test summaries
        self.test_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), "cchesslogs/test"), self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        #         self.sess.run(tf.local_variables_initializer())
        #         self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        self.train_restore()

    def residual_block(self, in_layer):
        orig = tf.identity(in_layer)

        layer = tf.layers.conv2d(in_layer, self.filters_size, 3, padding='SAME')  # filters 128(or 256)
        layer = tf.contrib.layers.batch_norm(layer, center=False, epsilon=1e-5, fused=True,
                                             is_training=self.training, activation_fn=tf.nn.relu)

        layer = tf.layers.conv2d(layer, self.filters_size, 3, padding='SAME')  # filters 128(or 256)
        layer = tf.contrib.layers.batch_norm(layer, center=False, epsilon=1e-5, fused=True, is_training=self.training)
        out = tf.nn.relu(tf.add(orig, layer))

        return out

    def train_restore(self):
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            # self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
            print("Successfully loaded:", tf.train.latest_checkpoint(self.save_dir))
            # print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def restore(self, file):
        print("Restoring from {0}".format(file))
        self.saver.restore(self.sess, file)  # self.ckpt

    def save(self, in_global_step):
        #         save_path = self.saver.save(self.sess, path, global_step=self.global_step)
        save_path = self.saver.save(self.sess, os.path.join(self.save_dir, 'best_model.ckpt'),
                                    global_step=in_global_step)    #self.global_step
        print("Model saved in file: {}".format(save_path))

    def train_step(self, positions, probs, winners, learning_rate):
        feed_dict = {
            self.inputs_: positions,
            self.training: True,
            self.learning_rate: learning_rate,
            self.pi_: probs,
            self.z_: winners
        }

        _, accuracy, loss, global_step, summary = self.sess.run([self.train_op, self.accuracy, self.loss, self.global_step, self.summaries_op], feed_dict=feed_dict)
        self.train_writer.add_summary(summary, global_step)
        # print(accuracy)
        # print(loss)
        return accuracy, loss, global_step

    #@profile
    def forward(self, positions):  # , probs, winners
        feed_dict = {
            self.inputs_: positions,
            self.training: False
        }
        #             ,
        #             self.pi_: probs,
        #             self.z_: winners
        action_probs, value = self.sess.run([self.policy_head, self.value_head], feed_dict=feed_dict)    # self.prediction
        # print(action_probs.shape)
        # print(value.shape)

        return action_probs, value
        # return action_probs, value