#coding:utf-8
import tensorflow as tf
import numpy as np

import os

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']


class policy_value_network_gpus(object):
    def __init__(self, num_gpus = 1, res_block_nums = 7):
        #         self.ckpt = os.path.join(os.getcwd(), 'models/best_model.ckpt-13999')    # TODO
        self.num_gpus = num_gpus
        self.save_dir = "./gpu_models"
        self.is_logging = True
        self.res_block_nums = res_block_nums

        """reset TF Graph"""
        tf.reset_default_graph()
        """Creat a new graph for the network"""
        # g = tf.Graph()

        config = tf.ConfigProto(
            inter_op_parallelism_threads=4,
            intra_op_parallelism_threads=4)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        """Assign a Session that excute the network"""
        # config.gpu_options.per_process_gpu_memory_fraction = 0.75
        # self.sess = tf.Session(config=config, graph=g)

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        # config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        # self.sess = tf.InteractiveSession()

        with tf.device('/cpu:0'):
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

            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            # self.learning_rate = tf.maximum(tf.train.exponential_decay(
            #     0.001, self.global_step, 1e3, 0.66), 1e-5)
            # self.learning_rate = tf.Variable(self.hps.lrn_rate, dtype=tf.float32, trainable=False)
            tf.summary.scalar('learning_rate', self.learning_rate)

            # 优化损失
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate, momentum=self.momentum, use_nesterov=True)    # , use_locking=True
            # optimizer = tf.train.AdamOptimizer(self.learning_rate)

            # First block
            self.pi_ = tf.placeholder(tf.float32, [None, self.prob_size], name='pi')
            self.z_ = tf.placeholder(tf.float32, [None, 1], name='z')

            # batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([self.inputs_, self.pi_, self.z_], capacity=3 * self.num_gpus)

            inputs_batches = tf.split(self.inputs_, self.num_gpus, axis=0)
            pi_batches = tf.split(self.pi_, self.num_gpus, axis=0)
            z_batches = tf.split(self.z_, self.num_gpus, axis=0)


            tower_grads = [None] * self.num_gpus

            self.loss = 0
            self.accuracy = 0
            self.policy_head = []
            self.value_head = []

            with tf.variable_scope(tf.get_variable_scope()):
                """Build the core model within the graph."""
                for i in range(self.num_gpus):
                    with tf.device(self.assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):    #tf.device('/gpu:{i}'):
                        with tf.name_scope('TOWER_{}'.format(i)) as scope:
                            inputs_batch, pi_batch, z_batch = inputs_batches[i], pi_batches[i], z_batches[i]    # batch_queue.dequeue() #
            # NWHC format
            # batch, 9 * 10, 14 channels
            # inputs_ = tf.reshape(self.inputs_, [-1, 9, 10, 14])
                            loss = self.tower_loss(inputs_batch, pi_batch, z_batch, i)
                            # reuse variable happens here
                            tf.get_variable_scope().reuse_variables()
                            grad = optimizer.compute_gradients(loss)
                            tower_grads[i] = grad

            self.loss /= self.num_gpus
            self.accuracy /= self.num_gpus
            grads = self.average_gradients(tower_grads)
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

    def tower_loss(self, inputs_batch, pi_batch, z_batch, i):
        with tf.variable_scope('init'):
            layer = tf.layers.conv2d(inputs_batch, self.filters_size, 3, padding='SAME')  # filters 128(or 256)

            layer = tf.contrib.layers.batch_norm(layer, center=False, epsilon=1e-5, fused=True,
                                                  is_training=self.training, activation_fn=tf.nn.relu)    # epsilon = 0.25

        # residual_block
        with tf.variable_scope("residual_block"):
            for _ in range(self.res_block_nums):
                layer = self.residual_block(layer)

        # policy_head
        with tf.variable_scope("policy_head"):
            policy_head = tf.layers.conv2d(layer, 2, 1, padding='SAME')
            policy_head = tf.contrib.layers.batch_norm(policy_head, center=False, epsilon=1e-5, fused=True,
                                                            is_training=self.training, activation_fn=tf.nn.relu)

            # print(self.policy_head.shape)  # (?, 9, 10, 2)
            policy_head = tf.reshape(policy_head, [-1, 9 * 10 * 2])
            policy_head = tf.contrib.layers.fully_connected(policy_head, self.prob_size, activation_fn=None)
            # prediction = tf.nn.softmax(policy_head)
            self.policy_head.append(policy_head)    #prediction

        # value_head
        with tf.variable_scope("value_head"):
            value_head = tf.layers.conv2d(layer, 1, 1, padding='SAME')
            value_head = tf.contrib.layers.batch_norm(value_head, center=False, epsilon=1e-5, fused=True,
                                           is_training=self.training, activation_fn=tf.nn.relu)
            # print(self.value_head.shape)  # (?, 9, 10, 1)
            value_head = tf.reshape(value_head, [-1, 9 * 10 * 1])
            value_head = tf.contrib.layers.fully_connected(value_head, 256, activation_fn=tf.nn.relu)
            value_head = tf.contrib.layers.fully_connected(value_head, 1, activation_fn=tf.nn.tanh)
            self.value_head.append(value_head)

        # loss
        with tf.variable_scope("loss"):
            policy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi_batch, logits=policy_head)    #self.pi_
            policy_loss = tf.reduce_mean(policy_loss)

            #             self.value_loss = tf.squared_difference(self.z_, self.value_head)
            value_loss = tf.losses.mean_squared_error(labels=z_batch, predictions=value_head)    #self.z_
            value_loss = tf.reduce_mean(value_loss)
            tf.summary.scalar('mse_tower_{}'.format(i), value_loss)

            regularizer = tf.contrib.layers.l2_regularizer(scale=self.c_l2)
            regular_variables = tf.trainable_variables()
            l2_loss = tf.contrib.layers.apply_regularization(regularizer, regular_variables)

            #             self.loss = self.value_loss - self.policy_loss + self.l2_loss
            loss = value_loss + policy_loss + l2_loss
            self.loss += loss
            tf.summary.scalar('loss_tower_{}'.format(i), loss)
                #     train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            # self.global_step = tf.Variable(0, name="global_step", trainable=False)
            #         optimizer = tf.train.AdamOptimizer(self.learning_rate)
            #         gradients = optimizer.compute_gradients(self.loss)
            #         train_op = optimizer.apply_gradients(gradients, global_step=global_step)

                            # self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                            # with tf.control_dependencies(self.update_ops):
                            #     self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        with tf.variable_scope("accuracy"):
            # Accuracy
            correct_prediction = tf.equal(tf.argmax(policy_head, 1), tf.argmax(pi_batch, 1))    #self.pi_
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction, name='accuracy')
            self.accuracy += accuracy
            tf.summary.scalar('move_accuracy_tower_{}'.format(i), accuracy)
        return loss


    # By default, all variables will be placed on '/gpu:0'
    # So we need a custom device function, to assign all variables to '/cpu:0'
    # Note: If GPUs are peered, '/gpu:0' can be a faster option

    def assign_to_device(self, device, ps_device='/cpu:0'):
        def _assign(op):
            node_def = op if isinstance(op, tf.NodeDef) else op.node_def
            if node_def.op in PS_OPS:
                return "/" + ps_device
            else:
                return device

        return _assign

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, var in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                # print('Network variables: {var.name}')
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

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
        

        # try:
        _, accuracy, loss, global_step, summary = self.sess.run([self.train_op, self.accuracy, self.loss, self.global_step, self.summaries_op], feed_dict=feed_dict)
        self.train_writer.add_summary(summary, global_step)
        # print(accuracy)
        # print(loss)
        return accuracy, loss, global_step
        # except tf.errors.InvalidArgumentError:
        #     print('Contains NaN gradients.')
            # continue

    #@profile
    def forward(self, positions):  # , probs, winners
        # print("positions.shape : ", positions.shape)
        positions = np.array(positions)
        batch_n = positions.shape[0] // self.num_gpus
        alone = positions.shape[0] % self.num_gpus

        if alone != 0:
            if(positions.shape[0] != 1):
                feed_dict = {
                    self.inputs_: positions[:positions.shape[0] - alone],
                    self.training: False
                }
                action_probs, value = self.sess.run([self.policy_head, self.value_head], feed_dict=feed_dict)
                action_probs, value = np.vstack(action_probs), np.vstack(value)

            new_positions = positions[positions.shape[0] - alone:]
            pos_lst = []
            while len(pos_lst) == 0 or (np.array(pos_lst).shape[0] * np.array(pos_lst).shape[1]) % self.num_gpus != 0:
                pos_lst.append(new_positions)

            if(len(pos_lst) != 0):
                shape = np.array(pos_lst).shape
                pos_lst = np.array(pos_lst).reshape([shape[0] * shape[1], 9, 10, 14])

            feed_dict = {
                self.inputs_: pos_lst,
                self.training: False
            }
            action_probs_2, value_2 = self.sess.run([self.policy_head, self.value_head], feed_dict=feed_dict)
            # print("action_probs_2.shape : ", np.array(action_probs_2).shape)
            # print("value_2.shape : ", np.array(value_2).shape)
            action_probs_2, value_2 = action_probs_2[0], value_2[0]
            # print("------------------------")
            # print("action_probs_2.shape : ", np.array(action_probs_2).shape)
            # print("value_2.shape : ", np.array(value_2).shape)

            if(positions.shape[0] != 1):
                action_probs = np.concatenate((action_probs, action_probs_2),axis=0)
                value = np.concatenate((value, value_2),axis=0)

                # print("action_probs.shape : ", np.array(action_probs).shape)
                # print("value.shape : ", np.array(value).shape)
                return action_probs, value
            else:
                return action_probs_2, value_2
        else:
            feed_dict = {
                self.inputs_: positions,
                self.training: False
            }
            action_probs, value = self.sess.run([self.policy_head, self.value_head], feed_dict=feed_dict)
            # print("np.vstack(action_probs) shape : ", np.vstack(action_probs).shape)
            # print("np.vstack(value) shape : ", np.vstack(value).shape)
            
            return np.vstack(action_probs), np.vstack(value)
        # feed_dict = {
        #     self.inputs_: positions if len(pos_lst) == 0 else pos_lst,
        #     self.training: False
        # }
        
        #             ,
        #             self.pi_: probs,
        #             self.z_: winners
                     
        # action_probs, value = self.sess.run([self.policy_head, self.value_head], feed_dict=feed_dict)
        # print(action_probs.shape)
        # print(value.shape)

        # with multi-gpu, porbs and values are separated in each outputs
        # so vstack will merge them together.
        
        # return np.vstack(action_probs), np.vstack(value)
        # return action_probs, value