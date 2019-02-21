#coding:utf-8
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import summary_ops_v2
import os


class policy_value_network(object):
    def __init__(self, learning_rate_fn, res_block_nums = 7):
        #         self.ckpt = os.path.join(os.getcwd(), 'models/best_model.ckpt-13999')    # TODO
        self.save_dir = "./models"
        self.is_logging = True

        if tf.io.gfile.exists(self.save_dir):
            #             print('Removing existing model dir: {}'.format(MODEL_DIR))
            #             tf.io.gfile.rmtree(MODEL_DIR)
            pass
        else:
            tf.io.gfile.makedirs(self.save_dir)

        train_dir = os.path.join(self.save_dir, 'summaries', 'train')
        test_dir = os.path.join(self.save_dir, 'summaries', 'eval')

        self.train_summary_writer = summary_ops_v2.create_file_writer(train_dir, flush_millis=10000)
        self.test_summary_writer = summary_ops_v2.create_file_writer(test_dir, flush_millis=10000, name='test')

        # Variables
        self.filters_size = 128    # or 256
        self.prob_size = 2086
        self.digest = None

        self.inputs_ = tf.keras.layers.Input([9, 10, 14], dtype='float32', name='inputs')  # TODO C plain x 2
        self.c_l2 = 0.0001
        self.momentum = 0.9
        self.global_norm = 100

        self.layer = tf.keras.layers.Conv2D(kernel_size=3, filters=self.filters_size, padding='same')(self.inputs_)
        self.layer = tf.keras.layers.BatchNormalization(epsilon=1e-5, fused=True)(self.layer)
        self.layer = tf.keras.layers.ReLU()(self.layer)

        # residual_block
        with tf.name_scope("residual_block"):
            for _ in range(res_block_nums):
                self.layer = self.residual_block(self.layer)

        # policy_head
        with tf.name_scope("policy_head"):
            self.policy_head = tf.keras.layers.Conv2D(filters=2, kernel_size=1, padding='same')(self.layer)
            self.policy_head = tf.keras.layers.BatchNormalization(epsilon=1e-5, fused=True)(self.policy_head)
            self.policy_head = tf.keras.layers.ReLU()(self.policy_head)

            self.policy_head = tf.keras.layers.Reshape([9 * 10 * 2])(self.policy_head)
            self.policy_head = tf.keras.layers.Dense(self.prob_size)(self.policy_head)

        # value_head
        with tf.name_scope("value_head"):
            self.value_head = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same')(self.layer)
            self.value_head = tf.keras.layers.BatchNormalization(epsilon=1e-5, fused=True)(
                self.value_head)
            self.value_head = tf.keras.layers.ReLU()(self.value_head)

            self.value_head = tf.keras.layers.Reshape([9 * 10 * 1])(self.value_head)
            self.value_head = tf.keras.layers.Dense(256, activation='relu')(self.value_head)
            self.value_head = tf.keras.layers.Dense(1, activation='tanh')(self.value_head)

        self.model = tf.keras.Model(
            inputs=[self.inputs_],
            outputs=[self.policy_head, self.value_head])

        self.model.summary()

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        #         optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # 优化损失
        self.optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=learning_rate_fn, momentum=self.momentum, use_nesterov=True)

        self.CategoricalCrossentropyLoss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.MSE = tf.keras.losses.MeanSquaredError()
        self.ComputeMetrics = tf.keras.metrics.MeanAbsoluteError()


        # self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(self.update_ops):
        #     self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

        # Restore variables on creation if a checkpoint exists.
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def residual_block(self, in_layer):
        orig = tf.convert_to_tensor(in_layer)    # tf.identity(in_layer)
        layer = tf.keras.layers.Conv2D(kernel_size=3, filters=self.filters_size, padding='same')(in_layer)
        layer = tf.keras.layers.BatchNormalization(epsilon=1e-5, fused=True)(layer)
        layer = tf.keras.layers.ReLU()(layer)

        layer = tf.keras.layers.Conv2D(kernel_size=3, filters=self.filters_size, padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(epsilon=1e-5, fused=True)(layer)
        add_layer = tf.keras.layers.add([orig, layer])
        out = tf.keras.layers.ReLU()(add_layer)

        return out

    # def train_restore(self):
    #     if not os.path.isdir(self.save_dir):
    #         os.mkdir(self.save_dir)
    #     checkpoint = tf.train.get_checkpoint_state(self.save_dir)
    #     if checkpoint and checkpoint.model_checkpoint_path:
    #         # self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
    #         self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
    #         print("Successfully loaded:", tf.train.latest_checkpoint(self.save_dir))
    #         # print("Successfully loaded:", checkpoint.model_checkpoint_path)
    #     else:
    #         print("Could not find old network weights")

    # def restore(self, file):
    #     print("Restoring from {0}".format(file))
    #     self.saver.restore(self.sess, file)  # self.ckpt

    def save(self, in_global_step):
        self.checkpoint.save(self.checkpoint_prefix)
        # print("Model saved in file: {}".format(save_path))

    def compute_metrics(self, pi_, policy_head):
        # Accuracy
        correct_prediction = tf.equal(tf.argmax(input=policy_head, axis=1), tf.argmax(input=pi_, axis=1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(input_tensor=correct_prediction, name='accuracy')

        summary_ops_v2.scalar('move_accuracy', accuracy)
        return accuracy

    def apply_regularization(self, regularizer, weights_list=None):
        """Returns the summed penalty by applying `regularizer` to the `weights_list`.
        Adding a regularization penalty over the layer weights and embedding weights
        can help prevent overfitting the training data. Regularization over layer
        biases is less common/useful, but assuming proper data preprocessing/mean
        subtraction, it usually shouldn't hurt much either.
        Args:
          regularizer: A function that takes a single `Tensor` argument and returns
            a scalar `Tensor` output.
          weights_list: List of weights `Tensors` or `Variables` to apply
            `regularizer` over. Defaults to the `GraphKeys.WEIGHTS` collection if
            `None`.
        Returns:
          A scalar representing the overall regularization penalty.
        Raises:
          ValueError: If `regularizer` does not return a scalar output, or if we find
              no weights.
        """
        # if not weights_list:
        #     weights_list = ops.get_collection(ops.GraphKeys.WEIGHTS)
        if not weights_list:
            raise ValueError('No weights to regularize.')
        with tf.name_scope('get_regularization_penalty',
                            values=weights_list) as scope:
            penalties = [regularizer(w) for w in weights_list]
            penalties = [
                p if p is not None else tf.constant(0.0) for p in penalties
            ]
            for p in penalties:
                if p.get_shape().ndims != 0:
                    raise ValueError('regularizer must return a scalar Tensor instead of a '
                                     'Tensor with rank %d.' % p.get_shape().ndims)

            summed_penalty = tf.add_n(penalties, name=scope)
            # ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, summed_penalty)
            return summed_penalty

    def compute_loss(self, pi_, z_, policy_head, value_head):

        # loss
        with tf.name_scope("loss"):
            policy_loss = tf.keras.losses.categorical_crossentropy(y_true=pi_, y_pred=policy_head, from_logits=True)
            policy_loss = tf.reduce_mean(policy_loss)

            value_loss = tf.keras.losses.mean_squared_error(z_, value_head)
            value_loss = tf.reduce_mean(value_loss)
            summary_ops_v2.scalar('mse_loss', value_loss)

            regularizer = tf.keras.regularizers.l2(self.c_l2)
            regular_variables = self.model.trainable_variables
            l2_loss = self.apply_regularization(regularizer, regular_variables)

            #             self.loss = value_loss - policy_loss + l2_loss
            self.loss = value_loss + policy_loss + l2_loss
            summary_ops_v2.scalar('loss', self.loss)

        return self.loss

    @tf.function
    def train_step(self, positions, pi, z, learning_rate=0):
        # Record the operations used to compute the loss, so that the gradient
        # of the loss with respect to the variables can be computed.
        #         metrics = 0

        with tf.GradientTape() as tape:
            policy_head, value_head = self.model(positions, training=True)
            loss = self.compute_loss(pi, z, policy_head, value_head)
            # self.ComputeMetrics(y, logits)
            metrics = self.compute_metrics(pi, policy_head)
        grads = tape.gradient(loss, self.model.trainable_variables)

        # grads = self.average_gradients(tower_grads)
        # grads = self.optimizer.compute_gradients(self.loss)
        # defensive step 2 to clip norm
        # grads0_lst = tf.map_fn(lambda x: x[0], grads)  # [g for g, _ in grads]
        clipped_grads, self.norm = tf.clip_by_global_norm(grads, self.global_norm)

        # defensive step 3 check NaN
        # See: https://stackoverflow.com/questions/40701712/how-to-check-nan-in-gradients-in-tensorflow-when-updating
        grad_check = [tf.debugging.check_numerics(g, message='NaN Found!') for g in clipped_grads]
        with tf.control_dependencies(grad_check):
            self.optimizer.apply_gradients(
                zip(clipped_grads, self.model.trainable_variables),  # [v for _, v in grads]
                global_step=self.global_step, name='train_step')

        if self.is_logging:
            for grad, var in zip(grads, self.model.trainable_variables):
                if grad is not None:
                    summary_ops_v2.histogram(var.name + '/gradients', grad)
            for var in self.model.trainable_variables:
                summary_ops_v2.histogram(var.name, var)

        return metrics, loss, self.global_step

    #@profile
    def forward(self, positions):

        positions=np.array(positions)
        if len(positions.shape) == 3:
            sp = positions.shape
            positions=np.reshape(positions, [1, sp[0], sp[1], sp[2]])
        action_probs, value = self.model(positions, training=False)

        return action_probs, value

