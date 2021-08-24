import torch
import tensorflow as tf
import time
import os

class Model:
    def __init__(self, 
                 is_training=None, 
                 data_format='NCHW',
                 img_size = 256,
                 color = False):
        self.data_format = data_format
        if is_training is None:
            self.is_training = tf.get_variable('is_training', dtype=tf.bool, \
                                    initializer=tf.constant_initializer(True), \
                                    trainable=False)
        else:
            self.is_training = is_training
        self.color = color
        self.img_size = img_size
        if self.color:
            self.x_input = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, 3])
        else:
            self.x_input = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, 1])
        self.y_input = tf.placeholder(tf.int64, shape=None)

    def _build_model(self, inputs):
        raise NotImplementedError('Here is your model definition')

    def _build_losses(self, labels, classes = 2):
        self.labels = tf.cast(labels, tf.int64)
        with tf.variable_scope('loss'):
            oh = tf.one_hot(self.labels, classes)
            xen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
                                                    labels=oh,logits=self.outputs))
            ################# SRNet and YeNet losses ###################
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.add_n([xen_loss] + reg_losses)
            ################ XuNet losses ###################
            # self.loss = xen_loss
            ##################################################
            am = tf.argmax(self.outputs, 1)
            equal = tf.equal(am, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
        return self.loss, self.accuracy

    def _get_outputs(self):
        return self.outputs

    def _get_features(self):
        return self.L[-1]


### Implementation of Adamax optimizer, taken from : https://github.com/openai/iaf/blob/master/tf_utils/adamax.py
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf

class AdamaxOptimizer(optimizer.Optimizer):
    """
    Optimizer that implements the Adamax algorithm. 
    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    @@__init__
    """

    def __init__(self, 
                 learning_rate=0.001, 
                 beta1=0.9, 
                 beta2=0.999, 
                 use_locking=False, 
                 name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")