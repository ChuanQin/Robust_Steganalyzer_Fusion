import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
import functools
import os
from .utils import Model

class SRNet(Model):
    def _build_model(self):
        if self.data_format == 'NCHW':
            reduction_axis = [2,3]
            _inputs = tf.cast(tf.transpose(self.x_input, [0, 3, 1, 2]), tf.float32)
        else:
            reduction_axis = [1,2]
            _inputs = tf.cast(self.x_input, tf.float32)
        self.L = []
        with arg_scope([layers.conv2d], num_outputs=16,
                       kernel_size=3, stride=1, padding='SAME',
                       data_format=self.data_format,
                       activation_fn=None,
                       weights_initializer=layers.variance_scaling_initializer(),
                       weights_regularizer=layers.l2_regularizer(2e-4),
                       biases_initializer=tf.constant_initializer(0.2),
                       biases_regularizer=None, 
                       trainable = self.is_training),\
            arg_scope([layers.batch_norm],
                       decay=0.9, center=True, scale=True, 
                       updates_collections=None, is_training=self.is_training,
                       fused=True, data_format=self.data_format),\
            arg_scope([layers.avg_pool2d],
                       kernel_size=[3,3], stride=[2,2], padding='SAME',
                       data_format=self.data_format):
            with tf.variable_scope('Layer1', reuse = tf.AUTO_REUSE): 
                conv=layers.conv2d(_inputs, num_outputs=64, kernel_size=3)
                actv=tf.nn.relu(layers.batch_norm(conv))
                self.L.append(actv)
            with tf.variable_scope('Layer2', reuse = tf.AUTO_REUSE): 
                conv=layers.conv2d(actv)
                actv=tf.nn.relu(layers.batch_norm(conv))
                self.L.append(actv)
            with tf.variable_scope('Layer3', reuse = tf.AUTO_REUSE): 
                conv1=layers.conv2d(actv)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn2=layers.batch_norm(conv2)
                res= tf.add(actv, bn2)
                self.L.append(res)
            with tf.variable_scope('Layer4', reuse = tf.AUTO_REUSE): 
                conv1=layers.conv2d(res)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn2=layers.batch_norm(conv2)
                res= tf.add(res, bn2)
                self.L.append(res)
            with tf.variable_scope('Layer5', reuse = tf.AUTO_REUSE): 
                conv1=layers.conv2d(res)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn=layers.batch_norm(conv2)
                res= tf.add(res, bn)
                self.L.append(res)
            with tf.variable_scope('Layer6', reuse = tf.AUTO_REUSE): 
                conv1=layers.conv2d(res)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn=layers.batch_norm(conv2)
                res= tf.add(res, bn)
                self.L.append(res)
            with tf.variable_scope('Layer7', reuse = tf.AUTO_REUSE): 
                conv1=layers.conv2d(res)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn=layers.batch_norm(conv2)
                res= tf.add(res, bn)
                self.L.append(res)
            with tf.variable_scope('Layer8', reuse = tf.AUTO_REUSE): 
                # side adding part of type 3 layer
                convs = layers.conv2d(res, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                # main stream conv, bn and average pooling
                conv1=layers.conv2d(res)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn=layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                # adding
                res= tf.add(convs, pool)
                self.L.append(res)
            with tf.variable_scope('Layer9', reuse = tf.AUTO_REUSE):  
                convs = layers.conv2d(res, num_outputs=64, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1=layers.conv2d(res, num_outputs=64)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1, num_outputs=64)
                bn=layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res= tf.add(convs, pool)
                self.L.append(res)
            with tf.variable_scope('Layer10', reuse = tf.AUTO_REUSE): 
                convs = layers.conv2d(res, num_outputs=128, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1=layers.conv2d(res, num_outputs=128)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1, num_outputs=128)
                bn=layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res= tf.add(convs, pool)
                self.L.append(res)
            with tf.variable_scope('Layer11', reuse = tf.AUTO_REUSE): 
                convs = layers.conv2d(res, num_outputs=256, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1=layers.conv2d(res, num_outputs=256)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1, num_outputs=256)
                bn=layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res= tf.add(convs, pool)
                self.L.append(res)
            with tf.variable_scope('Layer12', reuse = tf.AUTO_REUSE): 
                conv1=layers.conv2d(res, num_outputs=512)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1, num_outputs=512)
                bn=layers.batch_norm(conv2)
                avgp = tf.reduce_mean(bn, reduction_axis,  keep_dims=True )
                self.L.append(avgp)
        ip=layers.fully_connected(layers.flatten(avgp), num_outputs=2,
                    activation_fn=None, normalizer_fn=None,
                    weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01), 
                    biases_initializer=tf.constant_initializer(0.), reuse = tf.AUTO_REUSE, trainable = self.is_training, scope='ip')
        self.outputs = ip
        return self.outputs