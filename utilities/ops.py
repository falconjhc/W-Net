# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf

def batch_norm(x, is_training, epsilon=1e-5, decay=0.9, scope="batch_norm",
               parameter_update_device='-1'):
    with tf.device(parameter_update_device):
        var = tf.contrib.layers.batch_norm(x, decay=decay, updates_collections=None, epsilon=epsilon,
                                           scale=True, is_training=is_training, scope=scope)

    return var

def layer_norm(x, scope="layer_norm",
               parameter_update_device='-1'):
    with tf.device(parameter_update_device):
        var = tf.contrib.layers.layer_norm(x,scope=scope)

    return var

def instance_norm(x, scope="instance_norm",
                  parameter_update_device='-1'):
    with tf.device(parameter_update_device):
        var = tf.contrib.layers.instance_norm(x,scope=scope)

    return var

def adaptive_instance_norm(content,style,epsilon=1e-5):
    axes_style = [0,2,3]
    axes_content = [1,2]
    c_mean, c_var = tf.nn.moments(content,axes=axes_content,keep_dims=True)
    s_mean, s_var = tf.nn.moments(style,axes=axes_style,keep_dims=True)
    c_std, s_std = tf.sqrt(c_var+epsilon), tf.sqrt(s_var+epsilon)
    normed = tf.squeeze(s_std*(content-c_mean) / c_std + s_mean, axis=0)
    return normed


def conv2d(x,
           output_filters,
           weight_decay_rate,
           kh=5, kw=5, sh=2, sw=2,
           initializer='None',
           scope="conv2d",
           parameter_update_device='-1',
           weight_decay=False,
           name_prefix='None',
           padding='SAME'):
    weight_stddev = 0.02
    bias_init_stddev = 0.0
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()

        if initializer == 'NormalInit':
            W = variable_creation_on_device(name='W',
                                            shape=[kh, kw, shape[-1], output_filters],
                                            initializer=tf.truncated_normal_initializer(stddev=weight_stddev),
                                            parameter_update_device=parameter_update_device)
        elif initializer == 'XavierInit':
            W = variable_creation_on_device(name='W',
                                            shape=[kh, kw, shape[-1], output_filters],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                            parameter_update_device=parameter_update_device)

        if weight_decay:
            weight_decay = tf.multiply(tf.nn.l2_loss(W), weight_decay_rate, name='weight_decay')
            if not name_prefix.find('/') == -1:
                tf.add_to_collection(name_prefix[0:name_prefix.find('/')] + '_weight_decay', weight_decay)
            else:
                tf.add_to_collection(name_prefix + '_weight_decay', weight_decay)

        Wconv = tf.nn.conv2d(x, W, strides=[1, sh, sw, 1], padding=padding)

        biases = variable_creation_on_device('b',
                                             shape=[output_filters],
                                             initializer=tf.constant_initializer(bias_init_stddev),
                                             parameter_update_device=parameter_update_device)

        Wconv_plus_b = tf.reshape(tf.nn.bias_add(Wconv, biases), Wconv.get_shape())

        return Wconv_plus_b





def deconv2d(x, output_shape,weight_decay_rate,
             kh=5, kw=5, sh=2, sw=2,
             scope="deconv2d",
             initializer='None',
             parameter_update_device='-1',
             weight_decay=False,
             name_prefix='None',
             padding='SAME'):
    weight_stddev = 0.02
    bias_init_stddev = 0.0


    with tf.variable_scope(scope):
        # filter : [height, width, output_channels, in_channels]
        input_shape = x.get_shape().as_list()

        if initializer == 'NormalInit':
            W = variable_creation_on_device('W',shape=[kh, kw, output_shape[-1], input_shape[-1]],
                                            initializer=tf.random_normal_initializer(stddev=weight_stddev),
                                            parameter_update_device=parameter_update_device)
        else:
            W = variable_creation_on_device('W', shape=[kh, kw, output_shape[-1], input_shape[-1]],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                            parameter_update_device=parameter_update_device)

        if weight_decay:
            weight_decay = tf.multiply(tf.nn.l2_loss(W), weight_decay_rate, name='weight_decay')
            if not name_prefix.find('/') == -1:
                tf.add_to_collection(name_prefix[0:name_prefix.find('/')] + '_weight_decay', weight_decay)
            else:
                tf.add_to_collection(name_prefix + '_weight_decay', weight_decay)

        deconv = tf.nn.conv2d_transpose(x, W, output_shape=output_shape,
                                        strides=[1, sh, sw, 1],
                                        padding=padding)

        biases = variable_creation_on_device('b',
                                             shape=[output_shape[-1]],
                                             initializer=tf.constant_initializer(bias_init_stddev),
                                             parameter_update_device=parameter_update_device)
        deconv_plus_b = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv_plus_b




def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def relu (x):
    return tf.nn.relu(features=x)








def fc(x,
       output_size,weight_decay_rate,
       scope="fc",
       initializer='None',
       parameter_update_device='-1',
       name_prefix='None',
       weight_decay=False):
    weight_stddev = 0.02
    bias_stddev = 0.0
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()

        if initializer == 'NormalInit':
            W = variable_creation_on_device(name="W",
                                            shape=[shape[1], output_size],
                                            initializer=tf.random_normal_initializer(stddev=weight_stddev),
                                            parameter_update_device=parameter_update_device)
        elif initializer =='XavierInit':
            W = variable_creation_on_device(name="W",
                                            shape=[shape[1], output_size],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                            parameter_update_device=parameter_update_device)

        if weight_decay:
            weight_decay = tf.multiply(tf.nn.l2_loss(W), weight_decay_rate, name='weight_decay')
            if not name_prefix.find('/') == -1:
                tf.add_to_collection(name_prefix[0:name_prefix.find('/')] + '_weight_decay', weight_decay)
            else:
                tf.add_to_collection(name_prefix + '_weight_decay', weight_decay)


        b = variable_creation_on_device("b", shape=[output_size],
                                        initializer=tf.constant_initializer(bias_stddev),
                                        parameter_update_device=parameter_update_device)
        return tf.matmul(x, W) + b


def variable_creation_on_device(name,
                                shape,
                                initializer,
                                parameter_update_device='-1'):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device(parameter_update_device):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

