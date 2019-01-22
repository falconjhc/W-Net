# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

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

def emd_mixer(content,style,initializer,device='-1',scope="emd_mixer"):
    weight_stddev = 0.02
    shape_dimension = int(content.shape[1])
    K = shape_dimension
    bias_init_stddev = 0.0

    with tf.variable_scope(scope):

        if initializer == 'NormalInit':
            W = variable_creation_on_device(name='W',
                                            shape=[shape_dimension, K, shape_dimension],
                                            initializer=tf.random_normal_initializer(stddev=weight_stddev),
                                            parameter_update_device=device)

        else:
            W = variable_creation_on_device(name='W',
                                            shape=[shape_dimension, shape_dimension, shape_dimension],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                            parameter_update_device=device)

        travel_times = int(W.shape[2])
        for ii in range(travel_times):
            current_W = W[:,:,ii]
            current_cal1 = tf.expand_dims(tf.matmul(style,current_W),axis=2)
            if ii == 0:
                cal1 = current_cal1
            else:
                cal1=tf.concat([cal1,current_cal1],axis=2)

        batch_size = int(cal1.shape[0])
        for ii in range(batch_size):
            current_cal1 = cal1[ii,:,:]
            current_cal2 = tf.expand_dims(tf.transpose(tf.matmul(current_cal1,
                                                                 tf.transpose(content))),
                                          axis=0)
            if ii == 0:
                cal2 = current_cal2
            else:
                cal2 = tf.concat([cal2,current_cal2],axis=0)

        cal2 = tf.reduce_sum(cal2,axis=0)
        return cal2


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


def dilated_conv2d(x,
                   output_filters,
                   weight_decay_rate,
                   kh=5, kw=5, dilation=2,
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

        Wconv = tf.nn.atrous_conv2d(x, W, dilation, padding=padding)

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




def resblock(x,initializer,
             layer,sh,sw,kh,kw,batch_norm_used,is_training,
             weight_decay,weight_decay_rate,
             scope="resblock",
             parameter_update_devices='-1',
             ):
    filters = int(x.shape[3])
    with tf.variable_scope(scope):
        # filter : [height, width, output_channels, in_channels]
        conv1 = conv2d(x=x,
                       output_filters=filters,
                       scope="layer%d_conv1" % layer,
                       parameter_update_device=parameter_update_devices,
                       kh=kh,kw=kw,sh=sh,sw=sw,
                       initializer=initializer,
                       weight_decay=weight_decay,
                       name_prefix=scope,
                       weight_decay_rate=weight_decay_rate)

        if batch_norm_used:
            bn1 = batch_norm(x=conv1,
                             is_training=is_training,
                             scope="layer%d_bn1" % layer,
                             parameter_update_device=parameter_update_devices)

        act1 = lrelu(bn1)


        conv2 = conv2d(x=act1,
                       output_filters=filters,
                       scope="layer%d_conv2" % layer,
                       parameter_update_device=parameter_update_devices,
                       kh=kh,kw=kw,sh=sh,sw=sw,
                       initializer=initializer,
                       weight_decay=weight_decay,
                       name_prefix=scope,
                       weight_decay_rate=weight_decay_rate)

        if batch_norm_used:
            bn2 = batch_norm(x=conv2,
                             is_training=is_training,
                             scope="layer%d_bn2" % layer,
                             parameter_update_device=parameter_update_devices)

        act2 = lrelu(bn2)

        return act2 + x



def dilated_conv_resblock(x,initializer,
                          layer,dilation,kh,kw,is_training,batch_norm_used,
                          weight_decay,weight_decay_rate,
                          scope="dilated_resblock",
                          parameter_update_devices='-1'):
    filters = int(x.shape[3])
    with tf.variable_scope(scope):
        dilated_conv1 = dilated_conv2d(x=x,
                                       output_filters=filters,
                                       scope="layer%d_conv1" % layer,
                                       parameter_update_device=parameter_update_devices,
                                       kh=kh,kw=kw,dilation=dilation,
                                       initializer=initializer,
                                       weight_decay=weight_decay,
                                       name_prefix=scope,
                                       weight_decay_rate=weight_decay_rate)
        if batch_norm_used:
            bn1 = batch_norm(x=dilated_conv1,
                             is_training=is_training,
                             scope="layer%d_bn1" % layer,
                             parameter_update_device=parameter_update_devices)
        else:
            bn1 = dilated_conv1
        act1 = relu(bn1)

        if is_training:
            drop1 = tf.nn.dropout(act1,0.5)
        else:
            drop1 = act1

        dilated_conv2 = dilated_conv2d(x=drop1,
                                       output_filters=filters,
                                       scope="layer%d_conv2" % layer,
                                       parameter_update_device=parameter_update_devices,
                                       kh=kh,kw=kw,dilation=dilation,
                                       initializer=initializer,
                                       weight_decay=weight_decay,
                                       name_prefix=scope,
                                       weight_decay_rate=weight_decay_rate)

        return x + dilated_conv2


def normal_conv_resblock(x,initializer,is_training,
                         layer,kh,kw,sh,sw,batch_norm_used,
                         weight_decay,weight_decay_rate,
                         scope="dilated_resblock",
                         parameter_update_devices='-1'):
    filters = int(x.shape[3])
    with tf.variable_scope(scope):
        conv1 = conv2d(x=x,
                       output_filters=filters,
                       scope="layer%d_conv1" % layer,
                       parameter_update_device=parameter_update_devices,
                       kh=kh, kw=kw, sh=sh,sw=sw,
                       initializer=initializer,
                       weight_decay=weight_decay,
                       name_prefix=scope,
                       weight_decay_rate=weight_decay_rate)
        if batch_norm_used:
            bn1 = batch_norm(x=conv1,
                             is_training=is_training,
                             scope="layer%d_bn1" % layer,
                             parameter_update_device=parameter_update_devices)
        else:
            bn1 = conv1
        act1 = relu(bn1)

        if is_training:
            drop1 = tf.nn.dropout(act1, 0.5)
        else:
            drop1 = act1

        conv2 = conv2d(x=drop1,
                       output_filters=filters,
                       scope="layer%d_conv2" % layer,
                       parameter_update_device=parameter_update_devices,
                       kh=kh, kw=kw, sh=sh,sw=sw,
                       initializer=initializer,
                       weight_decay=weight_decay,
                       name_prefix=scope,
                       weight_decay_rate=weight_decay_rate)

        return x + conv2



def global_average_pooling(x):
    return tf.squeeze(tf.reduce_mean(x,axis=[1,2]))


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

