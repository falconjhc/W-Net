
import tensorflow as tf
import sys
sys.path.append('..')


import numpy as np
from utilities.ops import lrelu, relu, batch_norm, layer_norm, instance_norm, adaptive_instance_norm, resblock, desblock
from utilities.ops import conv2d, deconv2d, fc, dilated_conv2d, dilated_conv_resblock, normal_conv_resblock
from utilities.ops import emd_mixer


import math

print_separater="#########################################################"

eps = 1e-9
generator_dim = 64


def _calculate_batch_diff(input_feature):
    diff = tf.abs(tf.expand_dims(input_feature, 4) -
                  tf.expand_dims(tf.transpose(input_feature, [1, 2, 3, 0]), 0))
    diff = tf.reduce_sum(tf.exp(-diff), 4)
    return tf.reduce_mean(diff)


##############################################################################################
##############################################################################################
##############################################################################################
### Encoders #################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
def encoder_framework(images,
                      is_training,
                      encoder_device,
                      residual_at_layer,
                      residual_connection_mode,
                      scope,initializer,weight_decay,
                      weight_decay_rate,
                      reuse = False,
                      adain_use=False):
    def encoder(x, output_filters, layer):

        act = lrelu(x)
        conv = conv2d(x=act,
                      output_filters=output_filters,
                      scope="layer%d_conv" % layer,
                      parameter_update_device=encoder_device,
                      initializer=initializer,
                      weight_decay=weight_decay,
                      name_prefix=scope,
                      weight_decay_rate=weight_decay_rate)
        if not adain_use:
            enc = batch_norm(conv, is_training, scope="layer%d_bn" % layer,
                             parameter_update_device=encoder_device)
        elif adain_use==True and 'content' in scope:
            enc = instance_norm(x=conv, scope="layer%d_in" % layer,
                                parameter_update_device=encoder_device)
        else:
            enc = conv
        return enc


    return_str = "Encoder %d Layers" % int(np.floor(math.log(int(images.shape[1])) / math.log(2)))
    if not residual_at_layer == -1:
        return_str = return_str + " with residual blocks at layer %d" % residual_at_layer



    residual_input_list = list()
    full_feature_list = list()
    shortcut_list = list()
    batch_size = int(images.shape[0])


    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(encoder_device):
            with tf.variable_scope(scope):
                if reuse:
                    tf.get_variable_scope().reuse_variables()


                feature_size = int(images.shape[1])
                ii=0
                while not feature_size==1:
                    if ii == 0:
                        encoder_layer = conv2d(x=images,
                                               output_filters=generator_dim,
                                               scope="layer1_conv",
                                               parameter_update_device=encoder_device,
                                               initializer=initializer,
                                               weight_decay=weight_decay,
                                               name_prefix=scope,
                                               weight_decay_rate=weight_decay_rate)
                    else:
                        output_filter_num_expansion = np.min([np.power(2,ii),8])
                        output_filter_num = generator_dim * output_filter_num_expansion
                        encoder_layer = encoder(x=encoder_layer,
                                                output_filters=output_filter_num,
                                                layer=ii+1)
                    full_feature_list.append(encoder_layer)


                    feature_size = int(encoder_layer.shape[1])
                    if feature_size==1:
                        output_final_encoded = encoder_layer

                    residual_condition = (residual_connection_mode == 'Single' and (ii + 1 == residual_at_layer)) \
                                         or (residual_connection_mode == 'Multi' and (ii + 1 <= residual_at_layer))

                    # output for residual blocks
                    if residual_condition:
                        residual_input_list.append(encoder_layer)

                    # output for shortcut
                    if ii+1 > residual_at_layer:
                        shortcut_list.append(encoder_layer)
                    ii+=1

        return output_final_encoded, \
               shortcut_list, residual_input_list, full_feature_list, \
               return_str


def encoder_resemd_framework(images,
                             is_training,
                             encoder_device,
                             scope,initializer,
                             weight_decay,weight_decay_rate,
                             residual_at_layer=-1,
                             residual_connection_mode=None,
                             reuse=False,
                             adain_use=False):
    residual_connection_mode=None
    residual_at_layer=-1
    adain_use=False
    full_feature_list = list()
    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(encoder_device):
            with tf.variable_scope(scope):
                if reuse:
                    tf.get_variable_scope().reuse_variables()

                conv1 = lrelu(conv2d(x=images,
                                     output_filters=64,
                                     kh=5,kw=5, sh=1, sw=1,
                                     scope="layer%d_conv" % 1,
                                     parameter_update_device=encoder_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,
                                     weight_decay_rate=weight_decay_rate))
                full_feature_list.append(conv1)

                conv2 = lrelu(conv2d(x=conv1,
                                     output_filters=128,
                                     kh=3, kw=3, sh=2, sw=2,
                                     scope="layer%d_conv" % 2,
                                     parameter_update_device=encoder_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,
                                     weight_decay_rate=weight_decay_rate))
                full_feature_list.append(conv2)

                conv3 = lrelu(conv2d(x=conv2,
                                     output_filters=256,
                                     kh=3, kw=3, sh=2, sw=2,
                                     scope="layer%d_conv" % 3,
                                     parameter_update_device=encoder_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,
                                     weight_decay_rate=weight_decay_rate))
                full_feature_list.append(conv3)

                conv4 = lrelu(conv2d(x=conv3,
                                     output_filters=256,
                                     kh=3, kw=3, sh=2, sw=2,
                                     scope="layer%d_conv" % 4,
                                     parameter_update_device=encoder_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,
                                     weight_decay_rate=weight_decay_rate))
                full_feature_list.append(conv4)

                res1 = resblock(x=conv4,
                                initializer=initializer,
                                layer=5, kh=3, kw=3, sh=1, sw=1,
                                batch_norm_used=True,is_training=is_training,
                                weight_decay=weight_decay,weight_decay_rate=weight_decay_rate,
                                scope="layer%d_resblock" % 5,
                                parameter_update_devices=encoder_device)
                full_feature_list.append(res1)

                res2 = resblock(x=res1,
                                initializer=initializer,
                                layer=6, kh=3, kw=3, sh=1, sw=1,
                                batch_norm_used=True, is_training=is_training,
                                weight_decay=weight_decay, weight_decay_rate=weight_decay_rate,
                                scope="layer%d_resblock" % 6,
                                parameter_update_devices=encoder_device)
                full_feature_list.append(res2)

                res3 = resblock(x=res2,
                                initializer=initializer,
                                layer=7, kh=3, kw=3, sh=1, sw=1,
                                batch_norm_used=True, is_training=is_training,
                                weight_decay=weight_decay, weight_decay_rate=weight_decay_rate,
                                scope="layer%d_resblock" % 7,
                                parameter_update_devices=encoder_device)
                full_feature_list.append(res3)

                res4 = resblock(x=res3,
                                initializer=initializer,
                                layer=8, kh=3, kw=3, sh=1, sw=1,
                                batch_norm_used=True, is_training=is_training,
                                weight_decay=weight_decay, weight_decay_rate=weight_decay_rate,
                                scope="layer%d_resblock" % 8,
                                parameter_update_devices=encoder_device)
                full_feature_list.append(res4)


                return_str = "ResEmdNet-Encoder %d Layers" % (len(full_feature_list))

    return res4, -1, -1, full_feature_list, return_str


def encoder_adobenet_framework(images,
                               is_training,
                               encoder_device,
                               scope,initializer,
                               weight_decay,weight_decay_rate,
                               residual_at_layer=-1,
                               residual_connection_mode=None,
                               reuse=False,
                               adain_use=False):
    residual_connection_mode = None
    residual_at_layer = -1
    adain_use = False
    full_feature_list = list()

    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(encoder_device):
            with tf.variable_scope(scope):
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                conv1 = relu(conv2d(x=images,
                                    output_filters=64,
                                    kh=7,kw=7, sh=1, sw=1,
                                    scope="layer%d_conv" % 1,
                                    parameter_update_device=encoder_device,
                                    initializer=initializer,
                                    weight_decay=weight_decay,
                                    name_prefix=scope,
                                    weight_decay_rate=weight_decay_rate))
                full_feature_list.append(conv1)

                return_str = "AdobeNet-Encoder %d Layers" % (len(full_feature_list))

    return conv1, -1, -1, full_feature_list, return_str


def encoder_resmixernet_framework(images,
                                  is_training,
                                  encoder_device,
                                  scope,initializer,
                                  weight_decay,weight_decay_rate,
                                  residual_at_layer=-1,
                                  residual_connection_mode=None,
                                  reuse=False,
                                  adain_use=False):
    residual_connection_mode = None
    residual_at_layer = -1
    adain_use = False
    full_feature_list = list()

    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(encoder_device):
            with tf.variable_scope(scope):
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                conv1 = relu(instance_norm(conv2d(x=images,
                                                  output_filters=64,
                                                  kh=7,kw=7, sh=1, sw=1,
                                                  scope="layer%d_conv" % 1,
                                                  parameter_update_device=encoder_device,
                                                  initializer=initializer,
                                                  weight_decay=weight_decay,
                                                  name_prefix=scope,
                                                  weight_decay_rate=weight_decay_rate),
                                           scope="layer%d_in" % 1,
                                           parameter_update_device=encoder_device))
                full_feature_list.append(conv1)

                conv2 = relu(instance_norm(conv2d(x=conv1,
                                                  output_filters=128,
                                                  kh=3, kw=3, sh=2, sw=2,
                                                  scope="layer%d_conv" % 2,
                                                  parameter_update_device=encoder_device,
                                                  initializer=initializer,
                                                  weight_decay=weight_decay,
                                                  name_prefix=scope,
                                                  weight_decay_rate=weight_decay_rate),
                                           scope="layer%d_in" % 2,
                                           parameter_update_device=encoder_device))
                full_feature_list.append(conv2)

                conv3 = relu(instance_norm(conv2d(x=conv2,
                                                  output_filters=256,
                                                  kh=3, kw=3, sh=2, sw=2,
                                                  scope="layer%d_conv" % 3,
                                                  parameter_update_device=encoder_device,
                                                  initializer=initializer,
                                                  weight_decay=weight_decay,
                                                  name_prefix=scope,
                                                  weight_decay_rate=weight_decay_rate),
                                           scope="layer%d_in" % 3,
                                           parameter_update_device=encoder_device))
                full_feature_list.append(conv3)





                return_str = "DenseResNet-Encoder %d Layers" % (len(full_feature_list))

    return conv3, -1, -1, full_feature_list, return_str

