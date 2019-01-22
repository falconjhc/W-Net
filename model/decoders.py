
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




##############################################################################################
##############################################################################################
##############################################################################################
### Decoders #################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
def wnet_decoder_framework(encoded_layer_list,
                           decoder_input_org,
                           is_training,
                           output_width,
                           output_filters,
                           batch_size,
                           decoder_device,
                           scope,initializer, weight_decay,weight_decay_rate,
                           adain_use,
                           reuse=False,
                           other_info=None):
    def decoder(x,
                output_width,
                output_filters,
                layer,
                enc_layer,
                do_norm=False,
                dropout=False):
        dec = deconv2d(x=tf.nn.relu(x),
                       output_shape=[batch_size, output_width, output_width, output_filters],
                       scope="layer%d_conv" % layer,
                       parameter_update_device=decoder_device,
                       weight_decay=weight_decay,initializer=initializer,
                       name_prefix=scope,
                       weight_decay_rate=weight_decay_rate)
        if do_norm:
            # IMPORTANT: normalization for last layer
            # Very important, otherwise GAN is unstable
            if not adain_use:
                dec = batch_norm(dec, is_training, scope="layer%d_bn" % layer,
                                 parameter_update_device=decoder_device)
            else:
                dec = layer_norm(x=dec, scope="layer%d_ln" % layer,
                                 parameter_update_device=decoder_device)

        if dropout:
            dec = tf.nn.dropout(dec, 0.5)

        if not enc_layer == None:
            dec = tf.concat([dec, enc_layer], 3)
        return dec

    decoder_input = decoder_input_org
    return_str = "WNet-Decoder %d Layers" % int(np.floor(math.log(output_width) / math.log(2)))
    full_decoded_feature_list = list()

    full_encoder_layer_num = int(np.floor(math.log(output_width) / math.log(2)))
    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(decoder_device):
            with tf.variable_scope(scope):
                if reuse:
                    tf.get_variable_scope().reuse_variables()

                feature_size = int(decoder_input.shape[1])
                ii=0
                while not feature_size == output_width:
                    power_times = full_encoder_layer_num-2-ii
                    output_feature_size = output_width / np.power(2, full_encoder_layer_num - ii - 1)
                    if ii < full_encoder_layer_num-1:
                        output_filter_num_expansion = np.min([np.power(2,power_times), 8])
                        output_filter_num = generator_dim * output_filter_num_expansion
                        do_norm = True
                        do_drop= True and is_training
                        encoded_respective = encoded_layer_list[ii + 1]

                    else:
                        output_filter_num = output_filters
                        do_norm = False
                        do_drop = False
                        encoded_respective = None

                    decoder_output = decoder(x=decoder_input,
                                             output_width=output_feature_size,
                                             output_filters=output_filter_num,
                                             layer=ii + 1,
                                             enc_layer=encoded_respective,
                                             do_norm=do_norm,
                                             dropout=do_drop)
                    full_decoded_feature_list.append(decoder_output)

                    if ii == full_encoder_layer_num-1:
                        output = tf.nn.tanh(decoder_output)
                    ii+=1
                    decoder_input = decoder_output
                    feature_size = int(decoder_input.shape[1])

        return output, full_decoded_feature_list, return_str

def emdnet_decoder_framework(encoded_layer_list,
                             decoder_input_org,
                             is_training,
                             output_width,
                             output_filters,
                             batch_size,
                             decoder_device,
                             scope,initializer, weight_decay,weight_decay_rate,
                             adain_use,
                             reuse=False,
                             other_info=None):
    def decoder(x,
                output_width,
                output_filters,
                layer,
                do_norm=False,
                dropout=False):
        dec = deconv2d(x=tf.nn.relu(x),
                       output_shape=[batch_size, output_width, output_width, output_filters],
                       scope="layer%d_conv" % layer,
                       parameter_update_device=decoder_device,
                       weight_decay=weight_decay,initializer=initializer,
                       name_prefix=scope,
                       weight_decay_rate=weight_decay_rate)
        if do_norm:
            # IMPORTANT: normalization for last layer
            # Very important, otherwise GAN is unstable
            if not adain_use:
                dec = batch_norm(dec, is_training, scope="layer%d_bn" % layer,
                                 parameter_update_device=decoder_device)
            else:
                dec = layer_norm(x=dec, scope="layer%d_ln" % layer,
                                 parameter_update_device=decoder_device)

        if dropout:
            dec = tf.nn.dropout(dec, 0.5)

        return dec

    decoder_input = decoder_input_org
    return_str = "EmdNet-Decoder %d Layers" % int(np.floor(math.log(output_width) / math.log(2)))
    full_decoded_feature_list = list()

    full_encoder_layer_num = int(np.floor(math.log(output_width) / math.log(2)))
    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(decoder_device):
            with tf.variable_scope(scope):
                if reuse:
                    tf.get_variable_scope().reuse_variables()

                feature_size = int(decoder_input.shape[1])
                ii=0
                while not feature_size == output_width:
                    power_times = full_encoder_layer_num-2-ii
                    output_feature_size = output_width / np.power(2, full_encoder_layer_num - ii - 1)
                    if ii < full_encoder_layer_num-1:
                        output_filter_num_expansion = np.min([np.power(2,power_times), 8])
                        output_filter_num = generator_dim * output_filter_num_expansion
                        do_norm = True
                        do_drop= True and is_training


                    else:
                        output_filter_num = output_filters
                        do_norm = False
                        do_drop = False

                    encoded_respective = encoded_layer_list[ii]
                    if not encoded_respective == None:
                        decoder_input = tf.concat([decoder_input, encoded_respective], axis=3)

                    decoder_output = decoder(x=decoder_input,
                                             output_width=output_feature_size,
                                             output_filters=output_filter_num,
                                             layer=ii + 1,
                                             do_norm=do_norm,
                                             dropout=do_drop)
                    full_decoded_feature_list.append(decoder_output)

                    if ii == full_encoder_layer_num-1:
                        output = tf.nn.tanh(decoder_output)
                    ii+=1
                    decoder_input = decoder_output
                    feature_size = int(decoder_input.shape[1])

        return output, full_decoded_feature_list, return_str


def decoder_resemdnet_framework(encoded_layer_list,
                                decoder_input_org,
                                is_training,
                                output_width,
                                output_filters,
                                batch_size,
                                decoder_device,
                                scope,initializer, weight_decay,weight_decay_rate,
                                adain_use,
                                reuse=False,
                                other_info=None):

    residual_connection_mode = None
    residual_at_layer = -1
    adain_use = False
    full_feature_list = list()
    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(decoder_device):
            with tf.variable_scope(scope):
                if reuse:
                    tf.get_variable_scope().reuse_variables()

                res1 = resblock(x=decoder_input_org,
                                initializer=initializer,
                                layer=1, kh=3, kw=3, sh=1, sw=1,
                                batch_norm_used=True, is_training=is_training,
                                weight_decay=weight_decay, weight_decay_rate=weight_decay_rate,
                                scope="layer%d_resblock" % 1,
                                parameter_update_devices=decoder_device)
                full_feature_list.append(res1)

                res2 = resblock(x=res1,
                                initializer=initializer,
                                layer=2, kh=3, kw=3, sh=1, sw=1,
                                batch_norm_used=True, is_training=is_training,
                                weight_decay=weight_decay, weight_decay_rate=weight_decay_rate,
                                scope="layer%d_resblock" % 2,
                                parameter_update_devices=decoder_device)
                full_feature_list.append(res2)

                res3 = resblock(x=res2,
                                initializer=initializer,
                                layer=3, kh=3, kw=3, sh=1, sw=1,
                                batch_norm_used=True, is_training=is_training,
                                weight_decay=weight_decay, weight_decay_rate=weight_decay_rate,
                                scope="layer%d_resblock" % 3,
                                parameter_update_devices=decoder_device)
                full_feature_list.append(res3)

                res4 = resblock(x=res3,
                                initializer=initializer,
                                layer=4, kh=3, kw=3, sh=1, sw=1,
                                batch_norm_used=True, is_training=is_training,
                                weight_decay=weight_decay, weight_decay_rate=weight_decay_rate,
                                scope="layer%d_resblock" % 4,
                                parameter_update_devices=decoder_device)
                full_feature_list.append(res4)

                deconv1 = lrelu(deconv2d(x=res4,
                                         kh=3,kw=3,sh=2,sw=2,
                                         output_shape=[batch_size, int(res4.shape[2])*2,
                                                       int(res4.shape[2])*2, 256],
                                         scope="layer%d_deconv" % 5,
                                         parameter_update_device=decoder_device,
                                         initializer=initializer,
                                         weight_decay=weight_decay,
                                         name_prefix=scope,
                                         weight_decay_rate=weight_decay_rate))
                full_feature_list.append(deconv1)


                if other_info==None:
                    deconv2 = lrelu(deconv2d(x=deconv1,
                                             output_shape=[batch_size, int(deconv1.shape[2]) * 2,
                                                           int(deconv1.shape[2]) * 2, 128],
                                             kh=3, kw=3, sh=2, sw=2,
                                             scope="layer%d_deconv" % 6,
                                             parameter_update_device=decoder_device,
                                             initializer=initializer,
                                             weight_decay=weight_decay,
                                             name_prefix=scope,
                                             weight_decay_rate=weight_decay_rate))
                elif other_info=='NN':
                    deconv2 = lrelu(deconv2d(x=deconv1,
                                             output_shape=[batch_size, int(deconv1.shape[2]),
                                                           int(deconv1.shape[2]), 128],
                                             kh=3, kw=3, sh=1, sw=1,
                                             scope="layer%d_deconv" % 6,
                                             parameter_update_device=decoder_device,
                                             initializer=initializer,
                                             weight_decay=weight_decay,
                                             name_prefix=scope,
                                             weight_decay_rate=weight_decay_rate))

                    deconv2 = tf.image.resize_nearest_neighbor(images=deconv2,
                                                               size=[int(deconv2.shape[2])*2, int(deconv2.shape[2])*2])
                full_feature_list.append(deconv2)

                if other_info == None:
                    deconv3 = lrelu(deconv2d(x=deconv2,
                                             output_shape=[batch_size, int(deconv2.shape[2]) * 2,
                                                           int(deconv2.shape[2]) * 2, 64],
                                             kh=3, kw=3, sh=2, sw=2,
                                             scope="layer%d_deconv" % 7,
                                             parameter_update_device=decoder_device,
                                             initializer=initializer,
                                             weight_decay=weight_decay,
                                             name_prefix=scope,
                                             weight_decay_rate=weight_decay_rate))
                elif other_info=='NN':
                    deconv3 = lrelu(deconv2d(x=deconv2,
                                             output_shape=[batch_size, int(deconv2.shape[2]),
                                                           int(deconv2.shape[2]), 64],
                                             kh=3, kw=3, sh=1, sw=1,
                                             scope="layer%d_deconv" % 7,
                                             parameter_update_device=decoder_device,
                                             initializer=initializer,
                                             weight_decay=weight_decay,
                                             name_prefix=scope,
                                             weight_decay_rate=weight_decay_rate))
                    deconv3 = tf.image.resize_nearest_neighbor(images=deconv3,
                                                               size=[int(deconv3.shape[2]) * 2, int(deconv3.shape[2]) * 2])

                full_feature_list.append(deconv3)

                deconv4 = tf.nn.tanh(deconv2d(x=deconv3,
                                              output_shape=[batch_size,output_width,
                                                            output_width, output_filters],
                                              kh=5, kw=5, sh=1, sw=1,
                                              scope="layer%d_deconv" % 8,
                                              parameter_update_device=decoder_device,
                                              initializer=initializer,
                                              weight_decay=weight_decay,
                                              name_prefix=scope,
                                              weight_decay_rate=weight_decay_rate))
                full_feature_list.append(deconv4)

                return_str = "ResEmdNet-Decoder %d Layers" % len(full_feature_list)

                return deconv4, full_feature_list, return_str


def decoder_adobenet_framework(encoded_layer_list,
                               decoder_input_org,
                               is_training,
                               output_width,
                               output_filters,
                               batch_size,
                               decoder_device,
                               scope,initializer, weight_decay,weight_decay_rate,
                               adain_use,
                               reuse=False,
                               other_info=None):

    residual_connection_mode = None
    residual_at_layer = -1
    adain_use = False
    full_feature_list = list()
    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(decoder_device):
            with tf.variable_scope(scope):
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                normal_conv_resblock1 = normal_conv_resblock(x=decoder_input_org,
                                                             initializer=initializer,
                                                             is_training=is_training,
                                                             layer=1,
                                                             kh=3, kw=3, sh=1, sw=1,
                                                             batch_norm_used=True,
                                                             weight_decay=weight_decay,
                                                             weight_decay_rate=weight_decay_rate,
                                                             scope="layer%d_normal_resblock" % 1,
                                                             parameter_update_devices=decoder_device)
                full_feature_list.append(normal_conv_resblock1)

                dilated_conv_resblock1 = dilated_conv_resblock(x=normal_conv_resblock1,
                                                               initializer=initializer,
                                                               is_training=is_training,
                                                               layer=2,
                                                               dilation=2, kh=3, kw=3,
                                                               batch_norm_used=True,
                                                               weight_decay=weight_decay,
                                                               weight_decay_rate=weight_decay_rate,
                                                               scope="layer%d_dilated_resblock" % 2,
                                                               parameter_update_devices=decoder_device)
                full_feature_list.append(dilated_conv_resblock1)

                dilated_conv_resblock2 = dilated_conv_resblock(x=dilated_conv_resblock1,
                                                               initializer=initializer,
                                                               is_training=is_training,
                                                               layer=3,
                                                               dilation=4, kh=3, kw=3,
                                                               batch_norm_used=True,
                                                               weight_decay=weight_decay,
                                                               weight_decay_rate=weight_decay_rate,
                                                               scope="layer%d_dilated_resblock" % 3,
                                                               parameter_update_devices=decoder_device)
                full_feature_list.append(dilated_conv_resblock2)

                dilated_conv_1 = relu(batch_norm(x=dilated_conv2d(x=dilated_conv_resblock2,
                                                                  output_filters=128,
                                                                  weight_decay_rate=weight_decay_rate, weight_decay=weight_decay,
                                                                  kh=3, kw=3, dilation=2,
                                                                  initializer=initializer,
                                                                  scope="layer%d_dilated_conv" % 4,
                                                                  parameter_update_device=decoder_device,
                                                                  name_prefix=scope),
                                                 is_training=is_training,
                                                 scope="layer%d_bn"% 4,
                                                 parameter_update_device=decoder_device))


                full_feature_list.append(dilated_conv_1)

                generated_img = tf.nn.tanh(conv2d(x=dilated_conv_1,
                                                  output_filters=1,
                                                  weight_decay_rate=weight_decay_rate, weight_decay=weight_decay,
                                                  kh=3, kw=3, sw=1, sh=1,
                                                  initializer=initializer,
                                                  scope="layer%d_normal_conv" % 5,
                                                  parameter_update_device=decoder_device,
                                                  name_prefix=scope))
                full_feature_list.append(generated_img)

    return_str = "AdobeNet-Decoder %d Layers" % len(full_feature_list)

    return generated_img, full_feature_list, return_str


def decoder_resmixernet_framework(decoder_input_org,
                                  batch_size,
                                  decoder_device,
                                  scope,initializer, weight_decay,weight_decay_rate,
                                  reuse=False):

    residual_connection_mode = None
    residual_at_layer = -1
    adain_use = False
    full_feature_list = list()
    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(decoder_device):
            with tf.variable_scope(scope):
                if reuse:
                    tf.get_variable_scope().reuse_variables()

                deconv1 = lrelu(instance_norm(x=deconv2d(x=decoder_input_org,
                                                         kh=3, kw=3, sh=2, sw=2,
                                                         output_shape=[batch_size, int(decoder_input_org.shape[2]) * 2,
                                                                       int(decoder_input_org.shape[2]) * 2, 128],
                                                         scope="layer%d_deconv" % 1,
                                                         parameter_update_device=decoder_device,
                                                         initializer=initializer,
                                                         weight_decay=weight_decay,
                                                         name_prefix=scope,
                                                         weight_decay_rate=weight_decay_rate),
                                              scope="layer%d_in" % 1,
                                              parameter_update_device=decoder_device))
                full_feature_list.append(deconv1)

                deconv2 = lrelu(instance_norm(x=deconv2d(x=deconv1,
                                                         kh=3, kw=3, sh=2, sw=2,
                                                         output_shape=[batch_size, int(deconv1.shape[2]) * 2,
                                                                       int(deconv1.shape[2]) * 2, 64],
                                                         scope="layer%d_deconv" % 2,
                                                         parameter_update_device=decoder_device,
                                                         initializer=initializer,
                                                         weight_decay=weight_decay,
                                                         name_prefix=scope,
                                                         weight_decay_rate=weight_decay_rate),
                                              scope="layer%d_in" % 2,
                                              parameter_update_device=decoder_device))
                full_feature_list.append(deconv2)

                deconv3 = lrelu(instance_norm(x=deconv2d(x=deconv2,
                                                         kh=7, kw=7, sh=1, sw=1,
                                                         output_shape=[batch_size, int(deconv2.shape[2]),
                                                                       int(deconv2.shape[2]), 1],
                                                         scope="layer%d_deconv" % 3,
                                                         parameter_update_device=decoder_device,
                                                         initializer=initializer,
                                                         weight_decay=weight_decay,
                                                         name_prefix=scope,
                                                         weight_decay_rate=weight_decay_rate),
                                              scope="layer%d_in" % 3,
                                              parameter_update_device=decoder_device))
                full_feature_list.append(deconv3)
    return deconv3, full_feature_list


