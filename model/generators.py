
import tensorflow as tf
import sys
sys.path.append('..')


import numpy as np
from utilities.ops import lrelu, relu, batch_norm, layer_norm, instance_norm, adaptive_instance_norm, resblock, desblock
from utilities.ops import conv2d, deconv2d, fc, dilated_conv2d, dilated_conv_resblock, normal_conv_resblock
from utilities.ops import emd_mixer
from .mixers import wnet_feature_mixer_framework, emdnet_mixer_with_adain,emdnet_mixer_non_adain,resmixer
from .encoders import encoder_framework, encoder_resemd_framework,encoder_adobenet_framework,encoder_resmixernet_framework
from .decoders import wnet_decoder_framework, emdnet_decoder_framework,decoder_resemdnet_framework,decoder_adobenet_framework,decoder_resmixernet_framework


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
### GeneratorFrameworks ######################################################################
##############################################################################################
##############################################################################################
##############################################################################################
def WNet_Generator(content_prototype,
                   style_reference,
                   is_training,
                   batch_size,
                   generator_device,
                   residual_at_layer,
                   residual_block_num,
                   scope,
                   initializer,
                   weight_decay, weight_decay_rate,
                   reuse=False,
                   adain_use=False,
                   adain_preparation_model=None,
                   debug_mode=True,
                   other_info=None):

    style_input_number = len(style_reference)
    content_prototype_number = int(content_prototype.shape[3])

    # content prototype encoder part
    encoded_content_final, content_short_cut_interface, content_residual_interface, content_full_feature_list, _ = \
        encoder_framework(images=content_prototype,
                          is_training=is_training,
                          encoder_device=generator_device,
                          residual_at_layer=residual_at_layer,
                          residual_connection_mode='Multi',
                          scope=scope + '/content_encoder',
                          reuse=reuse,
                          initializer=initializer,
                          weight_decay=weight_decay,
                          weight_decay_rate=weight_decay_rate,
                          adain_use=adain_use)

    # style reference encoder part
    encoded_style_final_list = list()
    style_short_cut_interface_list = list()
    style_residual_interface_list = list()
    full_style_feature_list = list()
    for ii in range(style_input_number):
        if ii==0:
            curt_reuse=reuse
            current_weight_decay = weight_decay
        else:
            curt_reuse=True
            current_weight_decay = False

        encoded_style_final, current_style_short_cut_interface, current_style_residual_interface, current_full_feature_list, _ = \
            encoder_framework(images=style_reference[ii],
                              is_training=is_training,
                              encoder_device=generator_device,
                              residual_at_layer=residual_at_layer,
                              residual_connection_mode='Single',
                              scope=scope + '/style_encoder',
                              reuse=curt_reuse,
                              initializer=initializer,
                              weight_decay=current_weight_decay,
                              weight_decay_rate=weight_decay_rate,
                              adain_use=adain_use)
        encoded_style_final_list.append(encoded_style_final)
        style_short_cut_interface_list.append(current_style_short_cut_interface)
        style_residual_interface_list.append(current_style_residual_interface)
        full_style_feature_list.append(current_full_feature_list)

    encoded_layer_list, style_shortcut_batch_diff, style_residual_batch_diff,encoded_style_final = \
        wnet_feature_mixer_framework(generator_device=generator_device,
                                     scope=scope+'/mixer',
                                     is_training=is_training,
                                     reuse=reuse,
                                     initializer=initializer,
                                     debug_mode=debug_mode,
                                     weight_decay=weight_decay,
                                     weight_decay_rate=weight_decay_rate,
                                     style_input_number=style_input_number,
                                     residual_block_num=residual_block_num,
                                     residual_at_layer=residual_at_layer,
                                     encoded_style_final_list=encoded_style_final_list,
                                     style_short_cut_interface_list=style_short_cut_interface_list,
                                     style_residual_interface_list=style_residual_interface_list,
                                     content_short_cut_interface=content_short_cut_interface,
                                     content_residual_interface=content_residual_interface,
                                     full_style_feature_list=full_style_feature_list,
                                     adain_use=adain_use,
                                     adain_preparation_model=adain_preparation_model,
                                     other_info=other_info)


    return_str = ("W-Net-GeneratorEncoderDecoder %d Layers with %d ResidualBlocks connecting %d-th layer"
                  % (int(np.floor(math.log(int(content_prototype[0].shape[1])) / math.log(2))),
                     residual_block_num,
                     residual_at_layer))

    # decoder part
    img_width = int(content_prototype.shape[1])
    img_filters = int(int(content_prototype.shape[3]) / content_prototype_number)
    generated_img,decoder_full_feature_list, _ = \
        wnet_decoder_framework(encoded_layer_list=encoded_layer_list,
                               decoder_input_org=encoded_layer_list[0],
                               is_training=is_training,
                               output_width=img_width,
                               output_filters=img_filters,
                               batch_size=batch_size,
                               decoder_device=generator_device,
                               scope=scope+'/decoder',
                               reuse=reuse,
                               weight_decay=weight_decay,
                               initializer=initializer,
                               weight_decay_rate=weight_decay_rate,
                               adain_use=adain_use)

    return generated_img, encoded_content_final, encoded_style_final, return_str, \
           style_shortcut_batch_diff, style_residual_batch_diff, \
           content_full_feature_list, full_style_feature_list, decoder_full_feature_list

def EmdNet_Generator(content_prototype,
                     style_reference,
                     is_training,
                     batch_size,
                     generator_device,
                     residual_at_layer,
                     residual_block_num,
                     scope,
                     initializer,
                     weight_decay, weight_decay_rate,
                     reuse=False,
                     adain_use=False,
                     adain_preparation_model=None,
                     debug_mode=True,
                     other_info=None):

    style_input_number = len(style_reference)
    content_prototype_number = int(content_prototype.shape[3])

    # content prototype encoder part
    encoded_content_final, content_shortcut_interface, _, content_full_feature_list, _ = \
        encoder_framework(images=content_prototype,
                          is_training=is_training,
                          encoder_device=generator_device,
                          residual_at_layer=residual_at_layer,
                          residual_connection_mode='Multi',
                          scope=scope + '/content_encoder',
                          reuse=reuse,
                          initializer=initializer,
                          weight_decay=weight_decay,
                          weight_decay_rate=weight_decay_rate,
                          adain_use=adain_use)

    # style reference encoder part
    for ii in range(len(style_reference)):
        if ii == 0:
            style_reference_tensor = style_reference[ii]
        else:
            style_reference_tensor = tf.concat([style_reference_tensor,style_reference[ii]],
                                               axis=3)
    encoded_style_final, _, _, style_full_feature_list, _ = \
        encoder_framework(images=style_reference_tensor,
                          is_training=is_training,
                          encoder_device=generator_device,
                          residual_at_layer=residual_at_layer,
                          residual_connection_mode='Single',
                          scope=scope + '/style_encoder',
                          reuse=reuse,
                          initializer=initializer,
                          weight_decay=weight_decay,
                          weight_decay_rate=weight_decay_rate,
                          adain_use=adain_use)

    # emd network mixer
    if adain_use==0:
        valid_encoded_content_shortcut_list, mixed_fc, batch_diff = \
            emdnet_mixer_non_adain(generator_device=generator_device,
                                   reuse=reuse, scope=scope+'/mixer', initializer=initializer,
                                   weight_decay=weight_decay, weight_decay_rate=weight_decay_rate,
                                   encoded_content_final=encoded_content_final,
                                   content_shortcut_interface=content_shortcut_interface,
                                   encoded_style_final=encoded_style_final)
    else:
        valid_encoded_content_shortcut_list, mixed_fc, batch_diff = \
            emdnet_mixer_with_adain(generator_device=generator_device,
                                    reuse=reuse, scope=scope+'/mixer', initializer=initializer,
                                    weight_decay=weight_decay, weight_decay_rate=weight_decay_rate,
                                    encoded_content_final=encoded_content_final,
                                    content_shortcut_interface=content_shortcut_interface,
                                    encoded_style_final=encoded_style_final)

    # decoder part
    img_width = int(content_prototype.shape[1])
    img_filters = int(int(content_prototype.shape[3]) / content_prototype_number)
    generated_img, decoder_full_feature_list, _ = \
        emdnet_decoder_framework(encoded_layer_list=valid_encoded_content_shortcut_list,
                                 decoder_input_org=mixed_fc,
                                 is_training=is_training,
                                 output_width=img_width,
                                 output_filters=img_filters,
                                 batch_size=batch_size,
                                 decoder_device=generator_device,
                                 scope=scope + '/decoder',
                                 reuse=reuse,
                                 weight_decay=weight_decay,
                                 initializer=initializer,
                                 weight_decay_rate=weight_decay_rate,
                                 adain_use=adain_use)

    return_str = ("Emd-Net-GeneratorEncoderDecoder %d Layers"
                  % (int(np.floor(math.log(int(content_prototype[0].shape[1])) / math.log(2)))))


    return generated_img, encoded_content_final, encoded_style_final, return_str, \
           batch_diff, -1, \
           content_full_feature_list, style_full_feature_list, decoder_full_feature_list

def ResEmd_EmdNet_Generator(content_prototype,
                            style_reference,
                            is_training,
                            batch_size,
                            generator_device,
                            scope,
                            initializer,
                            weight_decay, weight_decay_rate,
                            reuse=False,
                            adain_use=False,
                            residual_at_layer=-1,
                            residual_block_num=-1,
                            adain_preparation_model=None,
                            debug_mode=True,
                            other_info=None):

    residual_at_layer=-1
    residual_block_num=-1
    adain_preparation_model=None
    adain_use=False

    style_input_number = len(style_reference)
    content_prototype_number = int(content_prototype.shape[3])

    # style reference encoder part
    for ii in range(len(style_reference)):
        if ii == 0:
            style_reference_tensor = style_reference[ii]
        else:
            style_reference_tensor = tf.concat([style_reference_tensor, style_reference[ii]], axis=3)
    encoded_style_final, _, _, style_full_feature_list, _ = \
        encoder_resemd_framework(images=style_reference_tensor,
                                 is_training=is_training,
                                 encoder_device=generator_device,
                                 scope=scope + '/style_encoder',
                                 reuse=reuse,
                                 initializer=initializer,
                                 weight_decay=weight_decay,
                                 weight_decay_rate=weight_decay_rate,
                                 adain_use=adain_use)

    # content prototype encoder part
    encoded_content_final, _, _, content_full_feature_list, _ = \
        encoder_resemd_framework(images=content_prototype,
                                 is_training=is_training,
                                 encoder_device=generator_device,
                                 residual_at_layer=residual_at_layer,
                                 scope=scope + '/content_encoder',
                                 reuse=reuse,
                                 initializer=initializer,
                                 weight_decay=weight_decay,
                                 weight_decay_rate=weight_decay_rate,
                                 adain_use=adain_use)

    # res-emd network mixer
    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(generator_device):
            with tf.variable_scope(scope+'/mixer'):
                if reuse:
                    tf.get_variable_scope().reuse_variables()

                mixed_feature = adaptive_instance_norm(content=encoded_content_final,
                                                       style=tf.expand_dims(encoded_style_final, axis=0))

                style_batch_diff = 0
                content_batch_diff = 0
                for ii in range(len(style_full_feature_list)):
                    current_batch_diff = _calculate_batch_diff(style_full_feature_list[ii])
                    style_batch_diff+=current_batch_diff
                style_batch_diff = style_batch_diff / len(style_full_feature_list)
                for ii in range(len(content_full_feature_list)):
                    current_batch_diff = _calculate_batch_diff(content_full_feature_list[ii])
                    content_batch_diff += current_batch_diff
                content_batch_diff = content_batch_diff / len(content_full_feature_list)


    # decoder part
    img_width = int(content_prototype.shape[1])
    img_filters = int(int(content_prototype.shape[3]) / content_prototype_number)
    generated_img, decoder_full_feature_list, _ = \
        decoder_resemdnet_framework(encoded_layer_list=-1,
                                    decoder_input_org=mixed_feature,
                                    is_training=is_training,
                                    output_width=img_width,
                                    output_filters=img_filters,
                                    batch_size=batch_size,
                                    decoder_device=generator_device,
                                    scope=scope + '/decoder',
                                    reuse=reuse,
                                    weight_decay=weight_decay,
                                    initializer=initializer,
                                    weight_decay_rate=weight_decay_rate,
                                    adain_use=adain_use,
                                    other_info=other_info)

    if other_info == None:
        return_str = ("Res-Emd-Net-GeneratorEncoderDecoder %d Layers"
                      % (int(np.floor(math.log(int(content_prototype[0].shape[1])) / math.log(2)))))
    elif other_info== 'NN':
        return_str = ("NN-Res-Emd-Net-GeneratorEncoderDecoder %d Layers"
                      % (int(np.floor(math.log(int(content_prototype[0].shape[1])) / math.log(2)))))


    return generated_img, encoded_content_final, encoded_style_final, return_str, \
           style_batch_diff, content_batch_diff, \
           content_full_feature_list, style_full_feature_list, decoder_full_feature_list





def AdobeNet_Generator(content_prototype,
                       style_reference,
                       is_training,
                       batch_size,
                       generator_device,
                       residual_at_layer,
                       residual_block_num,
                       scope,
                       initializer,
                       weight_decay, weight_decay_rate,
                       reuse=False,
                       adain_use=False,
                       adain_preparation_model=None,
                       debug_mode=True,
                       other_info=None):
    residual_at_layer = -1
    residual_block_num = -1
    adain_preparation_model = None
    adain_use = False


    style_input_number = len(style_reference)
    content_prototype_number = int(content_prototype.shape[3])

    # style reference encoder part
    for ii in range(len(style_reference)):
        if ii == 0:
            style_reference_tensor = style_reference[ii]
        else:
            style_reference_tensor = tf.concat([style_reference_tensor, style_reference[ii]], axis=3)

    encoded_style_final, _, _, style_full_feature_list, _ = \
        encoder_adobenet_framework(images=style_reference_tensor,
                                   is_training=is_training,
                                   encoder_device=generator_device,
                                   scope=scope + '/style_encoder',
                                   reuse=reuse,
                                   initializer=initializer,
                                   weight_decay=weight_decay,
                                   weight_decay_rate=weight_decay_rate,
                                   adain_use=adain_use)

    # content prototype encoder part
    encoded_content_final, _, _, content_full_feature_list, _ = \
        encoder_adobenet_framework(images=content_prototype,
                                   is_training=is_training,
                                   encoder_device=generator_device,
                                   residual_at_layer=residual_at_layer,
                                   scope=scope + '/content_encoder',
                                   reuse=reuse,
                                   initializer=initializer,
                                   weight_decay=weight_decay,
                                   weight_decay_rate=weight_decay_rate,
                                   adain_use=adain_use)

    # mixer
    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(generator_device):
            with tf.variable_scope(scope+'/mixer'):
                if reuse:
                    tf.get_variable_scope().reuse_variables()

                mixed_feature = tf.concat([encoded_content_final,encoded_style_final], axis=3)
                style_batch_diff=0
                content_batch_diff=0
                for ii in range(len(content_full_feature_list)):
                    content_batch_diff+=_calculate_batch_diff(content_full_feature_list[ii])
                content_batch_diff=content_batch_diff/len(content_full_feature_list)
                for ii in range(len(style_full_feature_list)):
                    style_batch_diff+=_calculate_batch_diff(style_full_feature_list[ii])
                style_batch_diff=style_batch_diff/len(style_full_feature_list)

    # decoder
    img_width = int(content_prototype.shape[1])
    img_filters = int(int(content_prototype.shape[3]) / content_prototype_number)
    generated_img, decoder_full_feature_list, _ = \
        decoder_adobenet_framework(encoded_layer_list=-1,
                                   decoder_input_org=mixed_feature,
                                   is_training=is_training,
                                   output_width=img_width,
                                   output_filters=img_filters,
                                   batch_size=batch_size,
                                   decoder_device=generator_device,
                                   scope=scope + '/decoder',
                                   reuse=reuse,
                                   weight_decay=weight_decay,
                                   initializer=initializer,
                                   weight_decay_rate=weight_decay_rate,
                                   adain_use=adain_use,
                                   other_info=other_info)

    return_str = ("Adobe-Net-GeneratorEncoderDecoder")

    return generated_img, encoded_content_final, encoded_style_final, return_str, \
           style_batch_diff, content_batch_diff, \
           content_full_feature_list, style_full_feature_list, decoder_full_feature_list




def ResMixerNet_Generator(content_prototype,
                          style_reference,
                          is_training,
                          batch_size,
                          generator_device,
                          residual_at_layer,
                          residual_block_num,
                          scope,
                          initializer,
                          weight_decay, weight_decay_rate,
                          reuse=False,
                          adain_use=False,
                          adain_preparation_model=None,
                          debug_mode=True,
                          other_info=None):
    residual_at_layer = -1
    residual_block_num = -1
    adain_preparation_model = None
    adain_use = False

    style_input_number = len(style_reference)
    content_prototype_number = int(content_prototype.shape[3])

    # style reference encoder part
    for ii in range(len(style_reference)):
        if ii == 0:
            style_reference_tensor = style_reference[ii]
        else:
            style_reference_tensor = tf.concat([style_reference_tensor, style_reference[ii]], axis=3)

    encoded_style_final, _, _, style_full_feature_list, _ = \
        encoder_resmixernet_framework(images=style_reference_tensor,
                                      is_training=is_training,
                                      encoder_device=generator_device,
                                      scope=scope + '/style_encoder',
                                      reuse=reuse,
                                      initializer=initializer,
                                      weight_decay=weight_decay,
                                      weight_decay_rate=weight_decay_rate,
                                      adain_use=adain_use)

    # content prototype encoder part
    encoded_content_final, _, _, content_full_feature_list, _ = \
        encoder_resmixernet_framework(images=content_prototype,
                                      is_training=is_training,
                                      encoder_device=generator_device,
                                      residual_at_layer=residual_at_layer,
                                      scope=scope + '/content_encoder',
                                      reuse=reuse,
                                      initializer=initializer,
                                      weight_decay=weight_decay,
                                      weight_decay_rate=weight_decay_rate,
                                      adain_use=adain_use)

    # mixer
    mixed_feature = tf.concat([encoded_content_final, encoded_style_final], axis=3)
    style_batch_diff = 0
    content_batch_diff = 0
    for ii in range(len(content_full_feature_list)):
        content_batch_diff += _calculate_batch_diff(content_full_feature_list[ii])
    content_batch_diff = content_batch_diff / len(content_full_feature_list)
    for ii in range(len(style_full_feature_list)):
        style_batch_diff += _calculate_batch_diff(style_full_feature_list[ii])
    style_batch_diff = style_batch_diff / len(style_full_feature_list)

    mixed_feature = \
        resmixer(generator_device=generator_device,
                 reuse=reuse,
                 scope=scope+'/mixer',
                 initializer=initializer,
                 is_training=is_training,
                 mixer_form=other_info,
                 weight_decay=weight_decay, weight_decay_rate=weight_decay_rate,
                 mixed_feature=mixed_feature)

    generated_img, decoder_full_feature_list, \
        = decoder_resmixernet_framework(decoder_input_org=mixed_feature,
                                        batch_size=batch_size,
                                        decoder_device=generator_device,
                                        scope=scope + '/decoder',
                                        initializer=initializer,
                                        weight_decay=weight_decay,weight_decay_rate=weight_decay_rate,
                                        reuse=reuse)

    return_str = ("%s-Net-GeneratorEncoderDecoder" % other_info)

    return generated_img, encoded_content_final, encoded_style_final, return_str, \
           style_batch_diff, content_batch_diff, \
           content_full_feature_list, style_full_feature_list, decoder_full_feature_list
