
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
### Mixers ###################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
def wnet_feature_mixer_framework(generator_device,scope,is_training,reuse,initializer,debug_mode,
                                 weight_decay,weight_decay_rate,style_input_number,
                                 residual_block_num, residual_at_layer,
                                 encoded_style_final_list,
                                 style_short_cut_interface_list,style_residual_interface_list,
                                 content_short_cut_interface, content_residual_interface,
                                 full_style_feature_list,
                                 adain_use,adain_preparation_model):



    # multiple encoded information average calculation for style reference encoder
    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(generator_device):
            with tf.variable_scope(scope):
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                for ii in range(style_input_number):
                    if ii == 0:
                        encoded_style_final = tf.expand_dims(encoded_style_final_list[ii], axis=0)
                        style_short_cut_interface = list()
                        for jj in range(len(style_short_cut_interface_list[ii])):
                            style_short_cut_interface.append(
                                tf.expand_dims(style_short_cut_interface_list[ii][jj], axis=0))
                        style_residual_interface = list()
                        for jj in range(len(style_residual_interface_list[ii])):
                            style_residual_interface.append(
                                tf.expand_dims(style_residual_interface_list[ii][jj], axis=0))
                    else:
                        encoded_style_final = tf.concat(
                            [encoded_style_final, tf.expand_dims(encoded_style_final_list[ii], axis=0)], axis=0)
                        for jj in range(len(style_short_cut_interface_list[ii])):
                            style_short_cut_interface[jj] = tf.concat([style_short_cut_interface[jj],
                                                                       tf.expand_dims(
                                                                           style_short_cut_interface_list[ii][jj],
                                                                           axis=0)], axis=0)

                        for jj in range(len(style_residual_interface_list[ii])):
                            style_residual_interface[jj] = tf.concat([style_residual_interface[jj], tf.expand_dims(
                                style_residual_interface_list[ii][jj], axis=0)], axis=0)

                encoded_style_final_avg = tf.reduce_mean(encoded_style_final, axis=0)
                encoded_style_final_max = tf.reduce_max(encoded_style_final, axis=0)
                encoded_style_final_min = tf.reduce_min(encoded_style_final, axis=0)
                encoded_style_final = tf.concat([encoded_style_final_avg, encoded_style_final_max, encoded_style_final_min], axis=3)

                style_shortcut_batch_diff = 0
                for ii in range(len(style_short_cut_interface)):
                    style_short_cut_avg = tf.reduce_mean(style_short_cut_interface[ii], axis=0)
                    style_short_cut_max = tf.reduce_max(style_short_cut_interface[ii], axis=0)
                    style_short_cut_min = tf.reduce_min(style_short_cut_interface[ii], axis=0)
                    style_short_cut_interface[ii] = tf.concat(
                        [style_short_cut_avg, style_short_cut_max, style_short_cut_min], axis=3)
                    style_shortcut_batch_diff += _calculate_batch_diff(input_feature=style_short_cut_interface[ii])
                style_shortcut_batch_diff = style_shortcut_batch_diff / len(style_short_cut_interface)

                style_residual_batch_diff = 0
                for ii in range(len(style_residual_interface)):
                    style_residual_avg = tf.reduce_mean(style_residual_interface[ii], axis=0)
                    style_residual_max = tf.reduce_max(style_residual_interface[ii], axis=0)
                    style_residual_min = tf.reduce_min(style_residual_interface[ii], axis=0)
                    style_residual_interface[ii] = tf.concat(
                        [style_residual_avg, style_residual_max, style_residual_min], axis=3)
                    style_residual_batch_diff += _calculate_batch_diff(input_feature=style_residual_interface[ii])
                style_residual_batch_diff = style_residual_batch_diff / len(style_residual_interface)

    # full style feature reformat
    if adain_use:
        full_style_feature_list_reformat = list()
        for ii in range(len(full_style_feature_list)):
            for jj in range(len(full_style_feature_list[ii])):

                current_feature = tf.expand_dims(full_style_feature_list[ii][jj], axis=0)
                if ii == 0:
                    full_style_feature_list_reformat.append(current_feature)
                else:
                    full_style_feature_list_reformat[jj] = tf.concat(
                        [full_style_feature_list_reformat[jj], current_feature], axis=0)
    else:
        full_style_feature_list_reformat = None

    # residual interfaces && short cut interfaces are fused together
    fused_residual_interfaces = list()
    fused_shortcut_interfaces = list()
    for ii in range(len(content_residual_interface)):
        current_content_residual_size = int(content_residual_interface[ii].shape[1])
        output_current_residual = content_residual_interface[ii]
        if adain_use:  # for adaptive instance normalization
            for jj in range(len(full_style_feature_list_reformat)):
                if int(full_style_feature_list_reformat[jj].shape[2]) == int(output_current_residual.shape[1]):
                    break
            output_current_residual = adaptive_instance_norm(content=output_current_residual,
                                                             style=full_style_feature_list_reformat[jj])

        for jj in range(len(style_residual_interface)):
            current_style_residual_size = int(style_residual_interface[jj].shape[1])
            if current_style_residual_size == current_content_residual_size:
                output_current_residual = tf.concat([output_current_residual, style_residual_interface[jj]], axis=3)
        fused_residual_interfaces.append(output_current_residual)
    for ii in range(len(content_short_cut_interface)):
        current_content_shortcut_size = int(content_short_cut_interface[ii].shape[1])
        output_current_shortcut = content_short_cut_interface[ii]

        if adain_use:  # for adaptive instance normalization
            for jj in range(len(full_style_feature_list_reformat)):
                if int(full_style_feature_list_reformat[jj].shape[2]) == int(output_current_shortcut.shape[1]):
                    break
            output_current_shortcut = adaptive_instance_norm(content=output_current_shortcut,
                                                             style=full_style_feature_list_reformat[jj])

        for jj in range(len(style_short_cut_interface)):
            current_style_short_cut_size = int(style_short_cut_interface[jj].shape[1])
            if current_style_short_cut_size == current_content_shortcut_size:
                output_current_shortcut = tf.concat([output_current_shortcut, style_short_cut_interface[jj]],
                                                    axis=3)
        fused_shortcut_interfaces.append(output_current_shortcut)

    # fused resudual interfaces are put into the residual blocks
    if not residual_block_num == 0 or not residual_at_layer == -1:
        residual_output_list, _ = residual_block_implementation(input_list=fused_residual_interfaces,
                                                                residual_num=residual_block_num,
                                                                residual_at_layer=residual_at_layer,
                                                                is_training=is_training,
                                                                residual_device=generator_device,
                                                                scope=scope + '/resblock',
                                                                reuse=reuse,
                                                                initializer=initializer,
                                                                weight_decay=weight_decay,
                                                                weight_decay_rate=weight_decay_rate,
                                                                style_features=full_style_feature_list_reformat,
                                                                adain_use=adain_use,
                                                                adain_preparation_model=adain_preparation_model,
                                                                debug_mode=debug_mode)

    # combination of all the encoder outputs
    fused_shortcut_interfaces.reverse()
    encoded_layer_list = fused_shortcut_interfaces
    encoded_layer_list.extend(residual_output_list)

    return encoded_layer_list, style_shortcut_batch_diff, style_residual_batch_diff, \
           encoded_style_final

def emdnet_mixer_with_adain(generator_device,reuse,scope,initializer,
                            weight_decay,weight_decay_rate,
                            encoded_content_final,content_shortcut_interface,encoded_style_final):

    # mixer
    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(generator_device):
            with tf.variable_scope(scope):
                if reuse:
                    tf.get_variable_scope().reuse_variables()

                encoded_style_final_expand = tf.expand_dims(encoded_style_final,axis=0)

                mixed_feature = adaptive_instance_norm(content=encoded_content_final,
                                                       style=encoded_style_final_expand)

                valid_encoded_content_shortcut_list = list()
                batch_diff = 0
                batch_diff_count = 0
                for ii in range(len(content_shortcut_interface)):
                    if ii == 0 or ii == len(content_shortcut_interface) - 1:
                        valid_encoded_content_shortcut_list.append(content_shortcut_interface[ii])
                        batch_diff += _calculate_batch_diff(input_feature=content_shortcut_interface[ii])
                        batch_diff_count += 1
                    else:
                        valid_encoded_content_shortcut_list.append(None)
                valid_encoded_content_shortcut_list.reverse()
                batch_diff = batch_diff / batch_diff_count

    return valid_encoded_content_shortcut_list, mixed_feature, batch_diff

def emdnet_mixer_non_adain(generator_device,reuse,scope,initializer,
                           weight_decay,weight_decay_rate,
                           encoded_content_final,content_shortcut_interface,encoded_style_final):

    # mixer
    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(generator_device):

            with tf.variable_scope(scope):
                if reuse:
                    tf.get_variable_scope().reuse_variables()

                encoded_content_final_squeeze = tf.squeeze(encoded_content_final)
                encoded_style_final_squeeze = tf.squeeze(encoded_style_final)
                encoded_content_fc = lrelu(fc(x=encoded_content_final_squeeze,
                                              output_size=generator_dim,
                                              scope="emd_mixer/content_fc",
                                              parameter_update_device=generator_device,
                                              initializer=initializer,
                                              weight_decay=weight_decay,
                                              name_prefix=scope, weight_decay_rate=weight_decay_rate))
                encoded_style_fc = lrelu(fc(x=encoded_style_final_squeeze,
                                            output_size=generator_dim,
                                            scope="emd_mixer/style_fc",
                                            parameter_update_device=generator_device,
                                            initializer=initializer,
                                            weight_decay=weight_decay,
                                            name_prefix=scope, weight_decay_rate=weight_decay_rate))
                mix_content_style = emd_mixer(content=encoded_content_fc,
                                              style=encoded_style_fc,
                                              initializer=initializer,
                                              device=generator_device)
                mixed_fc = relu(fc(x=mix_content_style,
                                   output_size=int(encoded_content_final.shape[3]),
                                   scope="emd_mixer/mixed_fc",
                                   parameter_update_device=generator_device,
                                   initializer=initializer,
                                   weight_decay=weight_decay,
                                   name_prefix=scope, weight_decay_rate=weight_decay_rate))

                mixed_fc = tf.expand_dims(mixed_fc, axis=1)
                mixed_fc = tf.expand_dims(mixed_fc, axis=1)

                valid_encoded_content_shortcut_list = list()
                batch_diff = 0
                batch_diff_count = 0
                for ii in range(len(content_shortcut_interface)):
                    if ii == 0 or ii == len(content_shortcut_interface) - 1:
                        valid_encoded_content_shortcut_list.append(content_shortcut_interface[ii])
                        batch_diff += _calculate_batch_diff(input_feature=content_shortcut_interface[ii])
                        batch_diff_count += 1
                    else:
                        valid_encoded_content_shortcut_list.append(None)
                valid_encoded_content_shortcut_list.reverse()
                batch_diff = batch_diff / batch_diff_count

    return valid_encoded_content_shortcut_list,mixed_fc,batch_diff

def resmixer(generator_device,reuse,is_training, scope,initializer, mixer_form,
             weight_decay,weight_decay_rate,
             mixed_feature):
    mix_num = int(mixer_form[0:mixer_form.find('-')])
    mix_form = mixer_form[mixer_form.find('-') + 1:]


    # mixer
    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(generator_device):
            with tf.variable_scope(scope):
                if reuse:
                    tf.get_variable_scope().reuse_variables()



                mix_input = mixed_feature
                filter_num = int(mixed_feature.shape[3])
                mix_output_list = list()
                for mix_counter in range(mix_num):
                    if 'Simple' in mix_form:
                        mix_output = resblock(x=mix_input,
                                              initializer=initializer,
                                              batch_norm_used=True, is_training=is_training,
                                              layer=mix_counter + 1,
                                              sh=1, sw=1, kh=3, kw=3,
                                              weight_decay=weight_decay, weight_decay_rate=weight_decay_rate,
                                              scope=scope,
                                              parameter_update_devices=generator_device)
                    elif 'Dense' in mix_form:
                        mix_output = desblock(input_x=mix_input,
                                              output_filters=filter_num,
                                              initializer=initializer,
                                              batch_norm_used=True, is_training=is_training,
                                              layer=mix_counter + 1,
                                              sh=1, sw=1, kh=3, kw=3,
                                              weight_decay=weight_decay, weight_decay_rate=weight_decay_rate,
                                              scope=scope,
                                              parameter_update_devices=generator_device)
                    if 'Simple' in mix_form:
                        mix_input=mix_output
                    elif 'Dense' in mix_form:
                        mix_output_list.append(mix_output)
                        for ii in range(len(mix_output_list)):
                            if ii == 0:
                                mix_input=mix_output
                            else:
                                mix_input = tf.concat([mix_input,mix_output_list[ii]],axis=3)

    return mix_output




##############################################################################################
##############################################################################################
##############################################################################################
### ResBlocks ################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
def single_resblock(adain_use, is_training, residual_device, initializer,scope,
                    weight_decay, weight_decay_rate,
                    x,layer,style):

    filters = int(x.shape[3])
    if not adain_use:
        norm1 = batch_norm(x=x,
                           is_training=is_training,
                           scope="layer%d_bn1" % layer,
                           parameter_update_device=residual_device)
    else:
        norm1 = adaptive_instance_norm(content=x,
                                       style=style)

    act1 = relu(norm1)
    conv1 = conv2d(x=act1,
                   output_filters=filters,
                   scope="layer%d_conv1" % layer,
                   parameter_update_device=residual_device,
                   kh=3,kw=3,sh=1,sw=1,
                   initializer=initializer,
                   weight_decay=weight_decay,
                   name_prefix=scope,
                   weight_decay_rate=weight_decay_rate)
    if not adain_use:
        norm2 = batch_norm(x=conv1,
                           is_training=is_training,
                           scope="layer%d_bn2" % layer,
                           parameter_update_device=residual_device)
    else:
        norm2 = adaptive_instance_norm(content=conv1,
                                       style=style)
    act2 = relu(norm2)
    conv2 = conv2d(x=act2,
                   output_filters=filters,
                   scope="layer%d_conv2" % layer,
                   parameter_update_device=residual_device,
                   initializer=initializer,
                   weight_decay=weight_decay,name_prefix=scope,
                   weight_decay_rate=weight_decay_rate,
                   kh=3,kw=3,sh=1,sw=1)

    output = x + conv2

    return output

def residual_block_implementation(input_list,
                                  residual_num,
                                  residual_at_layer,
                                  is_training,
                                  residual_device,
                                  reuse,scope,
                                  initializer,
                                  weight_decay,
                                  weight_decay_rate,
                                  style_features,
                                  adain_use=False,
                                  adain_preparation_model=None,
                                  debug_mode=False):


    return_str = "Residual %d Blocks" % residual_num
    input_list.reverse()
    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(residual_device):
            residual_output_list = list()

            if not reuse:
                print (print_separater)
                print ('Adaptive Instance Normalization for Residual Preparations: %s' % adain_preparation_model)
                print (print_separater)

            for ii in range(len(input_list)):
                current_residual_num = residual_num + 2 * ii
                current_residual_input = input_list[ii]
                current_scope = scope + '_onEncDecLyr%d' % (residual_at_layer - ii)


                if adain_use:
                    with tf.variable_scope(current_scope):
                        for jj in range(len(style_features)):
                            if int(style_features[jj].shape[2]) == int(current_residual_input.shape[1]):
                                break

                        for jj in range(int(style_features[ii].shape[0])):
                            if reuse or jj > 0:
                                tf.get_variable_scope().reuse_variables()

                            batch_size = int(style_features[ii][jj, :, :, :, :].shape[0])
                            if batch_size == 1:
                                current_init_residual_input = style_features[ii][jj, :, :, :, :]
                            else:
                                current_init_residual_input = tf.squeeze(style_features[ii][jj, :, :, :, :])

                            if adain_preparation_model == 'Multi':
                                # multiple cnn layer built to make the style_conv be incorporated with the dimension of the residual blocks
                                log_input = math.log(int(current_init_residual_input.shape[3])) / math.log(2)
                                if math.log(int(current_init_residual_input.shape[3])) < math.log(int(current_residual_input.shape[3])):
                                    if np.floor(log_input) < math.log(int(current_residual_input.shape[3])) / math.log(2):
                                        filter_num_start = int(np.floor(log_input)) + 1
                                    else:
                                        filter_num_start = int(np.floor(log_input))
                                    filter_num_start = int(math.pow(2,filter_num_start))
                                elif math.log(int(current_init_residual_input.shape[3])) > math.log(int(current_residual_input.shape[3])):
                                    if np.ceil(log_input) > math.log(int(current_residual_input.shape[3])) / math.log(2):
                                        filter_num_start = int(np.ceil(log_input)) - 1
                                    else:
                                        filter_num_start = int(np.ceil(log_input))
                                    filter_num_start = int(math.pow(2, filter_num_start))
                                else:
                                    filter_num_start = int(current_residual_input.shape[3])
                                filter_num_end = int(current_residual_input.shape[3])

                                if int(current_init_residual_input.shape[3]) == filter_num_end:
                                    continue_build = False
                                    style_conv = current_init_residual_input
                                else:
                                    continue_build = True


                                current_style_conv_input = current_init_residual_input
                                current_output_filter_num = filter_num_start
                                style_cnn_layer_num = 0
                                while continue_build:
                                    style_conv = conv2d(x=current_style_conv_input,
                                                        output_filters=current_output_filter_num,
                                                        scope="conv0_style_layer%d" % (style_cnn_layer_num+1),
                                                        parameter_update_device=residual_device,
                                                        kh=3, kw=3, sh=1, sw=1,
                                                        initializer=initializer,
                                                        weight_decay=weight_decay,
                                                        name_prefix=scope,
                                                        weight_decay_rate=weight_decay_rate)
                                    if not (reuse or jj > 0):
                                        print (style_conv)
                                    style_conv = relu(style_conv)


                                    current_style_conv_input = style_conv

                                    if filter_num_start < filter_num_end:
                                        current_output_filter_num = current_output_filter_num * 2
                                    else:
                                        current_output_filter_num = current_output_filter_num / 2
                                    style_cnn_layer_num += 1

                                    if current_output_filter_num > filter_num_end and \
                                            math.log(int(current_init_residual_input.shape[3])) < math.log(int(current_residual_input.shape[3])):
                                        current_output_filter_num = filter_num_end
                                    if current_output_filter_num < filter_num_end and \
                                            math.log(int(current_init_residual_input.shape[3])) > math.log(int(current_residual_input.shape[3])):
                                        current_output_filter_num = filter_num_end

                                    if int(style_conv.shape[3]) == filter_num_end:
                                        continue_build = False



                            elif adain_preparation_model == 'Single':
                                if int(current_init_residual_input.shape[3]) == int(current_residual_input.shape[3]):
                                    style_conv = current_init_residual_input
                                else:
                                    style_conv = conv2d(x=current_init_residual_input,
                                                        output_filters=int(current_residual_input.shape[3]),
                                                        scope="conv0_style_layer0",
                                                        parameter_update_device=residual_device,
                                                        kh=3, kw=3, sh=1, sw=1,
                                                        initializer=initializer,
                                                        weight_decay=weight_decay,
                                                        name_prefix=scope,
                                                        weight_decay_rate=weight_decay_rate)
                                    if not (reuse or jj > 0):
                                        print (style_conv)
                                    style_conv = relu(style_conv)



                            if jj == 0:
                                style_features_new = tf.expand_dims(style_conv, axis=0)
                            else:
                                style_features_new = tf.concat([style_features_new,
                                                                tf.expand_dims(style_conv, axis=0)],
                                                               axis=0)

                    if (not reuse) and (not math.log(int(current_init_residual_input.shape[3])) == math.log(int(current_residual_input.shape[3]))):
                        print (print_separater)


                else:
                    style_features_new=None


                with tf.variable_scope(current_scope):
                    if reuse:
                        tf.get_variable_scope().reuse_variables()

                    for jj in range(current_residual_num):
                        if jj == 0:
                            residual_input = current_residual_input
                        else:
                            residual_input = residual_block_output
                        residual_block_output = \
                            single_resblock(adain_use=adain_use,
                                            is_training=is_training,
                                            residual_device=residual_device,
                                            initializer=initializer,
                                            scope=scope,
                                            weight_decay=weight_decay,
                                            weight_decay_rate=weight_decay_rate,
                                            x=residual_input,
                                            layer=jj+1,
                                            style=style_features_new)
                        if jj == current_residual_num-1:
                            residual_output = residual_block_output

                    residual_output_list.append(residual_output)


    if (not reuse) and adain_use and (not debug_mode):
        print(print_separater)
        raw_input("Press enter to continue")
    print(print_separater)

    return residual_output_list, return_str

