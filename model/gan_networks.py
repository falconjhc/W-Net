
import tensorflow as tf
import sys
import math
sys.path.append('..')


import numpy as np
from utilities.ops import lrelu, relu, batch_norm, layer_norm, instance_norm, adaptive_instance_norm


from utilities.ops import conv2d, deconv2d, fc


import math

print_separater="#########################################################"

eps = 1e-9

def minibatch_discrimination(parameter_update_device,
                             input_pattern,
                             weight_decay,weight_decay_rate,initializer,batch_size,scope, is_training, build_type):

    def res_block(x, scope):
        filters = int(x.shape[3])
        bn = batch_norm(x=x,
                        is_training=is_training,
                        scope=scope+'_Res_Bn',
                        parameter_update_device=parameter_update_device)
        act = relu(bn)
        conv = conv2d(x=act,
                      output_filters=filters,
                      scope=scope+'_Res_Conv',
                      parameter_update_device=parameter_update_device,
                      kh=3,kw=3,sh=1,sw=1,
                      initializer=initializer,
                      weight_decay=weight_decay,
                      name_prefix=scope,
                      weight_decay_rate=weight_decay_rate)

        output = x + conv

        return output

    activation = input_pattern

    diffs = tf.abs(tf.expand_dims(activation,4)- tf.expand_dims(tf.transpose(activation,[1,2,3,0]), 0))
    return -1, tf.reduce_mean(diffs)


def encoder_framework(images,
                      is_training,
                      encoder_device,
                      residual_at_layer,
                      residual_connection_mode,
                      scope,initializer,weight_decay,
                      weight_decay_rate,
                      final_layer_logit_length,
                      reuse = False,adain_use=False):
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
    generator_dim = 64



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

                if not final_layer_logit_length == -1:
                    output_final_encoded_reshaped = tf.reshape(output_final_encoded, [batch_size, -1])
                    final_category_layer = fc(x=output_final_encoded_reshaped,
                                              output_size=final_layer_logit_length,
                                              scope='final_category_layer',
                                              parameter_update_device=encoder_device,
                                              initializer=initializer,
                                              weight_decay=weight_decay,
                                              name_prefix=scope,
                                              weight_decay_rate=weight_decay_rate)
                else:
                    final_category_layer = -1

        return output_final_encoded, final_category_layer, \
               shortcut_list, residual_input_list, full_feature_list, \
               return_str



def residual_block(input_list,
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
    def res_block(x,layer,style):

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
                        residual_block_output = res_block(x=residual_input,
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

def residual_block_concatenation(input_res_list1,input_res_list2):
    output_residual_block_list = input_res_list1

    for ii in input_res_list2:
        counter = 0
        for jj in input_res_list1:
            if int(ii.shape[1]) == int(jj.shape[1]):
                output_residual_block_list[counter] = \
                    tf.concat([jj, ii], axis=3)
            counter += 1

    return output_residual_block_list

def decoder_framework(encoded_layer_list,
                      is_training,
                      output_width,
                      output_filters,
                      batch_size,
                      decoder_device,
                      scope,initializer, weight_decay,weight_decay_rate,
                      adain_use,
                      reuse=False):
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

    decoder_input = encoded_layer_list[0]
    return_str = "Decoder %d Layers" % int(np.floor(math.log(output_width) / math.log(2)))
    generator_dim=64
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




def generator_inferring(content,style,
                        generator_device,
                        style_reference_num,
                        residual_at_layer,
                        residual_block_num,
                        img_filters, adain_use):
    weight_decay_rate = 0
    weight_decay = False
    initializer = 'XavierInit'
    scope = 'generator'
    label0_length = -1
    is_training = False

    # style encoder part
    style_short_cut_interface_list = list()
    style_residual_interface_list = list()
    style_full_feature_list = list()
    for ii in range(style_reference_num):
        if ii == 0:
            reuse = False
        else:
            reuse = True
        _, _, \
        current_style_short_cut_interface, current_style_residual_interface, current_style_full_feature_list, \
        _ = encoder_framework(images=tf.expand_dims(style[:, :, :, ii], axis=3),
                              is_training=False,
                              encoder_device=generator_device,
                              residual_at_layer=residual_at_layer,
                              residual_connection_mode='Single',
                              scope=scope+'/style_encoder',
                              reuse=reuse,
                              initializer='XavierInit',
                              weight_decay=False,
                              weight_decay_rate=0,
                              final_layer_logit_length=-1,
                              adain_use=adain_use)
        style_short_cut_interface_list.append(current_style_short_cut_interface)
        style_residual_interface_list.append(current_style_residual_interface)
        style_full_feature_list.append(current_style_full_feature_list)

    for ii in range(style_reference_num):
        if ii == 0:
            style_short_cut_interface = list()
            for jj in range(len(style_short_cut_interface_list[ii])):
                style_short_cut_interface.append(tf.expand_dims(style_short_cut_interface_list[ii][jj],
                                                                axis=0))
            style_residual_interface = list()
            for jj in range(len(style_residual_interface_list[ii])):
                style_residual_interface.append(tf.expand_dims(style_residual_interface_list[ii][jj],
                                                               axis=0))

        else:
            for jj in range(len(style_short_cut_interface_list[ii])):
                style_short_cut_interface[jj] = tf.concat([style_short_cut_interface[jj],
                                                           tf.expand_dims(style_short_cut_interface_list[ii][jj],
                                                                          axis=0)],
                                                          axis=0)

            for jj in range(len(style_residual_interface_list[ii])):
                style_residual_interface[jj] = tf.concat([style_residual_interface[jj],
                                                          tf.expand_dims(style_residual_interface_list[ii][jj],
                                                                         axis=0)],
                                                         axis=0)

    for ii in range(len(style_short_cut_interface)):
        style_short_cut_avg = tf.reduce_mean(style_short_cut_interface[ii], axis=0)
        style_short_cut_max = tf.reduce_max(style_short_cut_interface[ii], axis=0)
        style_short_cut_min = tf.reduce_min(style_short_cut_interface[ii], axis=0)
        style_short_cut_interface[ii] = tf.concat([style_short_cut_avg, style_short_cut_max, style_short_cut_min],
                                                  axis=3)
        # style_short_cut_interface[ii] = tf.reduce_mean(style_short_cut_interface[ii], axis=0)

    for ii in range(len(style_residual_interface)):
        style_residual_avg = tf.reduce_mean(style_residual_interface[ii], axis=0)
        style_residual_max = tf.reduce_max(style_residual_interface[ii], axis=0)
        style_residual_min = tf.reduce_min(style_residual_interface[ii], axis=0)
        style_residual_interface[ii] = tf.concat([style_residual_avg, style_residual_max, style_residual_min], axis=3)
        # style_residual_interface[ii] = tf.reduce_mean(style_residual_interface[ii], axis=0)

    # source encoder part
    encoded_content_final, content_category, content_short_cut_interface, content_residual_interface, content_full_faeture_list, _ = \
        encoder_framework(images=content,
                          is_training=is_training,
                          encoder_device=generator_device,
                          residual_at_layer=residual_at_layer,
                          residual_connection_mode='Multi',
                          scope=scope + '/content_encoder',
                          reuse=False,
                          initializer=initializer,
                          weight_decay=weight_decay,
                          weight_decay_rate=weight_decay_rate,
                          final_layer_logit_length=label0_length,
                          adain_use=adain_use)

    # full style feature reformat
    if adain_use:
        full_style_feature_list_reformat = list()
        for ii in range(len(style_full_feature_list)):
            for jj in range(len(style_full_feature_list[ii])):

                current_feature = tf.expand_dims(style_full_feature_list[ii][jj], axis=0)
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
                output_current_shortcut = tf.concat([output_current_shortcut, style_short_cut_interface[jj]], axis=3)
        fused_shortcut_interfaces.append(output_current_shortcut)


    # fused resudual interfaces are put into the residual blocks
    if not residual_block_num == 0 or not residual_at_layer == -1:
        residual_output_list, _ = residual_block(input_list=fused_residual_interfaces,
                                                 residual_num=residual_block_num,
                                                 residual_at_layer=residual_at_layer,
                                                 is_training=is_training,
                                                 residual_device=generator_device,
                                                 scope=scope + '/resblock',
                                                 reuse=False,
                                                 initializer=initializer,
                                                 weight_decay=weight_decay,
                                                 weight_decay_rate=weight_decay_rate,
                                                 style_features=full_style_feature_list_reformat,
                                                 adain_use=adain_use)

    # combination of all the encoder outputs
    fused_shortcut_interfaces.reverse()
    encoded_layer_list = fused_shortcut_interfaces
    encoded_layer_list.extend(residual_output_list)

    return_str = ("GeneratorEncoderDecoder %d Layers with %d ResidualBlocks connecting %d-th layer"
                  % (int(np.floor(math.log(int(content.shape[1])) / math.log(2))),
                     residual_block_num,
                     residual_at_layer))

    # decoder part
    img_width = int(content.shape[1])
    generated_img, decoder_full_feature_list, _ = \
        decoder_framework(encoded_layer_list=encoded_layer_list,
                          is_training=is_training,
                          output_width=img_width,
                          output_filters=img_filters,
                          batch_size=1,
                          decoder_device=generator_device,
                          scope=scope + '/decoder',
                          reuse=False,
                          weight_decay=weight_decay,
                          initializer=initializer,
                          weight_decay_rate=weight_decay_rate,
                          adain_use=adain_use)

    return generated_img, content_full_faeture_list, decoder_full_feature_list, style_full_feature_list




def generator_framework(content_prototype,style_reference,
                        is_training,
                        batch_size,
                        generator_device,
                        residual_at_layer,
                        residual_block_num,
                        label0_length,label1_length,
                        scope,
                        initializer,weight_decay,
                        weight_decay_rate,
                        style_input_number,
                        content_prototype_number,
                        reuse=False,
                        adain_use=False,
                        adain_preparation_model=None,
                        debug_mode=True):

    def _calculate_batch_diff(input_feature):
        diff = tf.abs(tf.expand_dims(input_feature, 4) -
                      tf.expand_dims(tf.transpose(input_feature, [1, 2, 3, 0]), 0))
        diff = tf.reduce_sum(tf.exp(-diff), 4)
        return tf.reduce_mean(diff)


    # content prototype encoder part
    encoded_content_final, content_category, content_short_cut_interface, content_residual_interface, content_full_feature_list, _ = \
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
                          final_layer_logit_length=label0_length,
                          adain_use=adain_use)

    # style reference encoder part
    encoded_style_final_list = list()
    style_category_list = list()
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

        encoded_style_final, style_category, current_style_short_cut_interface, current_style_residual_interface, current_full_feature_list, _ = \
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
                              final_layer_logit_length=label1_length,
                              adain_use=adain_use)
        encoded_style_final_list.append(encoded_style_final)
        style_category_list.append(style_category)
        style_short_cut_interface_list.append(current_style_short_cut_interface)
        style_residual_interface_list.append(current_style_residual_interface)
        full_style_feature_list.append(current_full_feature_list)

    # multiple encoded information average calculation for style reference encoder
    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device(generator_device):
            with tf.variable_scope(scope):
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                for ii in range(style_input_number):
                    if ii==0:
                        encoded_style_final = tf.expand_dims(encoded_style_final_list[ii],axis=0)
                        style_category = tf.expand_dims(style_category_list[ii],axis=0)
                        style_short_cut_interface=list()
                        for jj in range(len(style_short_cut_interface_list[ii])):
                            style_short_cut_interface.append(tf.expand_dims(style_short_cut_interface_list[ii][jj],axis=0))
                        style_residual_interface=list()
                        for jj in range(len(style_residual_interface_list[ii])):
                            style_residual_interface.append(tf.expand_dims(style_residual_interface_list[ii][jj],axis=0))
                    else:
                        encoded_style_final = tf.concat([encoded_style_final, tf.expand_dims(encoded_style_final_list[ii],axis=0)], axis=0)
                        style_category = tf.concat([style_category, tf.expand_dims(style_category_list[ii],axis=0)], axis=0)
                        for jj in range(len(style_short_cut_interface_list[ii])):
                            style_short_cut_interface[jj] = tf.concat([style_short_cut_interface[jj], tf.expand_dims(style_short_cut_interface_list[ii][jj],axis=0)], axis=0)

                        for jj in range(len(style_residual_interface_list[ii])):
                            style_residual_interface[jj] = tf.concat([style_residual_interface[jj], tf.expand_dims(style_residual_interface_list[ii][jj], axis=0)], axis=0)

                style_category = tf.reduce_mean(style_category,axis=0)
                encoded_style_final_avg = tf.reduce_mean(encoded_style_final,axis=0)
                encoded_style_final_max = tf.reduce_max(encoded_style_final,axis=0)
                encoded_style_final_min = tf.reduce_min(encoded_style_final,axis=0)
                encoded_style_final = tf.concat([encoded_style_final_avg, encoded_style_final_max, encoded_style_final_min], axis=3)

                style_shortcut_batch_diff=0
                for ii in range(len(style_short_cut_interface)):
                    style_short_cut_avg = tf.reduce_mean(style_short_cut_interface[ii], axis=0)
                    style_short_cut_max = tf.reduce_max(style_short_cut_interface[ii], axis=0)
                    style_short_cut_min = tf.reduce_min(style_short_cut_interface[ii], axis=0)
                    style_short_cut_interface[ii]= tf.concat([style_short_cut_avg,style_short_cut_max,style_short_cut_min],axis=3)
                    style_shortcut_batch_diff += _calculate_batch_diff(input_feature=style_short_cut_interface[ii])

                style_shortcut_batch_diff = style_shortcut_batch_diff / len(style_short_cut_interface)

                style_residual_batch_diff=0
                for ii in range(len(style_residual_interface)):
                    style_residual_avg = tf.reduce_mean(style_residual_interface[ii], axis=0)
                    style_residual_max = tf.reduce_max(style_residual_interface[ii], axis=0)
                    style_residual_min = tf.reduce_min(style_residual_interface[ii], axis=0)
                    style_residual_interface[ii] = tf.concat([style_residual_avg, style_residual_max, style_residual_min], axis=3)
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
        full_style_feature_list_reformat=None


    # residual interfaces && short cut interfaces are fused together
    fused_residual_interfaces = list()
    fused_shortcut_interfaces = list()
    for ii in range(len(content_residual_interface)):
        current_content_residual_size = int(content_residual_interface[ii].shape[1])
        output_current_residual = content_residual_interface[ii]
        if adain_use: # for adaptive instance normalization
            for jj in range(len(full_style_feature_list_reformat)):
                if int(full_style_feature_list_reformat[jj].shape[2]) == int(output_current_residual.shape[1]):
                    break
            output_current_residual = adaptive_instance_norm(content=output_current_residual,
                                                             style=full_style_feature_list_reformat[jj])


        for jj in range(len(style_residual_interface)):
            current_style_residual_size = int(style_residual_interface[jj].shape[1])
            if current_style_residual_size == current_content_residual_size:
                output_current_residual=tf.concat([output_current_residual,style_residual_interface[jj]],axis=3)
        fused_residual_interfaces.append(output_current_residual)
    for ii in range(len(content_short_cut_interface)):
        current_content_shortcut_size = int(content_short_cut_interface[ii].shape[1])
        output_current_shortcut = content_short_cut_interface[ii]

        if adain_use: # for adaptive instance normalization
            for jj in range(len(full_style_feature_list_reformat)):
                if int(full_style_feature_list_reformat[jj].shape[2]) == int(output_current_shortcut.shape[1]):
                    break
            output_current_shortcut = adaptive_instance_norm(content=output_current_shortcut,
                                                             style=full_style_feature_list_reformat[jj])

        for jj in range(len(style_short_cut_interface)):
            current_style_short_cut_size = int(style_short_cut_interface[jj].shape[1])
            if current_style_short_cut_size == current_content_shortcut_size:
                output_current_shortcut = tf.concat([output_current_shortcut, style_short_cut_interface[jj]], axis=3)
        fused_shortcut_interfaces.append(output_current_shortcut)

    # fused resudual interfaces are put into the residual blocks
    if not residual_block_num == 0 or not residual_at_layer == -1:
        residual_output_list, _ = residual_block(input_list=fused_residual_interfaces,
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


    return_str = ("GeneratorEncoderDecoder %d Layers with %d ResidualBlocks connecting %d-th layer"
                  % (int(np.floor(math.log(int(content_prototype[0].shape[1])) / math.log(2))),
                     residual_block_num,
                     residual_at_layer))

    # decoder part
    img_width = int(content_prototype.shape[1])
    img_filters = int(int(content_prototype.shape[3]) / content_prototype_number)
    generated_img,_, _ = \
        decoder_framework(encoded_layer_list=encoded_layer_list,
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

    return generated_img, encoded_content_final, encoded_style_final, \
           content_category, style_category, return_str, \
           style_shortcut_batch_diff, style_residual_batch_diff



def discriminator_mdy_5_convs(image,
                              is_training,
                              parameter_update_device,
                              category_logit_num,
                              batch_size,
                              critic_length,
                              initializer,weight_decay,scope,weight_decay_rate,
                              reuse=False):
    return_str = ("Discriminator-5Convs")
    return_str = "WST-" + return_str + "-Crc:%d" % critic_length
    discriminator_dim=64



    with tf.variable_scope("discriminator"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        h0 = lrelu(conv2d(x=image, output_filters=discriminator_dim,
                          scope="dis_h0_conv",
                          parameter_update_device=parameter_update_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,
                          weight_decay_rate=weight_decay_rate))
        h1 = lrelu(layer_norm(conv2d(x=h0,
                                     output_filters=discriminator_dim * 2,
                                     scope="dis_h1_conv",
                                     parameter_update_device=parameter_update_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,
                          weight_decay_rate=weight_decay_rate),
                              scope='dis_ln1',
                              parameter_update_device=parameter_update_device))
        h2 = lrelu(layer_norm(conv2d(x=h1,
                                     output_filters=discriminator_dim * 4,
                                     scope="dis_h2_conv",
                                     parameter_update_device=parameter_update_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,
                          weight_decay_rate=weight_decay_rate),
                              scope='dis_ln2',
                              parameter_update_device=parameter_update_device))
        h3 = lrelu(layer_norm(conv2d(x=h2,
                                     output_filters=discriminator_dim * 8,
                                     scope="dis_h3_conv",
                                     parameter_update_device=parameter_update_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,
                          weight_decay_rate=weight_decay_rate),
                              scope='dis_ln3',
                              parameter_update_device=parameter_update_device))

        h4 = lrelu(layer_norm(conv2d(x=h3,
                                     output_filters=discriminator_dim * 16,
                                     scope="dis_h4_conv",
                                     parameter_update_device=parameter_update_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,
                          weight_decay_rate=weight_decay_rate),
                              scope='dis_ln4',
                              parameter_update_device=parameter_update_device))


        h4_reshaped = tf.reshape(h4, [batch_size, -1])
        fc_input = h4_reshaped



        fc2 = fc(x=fc_input,
                 output_size=category_logit_num,
                 scope="dis_final_fc_category",
                 parameter_update_device=parameter_update_device,
                          weight_decay_rate=weight_decay_rate)

        fc1 = fc(x=fc_input,
                 output_size=critic_length,
                 scope="dis_final_fc_critic",
                 parameter_update_device=parameter_update_device,
                 initializer=initializer,
                 weight_decay=weight_decay,
                 name_prefix=scope,
                 weight_decay_rate=weight_decay_rate)

        return fc2, fc1, return_str


def discriminator_mdy_6_convs(image,
                              is_training,
                              parameter_update_device,
                              category_logit_num,
                              batch_size,
                              critic_length,
                              initializer,weight_decay,scope,weight_decay_rate,
                              reuse=False):
    return_str = ("Discriminator-6Convs")
    return_str = "WST-" + return_str + "-Crc:%d" % critic_length
    discriminator_dim=32

    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        h0 = lrelu(conv2d(x=image, output_filters=discriminator_dim,
                          scope="dis_h0_conv",
                          parameter_update_device=parameter_update_device,
                          initializer=initializer,
                          weight_decay=weight_decay,
                          name_prefix=scope,
                          weight_decay_rate=weight_decay_rate))
        h1 = lrelu(layer_norm(conv2d(x=h0,
                                     output_filters=discriminator_dim * 2,
                                     scope="dis_h1_conv",
                                     parameter_update_device=parameter_update_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,weight_decay_rate=weight_decay_rate),
                              scope="dis_ln1",
                              parameter_update_device=parameter_update_device))
        h2 = lrelu(layer_norm(conv2d(x=h1,
                                     output_filters=discriminator_dim * 4,
                                     scope="dis_h2_conv",
                                     parameter_update_device=parameter_update_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,weight_decay_rate=weight_decay_rate),
                              scope="dis_ln2",
                              parameter_update_device=parameter_update_device))
        h3 = lrelu(layer_norm(conv2d(x=h2,
                                     output_filters=discriminator_dim * 8,
                                     scope="dis_h3_conv",
                                     parameter_update_device=parameter_update_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,weight_decay_rate=weight_decay_rate),
                              scope="dis_ln3",
                              parameter_update_device=parameter_update_device))

        h4 = lrelu(layer_norm(conv2d(x=h3,
                                     output_filters=discriminator_dim * 16,
                                     scope="dis_h4_conv",
                                     parameter_update_device=parameter_update_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,weight_decay_rate=weight_decay_rate),
                              scope="dis_ln4",
                              parameter_update_device=parameter_update_device))

        h5 = lrelu(layer_norm(conv2d(x=h4,
                                     output_filters=discriminator_dim * 32,
                                     scope="dis_h5_conv",
                                     parameter_update_device=parameter_update_device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,weight_decay_rate=weight_decay_rate),
                              scope="dis_ln5",
                              parameter_update_device=parameter_update_device))


        h5_reshaped = tf.reshape(h5, [batch_size, -1])
        fc_input = h5_reshaped



        # category loss
        fc2 = fc(x=fc_input,
                 output_size=category_logit_num,
                 scope="dis_final_fc_category",
                 parameter_update_device=parameter_update_device,
                 initializer=initializer,
                 weight_decay=weight_decay,
                 name_prefix=scope,weight_decay_rate=weight_decay_rate)

        fc1 = fc(x=fc_input,
                 output_size=critic_length,
                 scope="dis_final_fc_critic",
                 parameter_update_device=parameter_update_device,
                 initializer=initializer,
                 weight_decay=weight_decay,
                 name_prefix=scope, weight_decay_rate=weight_decay_rate)

        return fc2, fc1, return_str


def discriminator_mdy_6_convs_tower_version1(image,
                                             is_training,
                                             parameter_update_device,
                                             category_logit_num,
                                             batch_size,
                                             critic_length,
                                             initializer,weight_decay,scope,weight_decay_rate,
                                             reuse=False):

    tower_num = int(image.shape[3])
    return_str = ("Discriminator-6Convs-Tower-Version1")
    return_str = "WST-" + return_str + "-Crc:%d" % critic_length
    discriminator_dim = 32

    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        for ii in range(tower_num):
            curt_input = image[:,:,:,ii]
            curt_input = tf.reshape(curt_input,
                                    [int(curt_input.shape[0]),
                                     int(curt_input.shape[1]),
                                     int(curt_input.shape[2]),
                                     1])


            h0 = lrelu(conv2d(x=curt_input,
                              output_filters=discriminator_dim,
                              scope="dis_twr%d_h0_conv" % ii,
                              parameter_update_device=parameter_update_device,
                              initializer=initializer,
                              weight_decay=weight_decay,
                              name_prefix=scope,
                              weight_decay_rate=weight_decay_rate))
            h1 = lrelu(layer_norm(conv2d(x=h0,
                                         output_filters=discriminator_dim * 2,
                                         scope="dis_twr%d_h1_conv" % ii,
                                         parameter_update_device=parameter_update_device,
                                         initializer=initializer,
                                         weight_decay=weight_decay,
                                         name_prefix=scope, weight_decay_rate=weight_decay_rate),
                                  scope="dis_twr%d_ln1" % ii,
                                  parameter_update_device=parameter_update_device))
            h2 = lrelu(layer_norm(conv2d(x=h1,
                                         output_filters=discriminator_dim * 4,
                                         scope="dis_twr%d_h2_conv" % ii,
                                         parameter_update_device=parameter_update_device,
                                         initializer=initializer,
                                         weight_decay=weight_decay,
                                         name_prefix=scope, weight_decay_rate=weight_decay_rate),
                                  scope="dis_twr%d_ln2" % ii,
                                  parameter_update_device=parameter_update_device))
            h3 = lrelu(layer_norm(conv2d(x=h2,
                                         output_filters=discriminator_dim * 8,
                                         scope="dis_twr%d_h3_conv" % ii,
                                         parameter_update_device=parameter_update_device,
                                         initializer=initializer,
                                         weight_decay=weight_decay,
                                         name_prefix=scope, weight_decay_rate=weight_decay_rate),
                                  scope="dis_twr%d_ln3" % ii,
                                  parameter_update_device=parameter_update_device))

            h4 = lrelu(layer_norm(conv2d(x=h3,
                                         output_filters=discriminator_dim * 16,
                                         scope="dis_twr%d_h4_conv" % ii,
                                         parameter_update_device=parameter_update_device,
                                         initializer=initializer,
                                         weight_decay=weight_decay,
                                         name_prefix=scope, weight_decay_rate=weight_decay_rate),
                                  scope="dis_twr%d_ln4" % ii,
                                  parameter_update_device=parameter_update_device))

            h5 = lrelu(layer_norm(conv2d(x=h4,
                                         output_filters=discriminator_dim * 32,
                                         scope="dis_twr%d_h5_conv" % ii,
                                         parameter_update_device=parameter_update_device,
                                         initializer=initializer,
                                         weight_decay=weight_decay,
                                         name_prefix=scope, weight_decay_rate=weight_decay_rate),
                                  scope="dis_twr%d_ln5" % ii,
                                  parameter_update_device=parameter_update_device))

            h5_reshaped = tf.reshape(h5, [batch_size, -1])

            if ii == 0:
                h5_reshaped_total = h5_reshaped
            else:
                h5_reshaped_total = tf.concat([h5_reshaped_total, h5_reshaped],axis=1)


        # category loss
        fc2 = fc(x=h5_reshaped_total,
                 output_size=category_logit_num,
                 scope="dis_final_fc_category",
                 parameter_update_device=parameter_update_device,
                 initializer=initializer,
                 weight_decay=weight_decay,
                 name_prefix=scope, weight_decay_rate=weight_decay_rate)

        fc1 = fc(x=h5_reshaped_total,
                 output_size=critic_length,
                 scope="dis_final_fc_critic",
                 parameter_update_device=parameter_update_device,
                 initializer=initializer,
                 weight_decay=weight_decay,
                 name_prefix=scope, weight_decay_rate=weight_decay_rate)

        return fc2, fc1, return_str





### implementation for externet as feature extractors
def vgg_16_net(image,
               batch_size,
               device,
               label0_length,label1_length,
               keep_prob,
               initializer,
               reuse=False,
               network_usage='-1',
               output_high_level_features=[-1]):
    is_training = False
    weight_decay = False
    return_str="Vgg16Net"
    weight_decay_rate = eps

    usage_scope = network_usage + '/ext_vgg16net'

    with tf.variable_scope(usage_scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        ## block 1
        conv1_1 = relu(batch_norm(x=conv2d(x=image, output_filters=64,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv1_1',
                                           weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn1_1',
                                  parameter_update_device=device))

        conv1_2 = relu(batch_norm(x=conv2d(x=conv1_1, output_filters=64,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv1_2',
                                         weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn1_2',
                                  parameter_update_device=device))

        pool1 = tf.nn.max_pool(value=conv1_2,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')


        ## block 2
        conv2_1 = relu(batch_norm(x=conv2d(x=pool1, output_filters=128,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv2_1',
                                         weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn2_1',
                                  parameter_update_device=device))

        conv2_2 = relu(batch_norm(x=conv2d(x=conv2_1, output_filters=128,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv2_2',
                                         weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn2_2',
                                  parameter_update_device=device))

        pool2 = tf.nn.max_pool(value=conv2_2,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')


        ## block 3
        conv3_1 = relu(batch_norm(x=conv2d(x=pool2, output_filters=256,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv3_1',
                                         weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn3_1',
                                  parameter_update_device=device))

        conv3_2 = relu(batch_norm(x=conv2d(x=conv3_1, output_filters=256,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv3_2',
                                         weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn3_2',
                                  parameter_update_device=device))

        conv3_3 = relu(batch_norm(x=conv2d(x=conv3_2, output_filters=256,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv3_3',
                                         weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn3_3',
                                  parameter_update_device=device))

        pool3 = tf.nn.max_pool(value=conv3_3,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')

        ## block 4
        conv4_1 = relu(batch_norm(x=conv2d(x=pool3, output_filters=512,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv4_1',
                                         weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn4_1',
                                  parameter_update_device=device))

        conv4_2 = relu(batch_norm(x=conv2d(x=conv4_1, output_filters=512,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv4_2',
                                         weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn4_2',
                                  parameter_update_device=device))

        conv4_3 = relu(batch_norm(x=conv2d(x=conv4_2, output_filters=512,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv4_3',
                                         weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn4_3',
                                  parameter_update_device=device))

        pool4 = tf.nn.max_pool(value=conv4_3,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')


        ## block 5
        conv5_1 = relu(batch_norm(x=conv2d(x=pool4, output_filters=512,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv5_1',
                                         weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn5_1',
                                  parameter_update_device=device))

        conv5_2 = relu(batch_norm(x=conv2d(x=conv5_1, output_filters=512,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv5_2',
                                         weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn5_2',
                                  parameter_update_device=device))

        conv5_3 = relu(batch_norm(x=conv2d(x=conv5_2, output_filters=512,
                                           kh=3, kw=3,
                                           sh=1, sw=1,
                                           padding='SAME',
                                           parameter_update_device=device,
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           scope='conv5_3',
                                         weight_decay_rate=weight_decay_rate),
                                  is_training=is_training,
                                  scope='bn5_3',
                                  parameter_update_device=device))

        pool5 = tf.nn.max_pool(value=conv5_3,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool5')





        # block 6
        fc6 = tf.reshape(pool5, [batch_size, -1])
        fc6 = tf.nn.dropout(x=relu(fc(x=fc6,
                                      output_size=4096,
                                      scope="fc6",
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      parameter_update_device=device,
                                         weight_decay_rate=weight_decay_rate)),
                            keep_prob=keep_prob)

        # block 7
        fc7 = tf.nn.dropout(x=relu(fc(x=fc6,
                                      output_size=4096,
                                      scope="fc7",
                                      weight_decay=weight_decay,
                                      initializer=initializer,
                                      parameter_update_device=device,
                                      weight_decay_rate=weight_decay_rate)),
                            keep_prob=keep_prob)

        # block 8
        if not label1_length == -1:
            output_label1 = fc(x=fc7,
                               output_size=label1_length,
                               scope="output_label1",
                               weight_decay=weight_decay,
                               initializer=initializer,
                               parameter_update_device=device,
                               weight_decay_rate=weight_decay_rate)
        else:
            output_label1=-1


        if not label0_length ==  -1:
            output_label0 = fc(x=fc7,
                               output_size=label0_length,
                               scope="output_label0",
                               weight_decay=weight_decay,
                               initializer=initializer,
                               parameter_update_device=device,
                               weight_decay_rate=weight_decay_rate)
        else:
            output_label0 = -1

        features = list()
        if 1 in output_high_level_features:
            features.append(conv1_2)
        if 2 in output_high_level_features:
            features.append(conv2_2)
        if 3 in output_high_level_features:
            features.append(conv3_3)
        if 4 in output_high_level_features:
            features.append(conv4_3)
        if 5 in output_high_level_features:
            features.append(conv5_3)


        return output_label1, output_label0, features, return_str