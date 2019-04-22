

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import scipy.misc as misc
from utilities.utils import image_show

from utilities.ops import relu, batch_norm
from utilities.ops import conv2d, fc
from utilities.ops import lrelu,  layer_norm



from PIL import Image
import matplotlib.pyplot as plt


GRAYSCALE_AVG = 127.5
STANDARD_GRAYSCALE_THRESHOLD_VALUE = 240
pdf = 1e-9


read_char_path = '/DataA/Harric/ChineseCharacterExp/CASIA_Dataset/PrintedData/GB2312_L1/Font_No_0055/000000810_184219_00055.png'
read_char_path_random_style = '/DataA/Harric/ChineseCharacterExp/CASIA_Dataset/PrintedData/GB2312_L1/Font_No_0063/000000810_184219_00063.png'

model_path_list = list()
model_path_list.append('/DataA/Harric/ChineseCharacterExp/TrainedModels_Vgg16/Exp20190119_FeatureExtractor_ContentStyle_HW300Pf144_vgg16net/')
model_path_list.append('/DataA/Harric/ChineseCharacterExp/TrainedModels_Vgg16/Exp20190119_FeatureExtractor_Content_HW300Pf144_vgg16net/')
model_path_list.append('/DataA/Harric/ChineseCharacterExp/TrainedModels_Vgg16/Exp20190119_FeatureExtractor_Style_HW300Pf144_vgg16net/')
model_path_list.append('/Data_HDD/Harric/ChineseCharacterExp/tfModels2019_WNet/checkpoint/Exp20190129-WNet-DenseMixer-NonAdaIN_StylePf144_ContentPf64_GenEncDec6-Des7@Lyr3_DisMdy6conv/discriminator/')


rotate_angle_max = 15
rotate_interval = 1
translate_dist_max = 15
translate_interval = 1
saving_path_dir = '/home/harric/Desktop/EvaluationMetrics/'
MAX_EXP_NUM = 21

def feature_extractor_network(image,
                              batch_size,
                              device,
                              keep_prob,
                              initializer,
                              reuse=False,
                              network_usage='-1',
                              output_high_level_features=[-1]):
    is_training = False
    weight_decay = False
    return_str="Vgg16Net"
    weight_decay_rate = pdf

    usage_scope = network_usage + '/ext_vgg16net'

    with tf.variable_scope(usage_scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        features = list()

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

        conv1_2=conv2d(x=conv1_1, output_filters=64,
                       kh=3, kw=3,
                       sh=1, sw=1,
                       padding='SAME',
                       parameter_update_device=device,
                       weight_decay=weight_decay,
                       initializer=initializer,
                       scope='conv1_2',
                       weight_decay_rate=weight_decay_rate)
        if 1 in output_high_level_features:
            features.append(conv1_2)
        conv1_2 = relu(batch_norm(x=conv1_2,
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
        conv2_2=conv2d(x=conv2_1, output_filters=128,
                       kh=3, kw=3,
                       sh=1, sw=1,
                       padding='SAME',
                       parameter_update_device=device,
                       weight_decay=weight_decay,
                       initializer=initializer,
                       scope='conv2_2',
                       weight_decay_rate=weight_decay_rate)
        if 2 in output_high_level_features:
            features.append(conv2_2)
        conv2_2 = relu(batch_norm(x=conv2_2,
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
        conv3_3=conv2d(x=conv3_2, output_filters=256,
                       kh=3, kw=3,
                       sh=1, sw=1,
                       padding='SAME',
                       parameter_update_device=device,
                       weight_decay=weight_decay,
                       initializer=initializer,
                       scope='conv3_3',
                       weight_decay_rate=weight_decay_rate)
        if 3 in output_high_level_features:
            features.append(conv3_3)
        conv3_3 = relu(batch_norm(x=conv3_3,
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
        conv4_3=conv2d(x=conv4_2, output_filters=512,
                       kh=3, kw=3,
                       sh=1, sw=1,
                       padding='SAME',
                       parameter_update_device=device,
                       weight_decay=weight_decay,
                       initializer=initializer,
                       scope='conv4_3',
                       weight_decay_rate=weight_decay_rate)
        if 4 in output_high_level_features:
            features.append(conv4_3)
        conv4_3 = relu(batch_norm(x=conv4_3,
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
        conv5_3=conv2d(x=conv5_2, output_filters=512,
                       kh=3, kw=3,
                       sh=1, sw=1,
                       padding='SAME',
                       parameter_update_device=device,
                       weight_decay=weight_decay,
                       initializer=initializer,
                       scope='conv5_3',
                       weight_decay_rate=weight_decay_rate)
        if 5 in output_high_level_features:
            features.append(conv5_3)
        conv5_3 = relu(batch_norm(x=conv5_3,
                                  is_training=is_training,
                                  scope='bn5_3',
                                  parameter_update_device=device))
        pool5 = tf.nn.max_pool(value=conv5_3,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool5')
        # block 6
        fc6 = tf.reshape(pool5, [batch_size, -1])
        fc6 = fc(x=fc6,
                 output_size=4096,
                 scope="fc6",
                 weight_decay=weight_decay,
                 initializer=initializer,
                 parameter_update_device=device,
                 weight_decay_rate=weight_decay_rate)
        if 6 in output_high_level_features:
            features.append(fc6)
        fc6 = tf.nn.dropout(x=relu(fc6),
                            keep_prob=keep_prob)

        # block 7
        fc7 = tf.reshape(fc6, [batch_size, -1])
        fc7 = fc(x=fc7,
                 output_size=4096,
                 scope="fc7",
                 weight_decay=weight_decay,
                 initializer=initializer,
                 parameter_update_device=device,
                 weight_decay_rate=weight_decay_rate)
        if 7 in output_high_level_features:
            features.append(fc7)


        return features, fc7, return_str

def discriminator_mdy_6_convs(image,
                              device,
                              initializer,
                              reuse=False):
    critic_length = 512
    weight_decay = False
    weight_decay_rate = 0
    discriminator_dim = 32
    batch_size = 1
    scope = 'discriminator'

    return_str = ("Discriminator-6Convs")
    return_str = "WST-" + return_str + "-Crc:%d" % critic_length

    image = tf.tile(image,[1,1,1,3])

    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        h0 = lrelu(conv2d(x=image, output_filters=discriminator_dim,
                          scope="dis_h0_conv",
                          parameter_update_device=device,
                          initializer=initializer,
                          weight_decay=weight_decay,
                          name_prefix=scope,
                          weight_decay_rate=weight_decay_rate))
        h1 = lrelu(layer_norm(conv2d(x=h0,
                                     output_filters=discriminator_dim * 2,
                                     scope="dis_h1_conv",
                                     parameter_update_device=device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,weight_decay_rate=weight_decay_rate),
                              scope="dis_ln1",
                              parameter_update_device=device))
        h2 = lrelu(layer_norm(conv2d(x=h1,
                                     output_filters=discriminator_dim * 4,
                                     scope="dis_h2_conv",
                                     parameter_update_device=device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,weight_decay_rate=weight_decay_rate),
                              scope="dis_ln2",
                              parameter_update_device=device))
        h3 = lrelu(layer_norm(conv2d(x=h2,
                                     output_filters=discriminator_dim * 8,
                                     scope="dis_h3_conv",
                                     parameter_update_device=device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,weight_decay_rate=weight_decay_rate),
                              scope="dis_ln3",
                              parameter_update_device=device))

        h4 = lrelu(layer_norm(conv2d(x=h3,
                                     output_filters=discriminator_dim * 16,
                                     scope="dis_h4_conv",
                                     parameter_update_device=device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,weight_decay_rate=weight_decay_rate),
                              scope="dis_ln4",
                              parameter_update_device=device))

        h5 = lrelu(layer_norm(conv2d(x=h4,
                                     output_filters=discriminator_dim * 32,
                                     scope="dis_h5_conv",
                                     parameter_update_device=device,
                                     initializer=initializer,
                                     weight_decay=weight_decay,
                                     name_prefix=scope,weight_decay_rate=weight_decay_rate),
                              scope="dis_ln5",
                              parameter_update_device=device))


        h5_reshaped = tf.reshape(h5, [batch_size, -1])
        fc_input = h5_reshaped



        fc1 = fc(x=fc_input,
                 output_size=critic_length,
                 scope="dis_final_fc_critic",
                 parameter_update_device=device,
                 initializer=initializer,
                 weight_decay=weight_decay,
                 name_prefix=scope, weight_decay_rate=weight_decay_rate)

        return fc1, return_str

def image_rotate(img_org, rotate_angle):
    if not rotate_angle == 0:
        img_org_reversed = 1 - (img_org + 1) / 2

        img_org_reversed = img_org_reversed * (GRAYSCALE_AVG * 2)
        img_org_reversed = Image.fromarray(np.uint8(img_org_reversed))
        img_rotated = img_org_reversed.rotate(rotate_angle)
        img_rotated = np.asarray(img_rotated)
        img_rotated = (1 - img_rotated / (GRAYSCALE_AVG * 2)) * 2 - 1
    else:
        img_rotated = img_org

    return img_rotated

def image_translate(img_org, translate_pixel, axis):
    img_width = img_org.shape[0]
    if translate_pixel<0:
        translate_pixel = - translate_pixel
        if axis==0:
            pixel_1 = img_org[0:translate_pixel,:]
            pixel_2 = img_org[translate_pixel:,:]
            img_translated = np.concatenate([pixel_2,pixel_1], axis=0)
        elif axis==1:
            pixel_1 = img_org[:, 0:translate_pixel]
            pixel_2 = img_org[:, translate_pixel:]
            img_translated = np.concatenate([pixel_2, pixel_1], axis=1)
    elif translate_pixel>0:
        translate_pixel = img_width - translate_pixel
        if axis==0:
            pixel_1 = img_org[0:translate_pixel,:]
            pixel_2 = img_org[translate_pixel:,:]
            img_translated = np.concatenate([pixel_2,pixel_1], axis=0)
        elif axis == 1:
            pixel_1 = img_org[:, 0:translate_pixel]
            pixel_2 = img_org[:, translate_pixel:]
            img_translated = np.concatenate([pixel_2, pixel_1], axis=1)
    else:
        img_translated = img_org

    return img_translated

def pixel_calculate(current_rotated,img_org):
    threshold = STANDARD_GRAYSCALE_THRESHOLD_VALUE / GRAYSCALE_AVG - 1
    binaried_rotated = np.copy(current_rotated)
    binraied_org = np.copy(img_org)
    binaried_rotated[np.where(binaried_rotated <= threshold)] = 0
    binaried_rotated[np.where(binaried_rotated > threshold)] = 1
    binraied_org[np.where(binraied_org <= threshold)] = 0
    binraied_org[np.where(binraied_org > threshold)] = 1
    current_pdar_diff = np.abs(binaried_rotated - binraied_org)
    current_pdar = float(np.sum(current_pdar_diff)) / (
            current_pdar_diff.shape[0] * current_pdar_diff.shape[1] * current_pdar_diff.shape[2])


    current_l1 = np.abs(img_org - current_rotated)
    current_squared_diff = current_l1 ** 2
    current_mse = np.sqrt(np.mean(current_squared_diff))
    current_l1 = np.mean(current_l1)

    return current_l1, current_mse, current_pdar


def restore_model(sess, saver, model_dir, model_name):
    def correct_ckpt_path(real_dir, maybe_path):
        maybe_path_dir = str(os.path.split(os.path.realpath(maybe_path))[0])
        if not maybe_path_dir == real_dir:
            return os.path.join(real_dir, str(os.path.split(os.path.realpath(maybe_path))[1]))
        else:
            return maybe_path

    ckpt = tf.train.get_checkpoint_state(model_dir)
    corrected_ckpt_path = correct_ckpt_path(real_dir=model_dir,
                                            maybe_path=ckpt.model_checkpoint_path)
    if ckpt:
        saver.restore(sess, corrected_ckpt_path)
        print("ModelRestored:%s" % model_name)
        print("@%s" % model_dir)
        return True
    else:
        print("fail to restore model %s" % model_dir)
        return False

def feature_extractor_build(org_img, rotated_img, device, initializer):

    def find_norm_avg_var(var_list):
        var_list_new = list()
        for ii in var_list:
            var_list_new.append(ii)

        all_vars = tf.global_variables()
        bn_var_list = [var for var in var_list if 'bn' in var.name] # for batch norms
        #ln_var_list = [var for var in var_list if 'ln' in var.name] # for layer norms

        norm_var_list = list()
        norm_var_list.extend(bn_var_list)
        #norm_var_list.extend(ln_var_list)

        output_avg_var = list()
        for n_var in norm_var_list:
            if 'gamma' in n_var.name:
                continue
            n_var_name = n_var.name
            variance_name = n_var_name.replace('beta', 'moving_variance')
            average_name = n_var_name.replace('beta', 'moving_mean')
            variance_var = [var for var in all_vars if variance_name in var.name][0]
            average_var = [var for var in all_vars if average_name in var.name][0]
            output_avg_var.append(variance_var)
            output_avg_var.append(average_var)

        var_list_new.extend(output_avg_var)

        output = list()
        for ii in var_list_new:
            if ii not in output:
                output.append(ii)

        return output

    def variable_dict(var_input,delete_name_from_character):
        var_output = {}
        for ii in var_input:
            prefix_pos = ii.name.find(delete_name_from_character)
            renamed = ii.name[prefix_pos + 1:]
            parafix_pos = renamed.find(':')
            renamed = renamed[0:parafix_pos]
            var_output.update({renamed: ii})
        return var_output


    def feature_linear_norm(feature):
        min_v = tf.reduce_min(feature)
        feature = feature - min_v
        max_v = tf.reduce_max(feature)
        feature = feature / max_v
        return feature + pdf

    def calculate_high_level_feature_loss(feature1, feature2):

        mse_diff_list = list()
        vn_diff_list = list()
        for counter in range(len(feature1)):
            # mse calculas
            feature_diff = feature1[counter] - feature2[counter]
            if not feature_diff.shape.ndims == 4:
                feature_diff = tf.reshape(feature_diff, [int(feature_diff.shape[0]), int(feature_diff.shape[1]), 1, 1])
            squared_feature_diff = feature_diff ** 2
            mean_squared_feature_diff = tf.reduce_mean(squared_feature_diff, axis=[1, 2, 3])
            square_root_mean_squared_feature_diff = tf.sqrt(pdf + mean_squared_feature_diff)
            square_root_mean_squared_feature_diff = tf.reshape(square_root_mean_squared_feature_diff,
                                                               [int(square_root_mean_squared_feature_diff.shape[0]), 1])
            this_mse_loss = tf.reduce_sum(square_root_mean_squared_feature_diff, axis=0)
            this_mse_loss = tf.reshape(this_mse_loss, shape=[1, 1])
            mse_diff_list.append(this_mse_loss)



            # vn divergence calculas
            feature1_normed = feature_linear_norm(feature=feature1[counter])
            feature2_normed = feature_linear_norm(feature=feature2[counter])

            if not feature1_normed.shape.ndims == 2:
                vn_loss = tf.trace(tf.multiply(feature1_normed, tf.log(feature1_normed)) -
                                   tf.multiply(feature1_normed, tf.log(feature2_normed)) +
                                   - feature1_normed + feature2_normed + pdf)
                vn_loss = tf.reduce_mean(vn_loss, axis=1)
                vn_loss = tf.reduce_sum(vn_loss)
                vn_loss = tf.reshape(vn_loss, shape=[1, 1])
            vn_diff_list.append(vn_loss)



        return mse_diff_list, vn_diff_list

    def build_feature_extractor(input_true_img, input_generated_img,
                                reuse, reuse_discriminator,
                                extractor_usage, output_high_level_features,
                                device,initializer):
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(device):
                real_features, real_logits, network_info = \
                    feature_extractor_network(image=input_true_img,
                                              batch_size=1,
                                              device=device,
                                              reuse=reuse,
                                              keep_prob=1,
                                              initializer=initializer,
                                              network_usage=extractor_usage,
                                              output_high_level_features=output_high_level_features)
                fake_features, fake_logits, _ = \
                    feature_extractor_network(image=input_generated_img,
                                              batch_size=1,
                                              device=device,
                                              reuse=True,
                                              keep_prob=1,
                                              initializer=initializer,
                                              network_usage=extractor_usage,
                                              output_high_level_features=output_high_level_features)

                real_wasserstein,return_str = discriminator_mdy_6_convs(image=input_true_img,
                                                                        device=device,
                                                                        initializer=initializer,
                                                                        reuse=reuse_discriminator)
                fake_wasserstein, _ = discriminator_mdy_6_convs(image=input_generated_img,
                                                                device=device,
                                                                initializer=initializer,
                                                                reuse=True)

        feature_loss_mse, feature_loss_vn = calculate_high_level_feature_loss(feature1=real_features,
                                                                              feature2=fake_features)
        wasserstein_diff = tf.abs(tf.reduce_mean(real_wasserstein - fake_wasserstein))

        real_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = real_logits, labels = tf.nn.softmax(real_logits))
        fake_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= fake_logits, labels=tf.nn.softmax(fake_logits))
        entropy_diff = tf.abs(real_entropy-fake_entropy)


        return feature_loss_mse, feature_loss_vn, entropy_diff, wasserstein_diff, network_info

    saver_list = list()
    mse_list = list()
    vn_list = list()
    entropy_list = list()

    true_fake_feature_mse, true_fake_feature_vn, true_fake_entropy, wasserstein_diff, network_info = \
        build_feature_extractor(input_true_img=org_img,
                                input_generated_img=rotated_img,
                                extractor_usage='TrueFake_FeatureExtractor',
                                output_high_level_features=[1, 2, 3, 4, 5, 6, 7],
                                reuse=False,reuse_discriminator=False,
                                device=device,
                                initializer=initializer)

    extr_vars_true_fake = [var for var in tf.trainable_variables() if 'TrueFake_FeatureExtractor' in var.name]
    extr_vars_true_fake = find_norm_avg_var(extr_vars_true_fake)
    extr_vars_true_fake = variable_dict(var_input=extr_vars_true_fake, delete_name_from_character='/')
    saver_extractor_true_fake = tf.train.Saver(max_to_keep=1, var_list=extr_vars_true_fake)
    print("TrueFakeExtractor @ %s with %s;" % (device, network_info))

    dis_vars_train = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
    dis_vars_save = find_norm_avg_var(dis_vars_train)
    saver_discriminataor = tf.train.Saver(max_to_keep=1, var_list=dis_vars_save)

    content_prototype_feature_mse,  content_prototype_feature_vn, content_entropy, _, network_info = \
        build_feature_extractor(input_true_img=org_img,
                                input_generated_img=rotated_img,
                                extractor_usage='ContentPrototype_FeatureExtractor',
                                output_high_level_features=[1, 2, 3, 4, 5, 6, 7],
                                reuse=False,reuse_discriminator=True,
                                device=device,
                                initializer=initializer)
    extr_vars_content_prototype = [var for var in tf.trainable_variables() if
                                   'ContentPrototype_FeatureExtractor' in var.name]
    extr_vars_content_prototype = find_norm_avg_var(extr_vars_content_prototype)
    extr_vars_content_prototype = variable_dict(var_input=extr_vars_content_prototype,
                                                delete_name_from_character='/')
    saver_extractor_content_prototype = tf.train.Saver(max_to_keep=1, var_list=extr_vars_content_prototype)
    print("ContentPrototypeExtractor @ %s with %s;" % (device, network_info))

    style_reference_feature_mse, style_reference_feature_vn, style_entropy, _, network_info = \
        build_feature_extractor(input_true_img=org_img,
                                input_generated_img=rotated_img,
                                extractor_usage='StyleReference_FeatureExtractor',
                                output_high_level_features=[1, 2, 3, 4, 5, 6, 7],
                                reuse=False,reuse_discriminator=True,
                                device=device,
                                initializer=initializer)
    extr_vars_style_reference = [var for var in tf.trainable_variables() if
                                 'StyleReference_FeatureExtractor' in var.name]
    extr_vars_style_reference = find_norm_avg_var(extr_vars_style_reference)
    extr_vars_style_reference = variable_dict(var_input=extr_vars_style_reference,
                                              delete_name_from_character='/')
    saver_extractor_style_reference = tf.train.Saver(max_to_keep=1, var_list=extr_vars_style_reference)
    print("StyleReferenceExtractor @ %s with %s;" % (device, network_info))


    saver_list.append(saver_extractor_true_fake)
    saver_list.append(saver_extractor_content_prototype)
    saver_list.append(saver_extractor_style_reference)
    saver_list.append(saver_discriminataor)

    mse_list.append(true_fake_feature_mse)
    mse_list.append(content_prototype_feature_mse)
    mse_list.append(style_reference_feature_mse)

    vn_list.append(true_fake_feature_vn)
    vn_list.append(content_prototype_feature_vn)
    vn_list.append(style_reference_feature_vn)

    entropy_list.append(true_fake_entropy)
    entropy_list.append(content_entropy)
    entropy_list.append(style_entropy)
    return saver_list, mse_list, vn_list, entropy_list, wasserstein_diff


def main():
    img_org = misc.imread(read_char_path)
    img_org = img_org / GRAYSCALE_AVG - 1

    img_upper_bound = misc.imread(read_char_path_random_style)
    img_upper_bound = img_upper_bound / GRAYSCALE_AVG - 1

    rotate_list_positive = np.arange(0, rotate_angle_max, rotate_interval)
    rotate_list_negative = np.arange(0, -rotate_angle_max, -rotate_interval)
    rotate_list = list()
    rotate_list.extend(rotate_list_positive)
    rotate_list.extend(rotate_list_negative)
    rotate_list = sorted(set(rotate_list), key=rotate_list.index)
    rotate_list.sort()

    translate_list_positive = range(0, translate_dist_max, translate_interval)
    translate_list_negative = range(0, -translate_dist_max, -translate_interval)
    translate_list = list()
    translate_list.extend(translate_list_positive)
    translate_list.extend(translate_list_negative)
    translate_list = sorted(set(translate_list), key=translate_list.index)
    translate_list.sort()




    weight_decay = True
    initializer = 'XavierInit'


    with tf.Graph().as_default():
        summary_seconds = 30

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        run_device = '/device:CPU:0'

        org_img_tensor = tf.placeholder(dtype=tf.float32,
                                        shape=[1, 64, 64, 1])
        rtt_img_tensor = tf.placeholder(dtype=tf.float32,
                                        shape=[1, 64, 64, 1])

        with tf.device(run_device):
            with tf.variable_scope(tf.get_variable_scope()):
                saver_list, feature_mse_tensor_list, feature_vn_tensor_list, entropy_list, wasserstein_dist = \
                    feature_extractor_build(org_img=org_img_tensor, rotated_img=rtt_img_tensor,
                                            device=run_device,initializer=initializer)

        restore_model(sess=sess, saver=saver_list[0], model_dir=model_path_list[0], model_name='TrueFake')
        restore_model(sess=sess, saver=saver_list[1], model_dir=model_path_list[1], model_name='Content')
        restore_model(sess=sess, saver=saver_list[2], model_dir=model_path_list[2], model_name='Style')
        restore_model(sess=sess, saver=saver_list[3], model_dir=model_path_list[3], model_name='Discriminator')


    # upper bound
    upper_bound_l1, upper_bound_mse, upper_bound_pdar = \
        pixel_calculate(current_rotated=img_upper_bound,
                        img_org=img_org)

    upper_bound_feature_mse_list, upper_bound_feature_vn_list, upper_bound_entropy, upper_bound_wasserstein = \
        sess.run([feature_mse_tensor_list, feature_vn_tensor_list, entropy_list, wasserstein_dist],
                 feed_dict={org_img_tensor: np.reshape(img_org[:, :, 0], [1, 64, 64, 1]),
                            rtt_img_tensor: np.reshape(img_upper_bound[:, :, 0], [1, 64, 64, 1])})

    # difference evaluation
    rotate_output_pixel_l1_list = list()
    rotate_output_pixel_mse_list = list()
    rotate_output_pixel_pdar_list = list()
    rotate_output_entropy_list = list()
    rotate_output_wasserstein_list = list()
    translate0_output_pixel_l1_list = list()
    translate0_output_pixel_mse_list = list()
    translate0_output_pixel_pdar_list = list()
    translate0_output_entropy_list = list()
    translate0_output_wasserstein_list = list()
    translate1_output_pixel_l1_list = list()
    translate1_output_pixel_mse_list = list()
    translate1_output_pixel_pdar_list = list()
    translate1_output_entropy_list = list()
    translate1_output_wasserstein_list = list()


    rotate_output_feature_true_fake_mse_list = list()
    rotate_output_feature_content_mse_list = list()
    rotate_output_feature_style_mse_list = list()
    rotate_output_feature_true_fake_vn_list = list()
    rotate_output_feature_content_vn_list = list()
    rotate_output_feature_style_vn_list = list()
    translate0_output_feature_true_fake_mse_list = list()
    translate0_output_feature_content_mse_list = list()
    translate0_output_feature_style_mse_list = list()
    translate0_output_feature_true_fake_vn_list = list()
    translate0_output_feature_content_vn_list = list()
    translate0_output_feature_style_vn_list = list()
    translate1_output_feature_true_fake_mse_list = list()
    translate1_output_feature_content_mse_list = list()
    translate1_output_feature_style_mse_list = list()
    translate1_output_feature_true_fake_vn_list = list()
    translate1_output_feature_content_vn_list = list()
    translate1_output_feature_style_vn_list = list()
    for ii in range(7):
        rotate_output_feature_true_fake_mse_list.append(list())
        rotate_output_feature_content_mse_list.append(list())
        rotate_output_feature_style_mse_list.append(list())
        rotate_output_feature_true_fake_vn_list.append(list())
        rotate_output_feature_content_vn_list.append(list())
        rotate_output_feature_style_vn_list.append(list())
        translate0_output_feature_true_fake_mse_list.append(list())
        translate0_output_feature_content_mse_list.append(list())
        translate0_output_feature_style_mse_list.append(list())
        translate0_output_feature_true_fake_vn_list.append(list())
        translate0_output_feature_content_vn_list.append(list())
        translate0_output_feature_style_vn_list.append(list())
        translate1_output_feature_true_fake_mse_list.append(list())
        translate1_output_feature_content_mse_list.append(list())
        translate1_output_feature_style_mse_list.append(list())
        translate1_output_feature_true_fake_vn_list.append(list())
        translate1_output_feature_content_vn_list.append(list())
        translate1_output_feature_style_vn_list.append(list())

    for translate_dist in translate_list:
        current_translated_0 = image_translate(img_org=img_org,
                                               translate_pixel=translate_dist,
                                               axis=0)
        current_l1_0, current_mse_0, current_pdar_0 = \
            pixel_calculate(current_rotated=current_translated_0,
                            img_org=img_org)
        current_feature_mse_list_0, current_feature_vn_list_0, current_entropy_0, current_wasserstein_0 = \
            sess.run([feature_mse_tensor_list, feature_vn_tensor_list, entropy_list, wasserstein_dist],
                     feed_dict={org_img_tensor: np.reshape(img_org[:, :, 0], [1, 64, 64, 1]),
                                rtt_img_tensor: np.reshape(current_translated_0[:, :, 0], [1, 64, 64, 1])})


        current_translated_1 = image_translate(img_org=img_org,
                                               translate_pixel=translate_dist,
                                               axis=1)
        current_l1_1, current_mse_1, current_pdar_1 = \
            pixel_calculate(current_rotated=current_translated_1,
                            img_org=img_org)
        current_feature_mse_list_1, current_feature_vn_list_1, current_entropy_1, current_wasserstein_1 = \
            sess.run([feature_mse_tensor_list, feature_vn_tensor_list, entropy_list, wasserstein_dist],
                     feed_dict={org_img_tensor: np.reshape(img_org[:, :, 0], [1, 64, 64, 1]),
                                rtt_img_tensor: np.reshape(current_translated_1[:, :, 0], [1, 64, 64, 1])})


        for ii in range(7):
            translate0_output_feature_true_fake_mse_list[ii].append(current_feature_mse_list_0[0][ii][0][0]/upper_bound_feature_mse_list[0][ii][0])
            translate0_output_feature_content_mse_list[ii].append(current_feature_mse_list_0[1][ii][0][0]/upper_bound_feature_mse_list[1][ii][0])
            translate0_output_feature_style_mse_list[ii].append(current_feature_mse_list_0[2][ii][0][0]/upper_bound_feature_mse_list[2][ii][0])
            translate0_output_feature_true_fake_vn_list[ii].append(current_feature_vn_list_0[0][ii][0][0]/upper_bound_feature_vn_list[0][ii][0])
            translate0_output_feature_content_vn_list[ii].append(current_feature_vn_list_0[1][ii][0][0]/upper_bound_feature_vn_list[1][ii][0])
            translate0_output_feature_style_vn_list[ii].append(current_feature_vn_list_0[2][ii][0][0]/upper_bound_feature_vn_list[2][ii][0])
            translate1_output_feature_true_fake_mse_list[ii].append(current_feature_mse_list_1[0][ii][0][0] / upper_bound_feature_mse_list[0][ii][0])
            translate1_output_feature_content_mse_list[ii].append(current_feature_mse_list_1[1][ii][0][0] / upper_bound_feature_mse_list[1][ii][0])
            translate1_output_feature_style_mse_list[ii].append(current_feature_mse_list_1[2][ii][0][0] / upper_bound_feature_mse_list[2][ii][0])
            translate1_output_feature_true_fake_vn_list[ii].append(current_feature_vn_list_1[0][ii][0][0] / upper_bound_feature_vn_list[0][ii][0])
            translate1_output_feature_content_vn_list[ii].append(current_feature_vn_list_1[1][ii][0][0] / upper_bound_feature_vn_list[1][ii][0])
            translate1_output_feature_style_vn_list[ii].append(current_feature_vn_list_1[2][ii][0][0] / upper_bound_feature_vn_list[2][ii][0])

        translate0_output_pixel_l1_list.append(current_l1_0 / upper_bound_l1)
        translate0_output_pixel_mse_list.append(current_mse_0 / upper_bound_mse)
        translate0_output_pixel_pdar_list.append(current_pdar_0 / upper_bound_pdar)
        translate0_output_entropy_list.append(current_entropy_0[2][0] / upper_bound_entropy[2][0])
        translate0_output_wasserstein_list.append(current_wasserstein_0 / upper_bound_wasserstein)
        translate1_output_pixel_l1_list.append(current_l1_1 / upper_bound_l1)
        translate1_output_pixel_mse_list.append(current_mse_1 / upper_bound_mse)
        translate1_output_pixel_pdar_list.append(current_pdar_1 / upper_bound_pdar)
        translate1_output_entropy_list.append(current_entropy_1[2][0] / upper_bound_entropy[2][0])
        translate1_output_wasserstein_list.append(current_wasserstein_1 / upper_bound_wasserstein)


    for rotate_angle in rotate_list:
        current_rotated = image_rotate(img_org=img_org,
                                       rotate_angle=rotate_angle)


        current_l1, current_mse, current_pdar = \
            pixel_calculate(current_rotated=current_rotated,
                            img_org=img_org)

        current_feature_mse_list, current_feature_vn_list, current_entropy,current_wasserstein = \
            sess.run([feature_mse_tensor_list,feature_vn_tensor_list, entropy_list, wasserstein_dist],
                     feed_dict={org_img_tensor:np.reshape(img_org[:,:,0],[1,64,64,1]),
                                rtt_img_tensor:np.reshape(current_rotated[:, :, 0], [1, 64, 64, 1])})

        for ii in range(7):
            rotate_output_feature_true_fake_mse_list[ii].append(current_feature_mse_list[0][ii][0][0]/upper_bound_feature_mse_list[0][ii][0])
            rotate_output_feature_content_mse_list[ii].append(current_feature_mse_list[1][ii][0][0]/upper_bound_feature_mse_list[1][ii][0])
            rotate_output_feature_style_mse_list[ii].append(current_feature_mse_list[2][ii][0][0]/upper_bound_feature_mse_list[2][ii][0])
            rotate_output_feature_true_fake_vn_list[ii].append(current_feature_vn_list[0][ii][0][0]/upper_bound_feature_vn_list[0][ii][0])
            rotate_output_feature_content_vn_list[ii].append(current_feature_vn_list[1][ii][0][0]/upper_bound_feature_vn_list[1][ii][0])
            rotate_output_feature_style_vn_list[ii].append(current_feature_vn_list[2][ii][0][0]/upper_bound_feature_vn_list[2][ii][0])

        rotate_output_pixel_pdar_list.append(current_pdar/upper_bound_pdar)
        rotate_output_pixel_l1_list.append(current_l1/upper_bound_l1)
        rotate_output_pixel_mse_list.append(current_mse/upper_bound_mse)
        rotate_output_entropy_list.append(current_entropy[2][0]/upper_bound_entropy[2][0])
        rotate_output_wasserstein_list.append(current_wasserstein/upper_bound_wasserstein)

    if not os.path.exists(saving_path_dir):
        os.makedirs(saving_path_dir)

    color_full = np.zeros(shape=[MAX_EXP_NUM, 3],
                          dtype=np.float32)
    for color_counter in range(MAX_EXP_NUM):
        current_random_color = np.random.uniform(low=0.3, high=0.7, size=[1, 3])
        color_full[color_counter, :] = current_random_color

    plt.figure(figsize=(10, 7))
    plt.plot(rotate_list, rotate_output_pixel_l1_list, color='red', label='Pixel-L1')
    plt.plot(rotate_list, rotate_output_pixel_mse_list, color='green', label='Pixel-MSE')
    plt.plot(rotate_list, rotate_output_pixel_pdar_list, color='blue', label='Pixel-PDAR')
    rotate_list = list(map(lambda n: n - 0.3, rotate_list))
    plt.plot(rotate_list, rotate_output_feature_style_mse_list[0], color='lightcoral', marker='o', label='B1-L2-MSE')
    plt.plot(rotate_list, rotate_output_feature_style_mse_list[1], color='chocolate', marker='o', label='B2-L2-MSE')
    plt.plot(rotate_list, rotate_output_feature_style_mse_list[2], color='gold', marker='o', label='B3-L3-MSE')
    plt.plot(rotate_list, rotate_output_feature_style_mse_list[3], color='lime', marker='o', label='B4-L3-MSE')
    plt.plot(rotate_list, rotate_output_feature_style_mse_list[4], color='darkslategrey', marker='o', label='B5-L3-MSE')
    # plt.plot(rotate_list, rotate_output_feature_style_mse_list[5], color='cornflowerblue', marker='o', label='B6-MSE')
    # plt.plot(rotate_list, rotate_output_feature_style_mse_list[6], color='lightpink', marker='o', label='B7-MSE')
    rotate_list = list(map(lambda n: n + 0.6, rotate_list))
    plt.plot(rotate_list, rotate_output_feature_style_vn_list[0], color='darkred', marker='*', label='B1-L2-VN')
    plt.plot(rotate_list, rotate_output_feature_style_vn_list[1], color='slategray', marker='*', label='B2-L2-VN')
    plt.plot(rotate_list, rotate_output_feature_style_vn_list[2], color='y', marker='*', label='B3-L3-VN')
    plt.plot(rotate_list, rotate_output_feature_style_vn_list[3], color='lightgreen', marker='*', label='B4-L3-VN')
    plt.plot(rotate_list, rotate_output_feature_style_vn_list[4], color='hotpink', marker='*', label='B5-L3-VN')
    # plt.plot(rotate_list, rotate_output_feature_style_vn_list[5], color='bisque', marker='*', label='B6-VN')
    # plt.plot(rotate_list, rotate_output_feature_style_vn_list[6], color='black', marker='*', label='B7-VN')
    rotate_list = list(map(lambda n: n - 0.3, rotate_list))
    plt.axis([-rotate_angle_max, rotate_angle_max, -0.01, 1.05])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Rotating Angle', fontsize=20)
    plt.ylabel('Relative Discrepancy', fontsize=20)
    plt.legend(fontsize=12, loc=4)
    # plt.show()
    plt.savefig(os.path.join(saving_path_dir, './Rotation.pdf'))

    plt.figure(figsize=(10, 7))
    plt.plot(translate_list, translate0_output_pixel_l1_list, color='red', label='Pixel-L1')
    plt.plot(translate_list, translate0_output_pixel_mse_list, color='green', label='Pixel-MSE')
    plt.plot(translate_list, translate0_output_pixel_pdar_list, color='blue', label='Pixel-PDAR')
    translate_list = list(map(lambda n: n - 0.3, translate_list))
    plt.plot(translate_list, translate0_output_feature_style_mse_list[0], color='lightcoral', marker='o',
             label='B1-L2-MSE')
    plt.plot(translate_list, translate0_output_feature_style_mse_list[1], color='chocolate', marker='o',
             label='B2-L2-MSE')
    plt.plot(translate_list, translate0_output_feature_style_mse_list[2], color='gold', marker='o', label='B3-L3-MSE')
    plt.plot(translate_list, translate0_output_feature_style_mse_list[3], color='lime', marker='o', label='B4-L3-MSE')
    plt.plot(translate_list, translate0_output_feature_style_mse_list[4], color='darkslategrey', marker='o',
             label='B5-L3-MSE')
    # plt.plot(translate_list, translate0_output_feature_style_mse_list[5], color='cornflowerblue', marker='o', label='B6-MSE')
    # plt.plot(translate_list, translate0_output_feature_style_mse_list[6], color='lightpink', marker='o', label='B7-MSE')
    translate_list = list(map(lambda n: n + 0.6, translate_list))
    plt.plot(translate_list, translate0_output_feature_style_vn_list[0], color='darkred', marker='*', label='B1-L2-VN')
    plt.plot(translate_list, translate0_output_feature_style_vn_list[1], color='slategray', marker='*', label='B2-L2-VN')
    plt.plot(translate_list, translate0_output_feature_style_vn_list[2], color='y', marker='*', label='B3-L3-VN')
    plt.plot(translate_list, translate0_output_feature_style_vn_list[3], color='lightgreen', marker='*',
             label='B4-L3-VN')
    plt.plot(translate_list, translate0_output_feature_style_vn_list[4], color='hotpink', marker='*', label='B5-L3-VN')
    # plt.plot(translate_list, translate0_output_feature_style_vn_list[5], color='bisque', marker='*', label='B6-VN')
    # plt.plot(translate_list, translate0_output_feature_style_vn_list[6], color='black', marker='*', label='B7-VN')
    translate_list = list(map(lambda n: n - 0.3, translate_list))
    plt.axis([-translate_dist_max, translate_dist_max, -0.01, 1.05])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Vertical Translating Pixel', fontsize=20)
    plt.ylabel('Relative Discrepancy', fontsize=20)
    plt.legend(fontsize=12, loc=4)
    # plt.show()
    plt.savefig(os.path.join(saving_path_dir, './VerticalTranslation.pdf'))

    plt.figure(figsize=(10, 7))
    plt.plot(translate_list, translate1_output_pixel_l1_list, color='red', label='Pixel-L1')
    plt.plot(translate_list, translate1_output_pixel_mse_list, color='green', label='Pixel-MSE')
    plt.plot(translate_list, translate1_output_pixel_pdar_list, color='blue', label='Pixel-PDAR')
    translate_list = list(map(lambda n: n - 0.3, translate_list))
    plt.plot(translate_list, translate1_output_feature_style_mse_list[0], color='lightcoral', marker='o',
             label='B1-L2-MSE')
    plt.plot(translate_list, translate1_output_feature_style_mse_list[1], color='chocolate', marker='o',
             label='B2-L2-MSE')
    plt.plot(translate_list, translate1_output_feature_style_mse_list[2], color='gold', marker='o', label='B3-L3-MSE')
    plt.plot(translate_list, translate1_output_feature_style_mse_list[3], color='lime', marker='o', label='B4-L3-MSE')
    plt.plot(translate_list, translate1_output_feature_style_mse_list[4], color='darkslategrey', marker='o',
             label='B5-L3-MSE')
    # plt.plot(translate_list, translate1_output_feature_style_mse_list[5], color='cornflowerblue', marker='o', label='B6-MSE')
    # plt.plot(translate_list, translate1_output_feature_style_mse_list[6], color='lightpink', marker='o', label='B7-MSE')
    translate_list = list(map(lambda n: n + 0.6, translate_list))
    plt.plot(translate_list, translate1_output_feature_style_vn_list[0], color='darkred', marker='*', label='B1-L2-VN')
    plt.plot(translate_list, translate1_output_feature_style_vn_list[1], color='slategray', marker='*', label='B2-L2-VN')
    plt.plot(translate_list, translate1_output_feature_style_vn_list[2], color='y', marker='*', label='B3-L3-VN')
    plt.plot(translate_list, translate1_output_feature_style_vn_list[3], color='lightgreen', marker='*',
             label='B4-L3-VN')
    plt.plot(translate_list, translate1_output_feature_style_vn_list[4], color='hotpink', marker='*', label='B5-L3-VN')
    # plt.plot(translate_list, translate1_output_feature_style_vn_list[5], color='bisque', marker='*', label='B6-VN')
    # plt.plot(translate_list, translate1_output_feature_style_vn_list[6], color='black', marker='*', label='B7-VN')
    plt.axis([-translate_dist_max, translate_dist_max, -0.01, 1.05])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Horizontal Translating Pixel', fontsize=20)
    plt.ylabel('Relative Discrepancy', fontsize=20)
    plt.legend(fontsize=12, loc=4)
    # plt.show()
    plt.savefig(os.path.join(saving_path_dir, './HorizontalTranslation.pdf'))














if __name__ == "__main__":
    main()