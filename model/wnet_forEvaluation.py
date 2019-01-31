# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

GRAYSCALE_AVG = 127.5

import sys
sys.path.append('..')
from utilities.utils import image_show


import tensorflow as tf
import numpy as np
import random as rnd
import scipy.misc as misc
import os
import shutil
import time
from collections import namedtuple
from dataset.dataset import DataProvider
import re

from utilities.utils import scale_back_for_img, scale_back_for_dif, merge, correct_ckpt_path
from model.discriminators import discriminator_mdy_6_convs


from model.vggs import vgg_16_net as feature_extractor_network


from model.generators import WNet_Generator as wnet_generator
from model.generators import EmdNet_Generator as emdnet_generator
from model.generators import ResEmd_EmdNet_Generator as resemdnet_generator
from model.generators import AdobeNet_Generator as adobenet_generator
from model.generators import ResMixerNet_Generator as resmixernet_generator


from model.encoders import encoder_framework as encoder_implementation
from model.encoders import encoder_resemd_framework as encoder_ResEmd_framework
from model.encoders import encoder_adobenet_framework as encoder_Adobe_framework
from model.encoders import encoder_resmixernet_framework as encoder_resmixernet_framework



import math
import utilities.infer_implementations as inf_tools

STANDARD_GRAYSCALE_THRESHOLD_VALUE = 240


# Auxiliary wrapper classes
# Used to save handles(important nodes in computation graph) for later evaluation
SummaryHandle = namedtuple("SummaryHandle", ["d_merged", "g_merged",
                                             "check_validate_image_summary", "check_train_image_summary",
                                             "check_validate_image", "check_train_image",
                                             "learning_rate",
                                             "trn_real_dis_extr_summaries","val_real_dis_extr_summaries",
                                             "trn_fake_dis_extr_summaries","val_fake_dis_extr_summaries"])

EvalHandle = namedtuple("EvalHandle",["inferring_generated_images","training_generated_images"])

GeneratorHandle = namedtuple("Generator",
                             ["generated_target"])

DiscriminatorHandle = namedtuple("Discriminator",
                                 ["current_critic_logit_penalty",
                                  "infer_label1","infer_content_prototype","infer_style_reference","infer_true_fake"])

FeatureExtractorHandle = namedtuple("FeatureExtractor",
                                    ["infer_input_img","true_label0","true_label1",
                                     "selected_content_prototype","selected_style_reference"])



generator_dict = {"WNet": wnet_generator,
                  "EmdNet": emdnet_generator,
                  "ResEmdNet": resemdnet_generator,
                  "AdobeNet": adobenet_generator,
                  "ResMixerNet": resmixernet_generator
                  }

encoder_dict = {"Normal": encoder_implementation,
                "ResEmdNet": encoder_ResEmd_framework,
                "AdobeNet": encoder_Adobe_framework,
                "ResMixerNet": encoder_resmixernet_framework}

eps = 1e-9

class WNet(object):

    # constructor
    def __init__(self,
                 debug_mode=-1,
                 style_input_number=-1,
                 evaluation_resule_save_dir='/tmp/',

                 experiment_id='0',
                 content_data_dir='/tmp/',
                 style_train_data_dir='/tmp/',
                 fixed_style_reference_dir=None,

                 file_list_txt_content=None,
                 file_list_txt_style_train=None,
                 fixed_file_list_txt_style_reference=None,
                 fixed_char_list_txt=None,
                 channels=-1,


                 batch_size=8, img_width=256,





                 generator_devices='/device:CPU:0',
                 feature_extractor_devices='/device:CPU:0',

                 generator_residual_at_layer=3,
                 generator_residual_blocks=5,
                 true_fake_target_extractor_dir='/tmp/',
                 content_prototype_extractor_dir='/tmp/',
                 style_reference_extractor_dir='/tmp/',
                 evaluating_generator_dir='/tmp/',



                 ):

        self.print_separater = "#############################################################################"

        self.initializer = 'XavierInit'
        self.style_input_number=style_input_number

        self.experiment_id = experiment_id
        self.evaluation_resule_save_dir=evaluation_resule_save_dir

        self.adain_use = ('AdaIN' in experiment_id) and (not 'NonAdaIN' in experiment_id)
        if self.adain_use and 'AdaIN-Multi' in self.experiment_id:
            self.adain_preparation_model = 'Multi'
            self.adain_mark = '1-Multi'
        elif self.adain_use and 'AdaIN-Single' in self.experiment_id:
            self.adain_preparation_model = 'Single'
            self.adain_mark = '1-Single'
        else:
            self.adain_preparation_model = None
            self.adain_mark = '0'

        if ('NonAdaIN' in experiment_id) and self.adain_use:
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)
            print("Error: AdaIN Comflicts in ExperimentID and AdaIN Marks")
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)

        if ('AdaIN' in experiment_id and (not self.adain_use)) or \
                ((not 'AdaIN' in experiment_id) and self.adain_use):
            if not 'NonAdaIN' in experiment_id:
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print("Error: AdaIN Comflicts in ExperimentID and AdaIN Marks")
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                return
        if ('AdaIN' in experiment_id or self.adain_use) and (('Res' in experiment_id and 'Emd' in experiment_id)
                                                             or 'Adobe' in experiment_id
                                                             or 'ResMixer' in experiment_id):

            if ('Res' in experiment_id and 'Emd' in experiment_id):
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print("Error: No AdaIN mode in ResEmdNet")
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)

            if 'Adobe' in experiment_id:
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print("Error: No AdaIN in AdobeNet")
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)

            if 'ResMixer' in experiment_id:
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print("Error: No AdaIN in ResMixerNet")
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
            return

        self.other_info=None
        self.generator_residual_at_layer = -1
        self.generator_residual_blocks = -1
        if 'Emd' in experiment_id:
            if 'Res' in experiment_id:
                self.generator_implementation=generator_dict['ResEmdNet']
                self.encoder_implementation = encoder_dict['ResEmdNet']
                if 'NN' in experiment_id:
                    self.other_info = 'NN'
            else:
                self.generator_implementation = generator_dict['EmdNet']
                self.encoder_implementation = encoder_dict['Normal']
        elif 'WNet' in experiment_id:
            self.generator_implementation = generator_dict['WNet']
            self.encoder_implementation = encoder_dict['Normal']
            self.generator_residual_at_layer = generator_residual_at_layer
            self.generator_residual_blocks = generator_residual_blocks
            if 'DenseMixer' in experiment_id:
                self.other_info='DenseMixer'
            elif 'ResidualMixer' in experiment_id:
                self.other_info='ResidualMixer'
        elif 'Adobe' in experiment_id:
            self.generator_implementation = generator_dict['AdobeNet']
            self.encoder_implementation = encoder_dict['AdobeNet']
        elif 'ResMixer' in experiment_id:
            self.generator_implementation = generator_dict['ResMixerNet']
            self.encoder_implementation = encoder_dict['ResMixerNet']
            if 'SimpleMixer' in experiment_id:
                other_info_pos = experiment_id.find('SimpleMixer')
                possible_pos = other_info_pos-10
                if possible_pos<0:
                    possible_pos=0
                possible_extracted_info = experiment_id[possible_pos:other_info_pos+11]
                other_info_pos = possible_extracted_info.find('SimpleMixer')
                self.other_info=possible_extracted_info[other_info_pos-len(re.findall('\d+',possible_extracted_info)[0])-1:]
            elif 'DenseMixer' in experiment_id:
                other_info_pos = experiment_id.find('DenseMixer')
                possible_pos = other_info_pos - 10
                if possible_pos < 0:
                    possible_pos = 0
                possible_extracted_info = experiment_id[possible_pos:other_info_pos+10]
                other_info_pos = possible_extracted_info.find('DenseMixer')
                self.other_info = possible_extracted_info[other_info_pos - len(re.findall('\d+', possible_extracted_info)[0]) - 1:]



        self.print_info_seconds=30
        self.debug_mode = debug_mode


        self.img2img_width = img_width
        self.source_img_width = img_width


        self.content_data_dir = content_data_dir
        self.style_train_data_dir = style_train_data_dir
        self.fixed_style_reference_dir=fixed_style_reference_dir
        self.file_list_txt_content = file_list_txt_content
        self.file_list_txt_style_train = file_list_txt_style_train
        self.fixed_file_list_txt_style_reference=fixed_file_list_txt_style_reference
        self.input_output_img_filter_num = channels


        self.batch_size = batch_size

        self.true_fake_target_extractor_dir=true_fake_target_extractor_dir
        self.content_prototype_extractor_dir=content_prototype_extractor_dir
        self.style_reference_extractor_dir=style_reference_extractor_dir
        self.evaluating_generator_dir = evaluating_generator_dir

        self.generator_devices = generator_devices
        self.feature_extractor_device=feature_extractor_devices


        self.accuracy_k=[1,3,5,10,20,50]

        self.fixed_char_list_txt=fixed_char_list_txt



        print(self.print_separater)
        print(self.print_separater)
        print(self.print_separater)
        print("Evaluation on:" )
        print(self.experiment_id)
        print(self.print_separater)
        print(self.print_separater)
        print(self.print_separater)
        # if not self.debug_mode:
        #     raw_input("Press enter to continue")



        # init all the directories
        self.sess = None


    def find_norm_avg_var(self,var_list):
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

    def variable_dict(self,var_input,delete_name_from_character):
        var_output = {}
        for ii in var_input:
            prefix_pos = ii.name.find(delete_name_from_character)
            renamed = ii.name[prefix_pos + 1:]
            parafix_pos = renamed.find(':')
            renamed = renamed[0:parafix_pos]
            var_output.update({renamed: ii})
        return var_output

    def restore_model(self, saver, model_dir, model_name):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        corrected_ckpt_path = correct_ckpt_path(real_dir=model_dir,
                                                maybe_path=ckpt.model_checkpoint_path)
        if ckpt:
            saver.restore(self.sess, corrected_ckpt_path)
            print("ModelRestored:%s" % model_name)
            print("@%s" % model_dir)
            print(self.print_separater)
            return True
        else:
            print("fail to restore model %s" % model_dir)
            print(self.print_separater)
            return False

    def pixel_wise_difference_build(self, generated, true_style,
                                    random_content,random_style):
        def calculas(img1,img2):
            l1=tf.abs(img1-img2)
            squared_diff = l1 ** 2
            mse = tf.reduce_mean(squared_diff,axis=[1,2,3])
            mse = tf.sqrt(eps+mse)
            l1 = tf.reduce_mean(l1,axis=[1,2,3])



            threshold = (STANDARD_GRAYSCALE_THRESHOLD_VALUE/GRAYSCALE_AVG-1) * \
                        tf.ones(shape=generated.shape,
                                dtype=generated.dtype)
            img1_binary_condition = tf.greater(img1, threshold)
            img2_binary_condition = tf.greater(img2, threshold)
            img1_binary = tf.where(img1_binary_condition,
                                   tf.ones_like(img1),
                                   tf.zeros_like(img1))
            img2_binary = tf.where(img2_binary_condition,
                                   tf.ones_like(img2),
                                   tf.zeros_like(img2))
            binary_diff = tf.abs(img1_binary-img2_binary)
            pdar = tf.div(tf.cast(tf.reduce_sum(binary_diff, axis=[1,2,3]), dtype=tf.float32),
                          tf.cast(int(binary_diff.shape[1]) *
                                  int(binary_diff.shape[2]) *
                                  int(binary_diff.shape[3]),
                                  dtype=tf.float32))


            l1 = tf.reduce_sum(l1)
            mse = tf.reduce_sum(mse)
            pdar = tf.reduce_sum(pdar)

            return l1, mse, pdar


        l1_1, mse_1, pdar_1 = calculas(generated, true_style)
        l1_2, mse_2, pdar_2 = calculas(generated, random_content)
        l1_3, mse_3, pdar_3 = calculas(generated, random_style)

        result = tf.concat([tf.reshape(tf.reduce_mean(l1_1),shape=[1,1]),
                            tf.reshape(tf.reduce_mean(mse_1), shape=[1, 1]),
                            tf.reshape(tf.reduce_mean(pdar_2), shape=[1, 1]),
                            tf.reshape(tf.reduce_mean(l1_2), shape=[1, 1]),
                            tf.reshape(tf.reduce_mean(mse_2), shape=[1, 1]),
                            tf.reshape(tf.reduce_mean(pdar_2), shape=[1, 1]),
                            tf.reshape(tf.reduce_mean(l1_3), shape=[1, 1]),
                            tf.reshape(tf.reduce_mean(mse_3), shape=[1, 1]),
                            tf.reshape(tf.reduce_mean(pdar_3), shape=[1, 1])],
                           axis=1)



        return result




    def feature_extractor_build(self, data_provider, input_generated_img):

        def feature_linear_norm(feature):
            min_v= tf.reduce_min(feature)
            feature = feature - min_v
            max_v = tf.reduce_max(feature)
            feature = feature / max_v
            return feature+eps

        def calculate_high_level_feature_loss(feature1,feature2):
            for counter in range(len(feature1)):

                # mse calculas
                feature_diff = feature1[counter] - feature2[counter]
                if not feature_diff.shape.ndims==4:
                    feature_diff = tf.reshape(feature_diff,[int(feature_diff.shape[0]),int(feature_diff.shape[1]),1,1])
                squared_feature_diff = feature_diff**2
                mean_squared_feature_diff = tf.reduce_mean(squared_feature_diff,axis=[1,2,3])
                square_root_mean_squared_feature_diff = tf.sqrt(eps+mean_squared_feature_diff)
                square_root_mean_squared_feature_diff = tf.reshape(square_root_mean_squared_feature_diff,
                                                                   [int(square_root_mean_squared_feature_diff.shape[0]),1])
                this_mse_loss = tf.reduce_sum(square_root_mean_squared_feature_diff,axis=0)
                this_mse_loss = tf.reshape(this_mse_loss,shape=[1,1])

                if counter == 0:
                    feature_mse_diff = this_mse_loss
                else:
                    feature_mse_diff = tf.concat([feature_mse_diff,this_mse_loss], axis=1)



                # vn divergence calculas
                feature1_normed = feature_linear_norm(feature=feature1[counter])
                feature2_normed = feature_linear_norm(feature=feature2[counter])


                if not feature1_normed.shape.ndims==2:
                    vn_loss = tf.trace(tf.multiply(feature1_normed, tf.log(feature1_normed)) -
                                       tf.multiply(feature1_normed, tf.log(feature2_normed)) +
                                       - feature1_normed + feature2_normed + eps)
                    vn_loss = tf.reduce_mean(vn_loss,axis=1)
                    vn_loss = tf.reduce_sum(vn_loss)
                    vn_loss = tf.reshape(vn_loss, shape=[1,1])
                    if counter == 0:
                        feature_vn_diff = vn_loss
                    else:
                        feature_vn_diff = tf.concat([feature_vn_diff, vn_loss], axis=1)



            return feature_mse_diff, feature_vn_diff

        def build_feature_extractor(input_true_img,input_generated_img,
                                    reuse,
                                    extractor_usage,output_high_level_features):
            with tf.variable_scope(tf.get_variable_scope()):
                with tf.device(self.feature_extractor_device):


                    real_features,network_info = \
                        feature_extractor_network(image=input_true_img,
                                                  batch_size=self.batch_size,
                                                  device=self.feature_extractor_device,
                                                  reuse=reuse,
                                                  keep_prob=1,
                                                  initializer=self.initializer,
                                                  network_usage=extractor_usage,
                                                  output_high_level_features=output_high_level_features)
                    fake_features, _ = \
                        feature_extractor_network(image=input_generated_img,
                                                  batch_size=self.batch_size,
                                                  device=self.feature_extractor_device,
                                                  reuse=True,
                                                  keep_prob=1,
                                                  initializer=self.initializer,
                                                  network_usage=extractor_usage,
                                                  output_high_level_features=output_high_level_features)



            feature_loss_mse, feature_loss_vn = calculate_high_level_feature_loss(feature1=real_features,
                                                                                  feature2=fake_features)

            return feature_loss_mse, feature_loss_vn, network_info


        saver_list = list()

        # true / fake
        true_fake_feature_loss_mse, true_fake_feature_loss_vn, network_info = \
            build_feature_extractor(input_true_img=data_provider.train_iterator.output_tensor_list[0],
                                    input_generated_img=input_generated_img,
                                    extractor_usage='TrueFake_FeatureExtractor',
                                    output_high_level_features=[1, 2, 3, 4, 5, 6, 7],
                                    reuse=False)
        mse_difference = true_fake_feature_loss_mse
        vn_difference = true_fake_feature_loss_vn

        extr_vars_true_fake = [var for var in tf.trainable_variables() if 'TrueFake_FeatureExtractor' in var.name]
        extr_vars_true_fake = self.find_norm_avg_var(extr_vars_true_fake)
        extr_vars_true_fake = self.variable_dict(var_input=extr_vars_true_fake, delete_name_from_character='/')
        saver_extractor_true_fake = tf.train.Saver(max_to_keep=1, var_list=extr_vars_true_fake)
        print("TrueFakeExtractor @ %s with %s;" % (self.feature_extractor_device, network_info))


        # content prototype
        content_prototype = data_provider.train_iterator.output_tensor_list[1]
        selected_index = tf.random_uniform(shape=[], minval=0, maxval=int(content_prototype.shape[3]), dtype=tf.int64)
        selected_content_prototype = tf.expand_dims(content_prototype[:, :, :, selected_index], axis=3)

        content_prototype_feature_mse_loss_random, content_prototype_feature_vn_loss_random, network_info = \
            build_feature_extractor(input_true_img=selected_content_prototype,
                                    input_generated_img=input_generated_img,
                                    extractor_usage='ContentPrototype_FeatureExtractor',
                                    output_high_level_features=[1, 2, 3, 4, 5, 6, 7],
                                    reuse=False)
        content_prototype_feature_mse_loss_same, content_prototype_feature_vn_loss_same, network_info = \
            build_feature_extractor(input_true_img=data_provider.train_iterator.output_tensor_list[0],
                                    input_generated_img=input_generated_img,
                                    extractor_usage='ContentPrototype_FeatureExtractor',
                                    output_high_level_features=[1, 2, 3, 4, 5, 6, 7],
                                    reuse=True)

        mse_difference = tf.concat([mse_difference,content_prototype_feature_mse_loss_random],axis=0)
        mse_difference = tf.concat([mse_difference, content_prototype_feature_mse_loss_same], axis=0)
        vn_difference = tf.concat([vn_difference, content_prototype_feature_vn_loss_random], axis=0)
        vn_difference = tf.concat([vn_difference, content_prototype_feature_vn_loss_same], axis=0)



        extr_vars_content_prototype = [var for var in tf.trainable_variables() if
                                       'ContentPrototype_FeatureExtractor' in var.name]
        extr_vars_content_prototype = self.find_norm_avg_var(extr_vars_content_prototype)
        extr_vars_content_prototype = self.variable_dict(var_input=extr_vars_content_prototype,
                                                         delete_name_from_character='/')
        saver_extractor_content_prototype = tf.train.Saver(max_to_keep=1, var_list=extr_vars_content_prototype)
        print("ContentPrototypeExtractor @ %s with %s;" % (self.feature_extractor_device, network_info))


        # style reference
        style_reference = data_provider.train_iterator.output_tensor_list[2]
        selected_index = tf.random_uniform(shape=[], minval=0, maxval=int(style_reference.shape[3]), dtype=tf.int64)
        selected_style_reference = tf.expand_dims(style_reference[:, :, :, selected_index], axis=3)

        style_reference_feature_mse_loss_random, style_reference_feature_vn_loss_random, network_info = \
            build_feature_extractor(input_true_img=selected_style_reference,
                                    input_generated_img=input_generated_img,
                                    extractor_usage='StyleReference_FeatureExtractor',
                                    output_high_level_features=[1, 2, 3, 4, 5, 6, 7],
                                    reuse=False)
        style_reference_feature_mse_loss_same, style_reference_feature_vn_loss_same, network_info = \
            build_feature_extractor(input_true_img=data_provider.train_iterator.output_tensor_list[0],
                                    input_generated_img=input_generated_img,
                                    extractor_usage='StyleReference_FeatureExtractor',
                                    output_high_level_features=[1, 2, 3, 4, 5, 6, 7],
                                    reuse=True)

        mse_difference = tf.concat([mse_difference, style_reference_feature_mse_loss_random], axis=0)
        mse_difference = tf.concat([mse_difference, style_reference_feature_mse_loss_same], axis=0)
        vn_difference = tf.concat([vn_difference, style_reference_feature_vn_loss_random], axis=0)
        vn_difference = tf.concat([vn_difference, style_reference_feature_vn_loss_same], axis=0)

        extr_vars_style_reference = [var for var in tf.trainable_variables() if
                                     'StyleReference_FeatureExtractor' in var.name]
        extr_vars_style_reference = self.find_norm_avg_var(extr_vars_style_reference)
        extr_vars_style_reference = self.variable_dict(var_input=extr_vars_style_reference,
                                                       delete_name_from_character='/')
        saver_extractor_style_reference = tf.train.Saver(max_to_keep=1, var_list=extr_vars_style_reference)
        print("StyleReferenceExtractor @ %s with %s;" % (self.feature_extractor_device, network_info))

        saver_list.append(saver_extractor_true_fake)
        saver_list.append(saver_extractor_content_prototype)
        saver_list.append(saver_extractor_style_reference)


        return saver_list,mse_difference,vn_difference, \
               data_provider.train_iterator.output_tensor_list[0], \
               selected_content_prototype, selected_style_reference


    def generator_build(self,data_provider):

        name_prefix = 'generator'

        # network architechture
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(self.generator_devices):


                content_prototype_train = data_provider.train_iterator.output_tensor_list[1]
                style_reference_train_list = list()
                for ii in range(self.style_input_number):
                    style_reference_train = tf.expand_dims(data_provider.train_iterator.output_tensor_list[2][:,:,:,ii], axis=3)
                    style_reference_placeholder = tf.placeholder(dtype=style_reference_train.dtype,
                                                                 shape=style_reference_train.shape)
                    style_reference_train_list.append(style_reference_placeholder)


                # build the generator
                generated_target, _, _, network_info, _, _, _, _, _= \
                    self.generator_implementation(content_prototype=content_prototype_train,
                                                  style_reference=style_reference_train_list,
                                                  is_training=False,
                                                  batch_size=self.batch_size,
                                                  generator_device=self.generator_devices,
                                                  residual_at_layer=self.generator_residual_at_layer,
                                                  residual_block_num=self.generator_residual_blocks,
                                                  scope=name_prefix,
                                                  reuse=False,
                                                  initializer=self.initializer,
                                                  weight_decay=False,
                                                  weight_decay_rate=eps,
                                                  adain_use=self.adain_use,
                                                  adain_preparation_model=self.adain_preparation_model,
                                                  debug_mode=self.debug_mode,
                                                  other_info=self.other_info)

                curt_generator_handle = GeneratorHandle(generated_target=generated_target)
                setattr(self, "generator_handle", curt_generator_handle)


        gen_vars_train = [var for var in tf.trainable_variables() if 'generator' in var.name]
        gen_vars_save = self.find_norm_avg_var(gen_vars_train)
        saver_generator = tf.train.Saver(max_to_keep=1, var_list=gen_vars_save)


        print(
            "Generator @%s with %s;" % (self.generator_devices, network_info))
        return generated_target, style_reference_train_list, gen_vars_train, saver_generator










    def model_initialization(self,saver_generator, feature_extractor_saver_list):
        # initialization of all the variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())

        # restore of high_level feature extractor
        extr_restored = self.restore_model(saver=feature_extractor_saver_list[0],
                                           model_dir=self.true_fake_target_extractor_dir,
                                           model_name="TrueFakeExtractor")
        extr_restored = self.restore_model(saver=feature_extractor_saver_list[1],
                                           model_dir=self.content_prototype_extractor_dir,
                                           model_name="ContentPrototypeExtractor")
        extr_restored = self.restore_model(saver=feature_extractor_saver_list[2],
                                           model_dir=self.style_reference_extractor_dir,
                                           model_name="StyleReferenceExtractor")

        generator_restored = self.restore_model(saver=saver_generator,
                                                model_dir=self.evaluating_generator_dir,
                                                model_name="Generator")



    def evaluate_process(self):

        timer_start = time.time()

        if self.debug_mode == 1:
            self.print_info_seconds = 5

        with tf.Graph().as_default():

            # tensorflow parameters
            # DO NOT MODIFY!!!
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)


            # define the data set
            data_provider = DataProvider(batch_size=self.batch_size,
                                         info_print_interval=self.print_info_seconds,
                                         input_width=self.source_img_width,
                                         input_filters=self.input_output_img_filter_num,
                                         style_input_num=self.style_input_number,
                                         content_data_dir=self.content_data_dir,
                                         style_train_data_dir=self.style_train_data_dir,
                                         style_validation_data_dir=None,
                                         file_list_txt_content=self.file_list_txt_content,
                                         file_list_txt_style_train=self.file_list_txt_style_train,
                                         file_list_txt_style_validation=None,
                                         debug_mode=self.debug_mode,
                                         fixed_style_reference_dir=self.fixed_style_reference_dir,
                                         fixed_file_list_txt_style_reference=self.fixed_file_list_txt_style_reference,
                                         dataset_mode='Eval',
                                         fixed_char_list_txt=self.fixed_char_list_txt)

            self.involved_label0_list, self.involved_label1_list = data_provider.get_involved_label_list()
            self.content_input_num = data_provider.content_input_num
            self.fixed_style_reference_num = len(data_provider.train_iterator.fixed_style_reference_char_list)


            #######################################################################################
            #######################################################################################
            #                                model building
            #######################################################################################
            #######################################################################################

            # for generator building
            generated_batch,style_reference_train_list, \
            gen_vars_train, saver_generator \
                = self.generator_build(data_provider=data_provider)

            # for feature extractor
            feature_extractor_saver_list, mse_difference, vn_difference, \
            true_style,random_selected_content, random_selected_style= \
                self.feature_extractor_build(data_provider=data_provider,
                                             input_generated_img=generated_batch)

            # for pixel-wise difference
            pixel_diff = \
                self.pixel_wise_difference_build(generated=generated_batch,
                                                 true_style=true_style,
                                                 random_content = random_selected_content,
                                                 random_style = random_selected_style)



            # model initialization
            self.model_initialization(saver_generator=saver_generator,
                                      feature_extractor_saver_list=feature_extractor_saver_list)
            print(self.print_separater)
            print(self.print_separater)


        print("BatchSize:%d"  % (self.batch_size))
        print("EvaluationSize:%d, StyleLabel0_Vec:%d, StyleLabel1_Vec:%d" %
              (len(data_provider.train_iterator.true_style.data_list),
               len(self.involved_label0_list),
               len(self.involved_label1_list)))
        print("ContentLabel0_Vec:%d, ContentLabel1_Vec:%d" % (len(data_provider.content_label0_vec),len(data_provider.content_label1_vec)))
        print("InvolvedLabel0:%d, InvolvedLabel1:%d" % (len(self.involved_label0_list),
                                                        len(self.involved_label1_list)))
        total_eval_epochs=int(self.fixed_style_reference_num/self.style_input_number)
        print("StyleReferenceNum:%d, FixedStyleReferenceNum:%d, EvaluatlionEpoch:%d"
              % (self.style_input_number, self.fixed_style_reference_num,total_eval_epochs))
        print(self.print_separater)

        print("AdaIN_Mode:%s" % self.adain_mark)
        print(self.print_separater)
        print("Initialization completed, and evaluation started right now.")

        if self.debug_mode == 0:
            raw_input("Press enter to continue")
        print(self.print_separater)



        for ei in range(total_eval_epochs):
            current_fixed_style_reference_img_list = \
                data_provider.train_iterator.fixed_style_reference_image_list[ei*self.style_input_number:(ei+1)*self.style_input_number]
            current_fixed_style_reference_char_list= \
                data_provider.train_iterator.fixed_style_reference_char_list[ei*self.style_input_number:(ei+1)*self.style_input_number]
            current_fixed_style_reference_data_path_list = \
                data_provider.train_iterator.fixed_style_reference_data_path_list[ei * self.style_input_number:(ei + 1) * self.style_input_number]

            self.itrs_for_current_epoch = data_provider.compute_total_batch_num()
            data_provider.dataset_reinitialization(sess=self.sess, init_for_val=False,
                                                   info_interval=self.print_info_seconds)


            for iter in range(self.itrs_for_current_epoch):



                if not iter == self.itrs_for_current_epoch-1:
                    current_batch_label1 = data_provider.train_iterator.true_style.label1_list[iter * self.batch_size:
                                                                                               (iter + 1) * self.batch_size]
                else:
                    current_batch_label1 = data_provider.train_iterator.true_style.label1_list[iter * self.batch_size:]
                    if len(current_batch_label1) < self.batch_size:
                        added_num = self.batch_size-len(current_batch_label1)
                        current_batch_label1_added = data_provider.train_iterator.true_style.label1_list[0:added_num]
                        current_batch_label1.extend(current_batch_label1_added)


                current_style_reference_feed_list = list()
                current_style_reference_data_path_list = list()
                for ii in range(len(current_fixed_style_reference_img_list)):
                    this_single_style_reference_data_path_list = list()
                    for jj in range(len(current_batch_label1)):
                        jj_index = data_provider.train_iterator.label1_vec.index(current_batch_label1[jj])
                        current_label1_num = current_fixed_style_reference_img_list[ii][jj_index].shape[0]
                        random_index_selected = np.random.randint(low=0, high=current_label1_num)
                        selected_char_img = current_fixed_style_reference_img_list[ii][jj_index][random_index_selected,:,:,:]
                        selected_char_img_path = current_fixed_style_reference_data_path_list[ii][jj_index][random_index_selected]
                        selected_char_img = np.expand_dims(selected_char_img,axis=0)
                        if jj==0:
                            this_single_img_style_references = selected_char_img
                        else:
                            this_single_img_style_references = np.concatenate([this_single_img_style_references,selected_char_img],axis=0)
                        this_single_style_reference_data_path_list.append(selected_char_img_path)
                    current_style_reference_feed_list.append(this_single_img_style_references)
                    current_style_reference_data_path_list.append(this_single_style_reference_data_path_list)

                feed_dict={}
                for ii in range(len(style_reference_train_list)):
                    feed_dict.update({style_reference_train_list[ii]:current_style_reference_feed_list[ii]})

                # tmp0, tmp1, tmp2, tmp3 = \
                #     self.sess.run([generated_batch, true_style,random_selected_content, random_selected_style],
                #                   feed_dict=feed_dict)

                calculated_mse, calculated_vn, calculated_pixel, label0, label1 = \
                    self.sess.run([mse_difference, vn_difference,pixel_diff,
                                   data_provider.train_iterator.output_tensor_list[5],
                                   data_provider.train_iterator.output_tensor_list[6]],
                                  feed_dict=feed_dict)

                if iter == 0:
                    full_mse = calculated_mse
                    full_vn = calculated_vn
                    full_pixel = calculated_pixel
                elif iter == self.itrs_for_current_epoch - 1:
                    full_mse = full_mse + calculated_mse / self.batch_size * (self.batch_size - added_num)
                    full_vn = full_vn + calculated_vn / self.batch_size * (self.batch_size - added_num)
                    full_pixel = full_pixel + calculated_pixel / self.batch_size * (self.batch_size - added_num)
                else:
                    full_mse = full_mse + calculated_mse
                    full_vn = full_vn + calculated_vn
                    full_pixel = full_pixel + calculated_pixel


                if time.time()-timer_start>self.print_info_seconds:
                    timer_start=time.time()



                    current_epoch_str = "Epoch:%d/%d, Iteration:%d/%d for Style Reference Chars: " % (ei+1, total_eval_epochs,
                                                                                                      iter + 1, self.itrs_for_current_epoch)
                    # for char in current_fixed_style_reference_char_list:
                    #     current_epoch_str = current_epoch_str + char
                    # current_epoch_str=current_epoch_str+":"
                    print(current_epoch_str)
                    print("FeatureDiffMSE:")
                    print("------------------------------------------------------------------------")
                    for ii in range(full_mse.shape[0]):
                        for jj in range(full_mse.shape[1]):
                            if jj == 0:
                                if ii == 0:
                                    prefix = "TrueFake     :"
                                elif ii == 1:
                                    prefix = 'ContentRandom:'
                                elif ii == 2:
                                    prefix = 'ContentSame  :'
                                elif ii == 3:
                                    prefix = 'StyleRandom  :'
                                elif ii == 4:
                                    prefix = 'StyleSame    :'
                                print_str_line = prefix+"|%3.5f|" % (full_mse[ii][jj]/((iter+1)*self.batch_size))
                            else:
                                print_str_line = print_str_line + "%3.5f|" % (full_mse[ii][jj]/((iter+1)*self.batch_size))
                        print(print_str_line)
                    print("------------------------------------------------------------------------")
                    print("FeatureDiffVN:")
                    print("------------------------------------------------------------------------")
                    for ii in range(full_vn.shape[0]):
                        for jj in range(full_vn.shape[1]):
                            if jj == 0:
                                if ii == 0:
                                    prefix = "TrueFake     :"
                                elif ii == 1:
                                    prefix = 'ContentRandom:'
                                elif ii == 2:
                                    prefix = 'ContentSame  :'
                                elif ii == 3:
                                    prefix = 'StyleRandom  :'
                                elif ii == 4:
                                    prefix = 'StyleSame    :'
                                print_str_line = prefix+"|%3.5f|" % (full_vn[ii][jj]/((iter+1)*self.batch_size))
                            else:
                                print_str_line = print_str_line + "%3.5f|" % (full_vn[ii][jj]/((iter+1)*self.batch_size))
                        print(print_str_line)
                    print("------------------------------------------------------------------------")
                    print("PixelDiff:")
                    print("----------------------------------------------------------------------------------------------------------------")
                    print("||   L1-Sm   |  MSE-Sm   |  PDAR-Sm  | L1-RdmCtn | MSE-RdmCtn|PDAR-RdmCtn| L1-RdmSty | MSE-RdmSty|PDAR-RdmSty||")
                    print("----------------------------------------------------------------------------------------------------------------")
                    print_str_line = '||'
                    for ii in range(full_pixel.shape[1]):
                        if ii == full_pixel.shape[1]-1:
                            print_str_line = print_str_line + "  %.5f  ||" % (full_pixel[0][ii]/((iter+1)*self.batch_size))
                        else:
                            print_str_line = print_str_line + "  %.5f  |" % (full_pixel[0][ii]/((iter+1)*self.batch_size))
                    print(print_str_line)
                    print("----------------------------------------------------------------------------------------------------------------")
                    print(self.print_separater)

            full_mse = full_mse / len(data_provider.train_iterator.true_style.data_list)
            full_vn = full_vn / len(data_provider.train_iterator.true_style.data_list)
            full_pixel = full_pixel / len(data_provider.train_iterator.true_style.data_list)


            print(self.print_separater)
            print(self.print_separater)
            current_epoch_str = "Epoch:%d/%d Completed for Style Reference Chars: " % (ei + 1, total_eval_epochs)
            # for char in current_fixed_style_reference_char_list:
            #     current_epoch_str = current_epoch_str + char
            # current_epoch_str = current_epoch_str + ":"
            print(current_epoch_str)
            print("FeatureDiffMSE:")
            print("------------------------------------------------------------------------")
            for ii in range(full_mse.shape[0]):
                for jj in range(full_mse.shape[1]):
                    if jj == 0:
                        if ii ==0:
                            prefix = "TrueFake     :"
                        elif ii ==1:
                            prefix = 'ContentRandom:'
                        elif ii ==2:
                            prefix = 'ContentSame  :'
                        elif ii == 3:
                            prefix = 'StyleRandom  :'
                        elif ii == 4:
                            prefix = 'StyleSame    :'
                        print_str_line = prefix+"|%3.5f|" % (full_mse[ii][jj])
                    else:
                        print_str_line = print_str_line + "%3.5f|" % (full_mse[ii][jj])
                print(print_str_line)
            print("------------------------------------------------------------------------")
            print("FeatureDiffVN:")
            print("------------------------------------------------------------------------")
            for ii in range(full_vn.shape[0]):
                for jj in range(full_vn.shape[1]):
                    if jj == 0:
                        if ii == 0:
                            prefix = "TrueFake     :"
                        elif ii == 1:
                            prefix = 'ContentRandom:'
                        elif ii == 2:
                            prefix = 'ContentSame  :'
                        elif ii == 3:
                            prefix = 'StyleRandom  :'
                        elif ii == 4:
                            prefix = 'StyleSame    :'
                        print_str_line = prefix+"|%3.5f|" % (full_vn[ii][jj])
                    else:
                        print_str_line = print_str_line + "%3.5f|" % (full_vn[ii][jj])
                print(print_str_line)
            print("------------------------------------------------------------------------")
            print("PixelDiff:")
            print("----------------------------------------------------------------------------------------------------------------")
            print("||   L1-Sm   |  MSE-Sm   |  PDAR-Sm  | L1-RdmCtn | MSE-RdmCtn|PDAR-RdmCtn| L1-RdmSty | MSE-RdmSty|PDAR-RdmSty||")
            print("----------------------------------------------------------------------------------------------------------------")
            print_str_line = '||'
            for ii in range(full_pixel.shape[1]):
                if ii == full_pixel.shape[1] - 1:
                    print_str_line = print_str_line + "  %.5f  ||" % (full_pixel[0][ii])
                else:
                    print_str_line = print_str_line + "  %.5f  |" % (full_pixel[0][ii])
            print(print_str_line)
            print("----------------------------------------------------------------------------------------------------------------")
            print(self.print_separater)
            print(self.print_separater)

            if ei==0:
                mse = full_mse
                vn = full_vn
                pixel = full_pixel
            else:
                mse += full_mse
                vn += full_vn
                pixel += full_pixel
        mse = mse / total_eval_epochs
        vn = vn / total_eval_epochs
        pixel = pixel / total_eval_epochs

        evaluation_resule_save_dir = os.path.join(self.evaluation_resule_save_dir,self.experiment_id)
        if not os.path.exists(evaluation_resule_save_dir):
            os.makedirs(evaluation_resule_save_dir)
        np.savetxt(os.path.join(evaluation_resule_save_dir,'MSE.csv'), mse, delimiter=',')
        np.savetxt(os.path.join(evaluation_resule_save_dir, 'VN.csv'), vn, delimiter=',')
        np.savetxt(os.path.join(evaluation_resule_save_dir, 'PIXEL.csv'), pixel, delimiter=',')
