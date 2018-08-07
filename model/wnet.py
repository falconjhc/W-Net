# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
GRAYSCALE_AVG = 127.5

import matplotlib


import sys
sys.path.append('..')

import copy as cp
import random as rd

import tensorflow as tf
import numpy as np
import random as rnd
import scipy.misc as misc
import os
import shutil
import time
from collections import namedtuple
from dataset.dataset import DataProvider

from utilities.utils import scale_back_for_img, scale_back_for_dif, merge, correct_ckpt_path
from utilities.utils import image_show
from model.gan_networks import discriminator_mdy_5_convs
from model.gan_networks import discriminator_mdy_6_convs
from model.gan_networks import discriminator_mdy_6_convs_tower_version1


from model.gan_networks import vgg_16_net as feature_extractor_network


from model.gan_networks import generator_framework as generator_implementation
from model.gan_networks import generator_inferring
from model.gan_networks import encoder_framework as encoder_implementation


import math

import utilities.infer_implementations as inf_tools




# Auxiliary wrapper classes
# Used to save handles(important nodes in computation graph) for later evaluation
SummaryHandle = namedtuple("SummaryHandle", ["d_merged", "g_merged",
                                             "check_validate_image_summary", "check_train_image_summary",
                                             "check_validate_image", "check_train_image",
                                             "learning_rate",
                                             "trn_real_summaries","val_real_summaries",
                                             "trn_fake_summaries","val_fake_summaries"])

EvalHandle = namedtuple("EvalHandle",["inferred_generated_images",
                                      "inferred_categorical_logits"])

GeneratorHandle = namedtuple("Generator",
                             ["content_prototype","true_style","style_reference_list",
                              "true_label0","true_label1",
                              "generated_target_train","generated_target_infer"])

DiscriminatorHandle = namedtuple("Discriminator",
                                 ["discriminator_true_label",
                                 "current_critic_logit_penalty",
                                  "content_prototype","true_style","style_reference",
                                  "infer_content_prototype","discriminator_infer_true_or_generated_style","infer_style_reference",
                                  "infer_C_logits","infer_Cr_logits"])

FeatureExtractorHandle = namedtuple("FeatureExtractor",
                                    ["infer_input_img","selected_content_prototype","selected_style_reference"])



discriminator_dict = {"DisMdy5conv": discriminator_mdy_5_convs,
                      "DisMdy6conv": discriminator_mdy_6_convs,
                      "DisMdy6conv-TowerVersion1": discriminator_mdy_6_convs_tower_version1}

eps = 1e-9

class WNet(object):

    # constructor
    def __init__(self,
                 debug_mode=-1,
                 print_info_seconds=-1,
                 train_data_augment=-1,
                 init_training_epochs=-1,
                 final_training_epochs=-1,

                 experiment_dir='/tmp/',
                 log_dir='/tmp/',
                 experiment_id='0',
                 content_data_dir='/tmp/',
                 style_train_data_dir='/tmp/',
                 style_validation_data_dir='/tmp/',
                 training_from_model=None,
                 file_list_txt_content=None,
                 file_list_txt_style_train=None,
                 file_list_txt_style_validation=None,
                 channels=-1,
                 epoch=-1,

                 optimization_method='adam',

                 batch_size=8, img_width=256,
                 lr=0.001, final_learning_rate_pctg=0.2,


                 L1_Penalty=100,
                 Lconst_content_Penalty=15, Lconst_style_Penalty=15,
                 Discriminative_Penalty=1,
                 Discriminator_Categorical_Penalty=1,Generator_Categorical_Penalty=0.1,
                 Discriminator_Gradient_Penalty=1,

                 Feature_Penalty_True_Fake_Target=5,
                 Feature_Penalty_Style_Reference=5,
                 Feature_Penalty_Content_Prototype=5,

                 generator_weight_decay_penalty = 0.001,
                 discriminator_weight_decay_penalty = 0.004,

                 resume_training=0,

                 generator_devices='/device:CPU:0',
                 discriminator_devices='/device:CPU:0',
                 feature_extractor_devices='/device:CPU:0',

                 generator_residual_at_layer=3,
                 generator_residual_blocks=5,
                 discriminator='DisMdy4conv',
                 true_fake_target_extractor_dir='/tmp/',
                 content_prototype_extractor_dir='/tmp/',
                 style_reference_extractor_dir='/tmp/',

                 ## for infer only
                 model_dir='./', infer_dir='./',

                 # for styleadd infer only
                 targeted_chars_txt='./',
                 target_file_path='-1',
                 save_mode='-1',
                 style_add_target_model_dir='./',
                 style_add_target_file_list='./tmp.txt',
                 target_label1_selection=[0000],
                 style_input_number=5,
                 source_font='./tmp.txt',
                 infer_mode=-1,

                 softmax_temperature=1

                 ):

        self.initializer = 'XavierInit'

        self.print_info_seconds=print_info_seconds
        self.discriminator_initialization_iters=25
        self.init_training_epochs=init_training_epochs
        self.final_training_epochs=final_training_epochs
        self.final_training_epochs=final_training_epochs
        self.model_save_epochs=3
        self.debug_mode = debug_mode
        self.experiment_dir = experiment_dir
        self.log_dir=log_dir
        self.experiment_id = experiment_id
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoint")
        self.training_from_model = training_from_model
        self.epoch=epoch

        self.inf_data_dir = os.path.join(self.experiment_dir, "infs")
        self.img2img_width = img_width
        self.source_img_width = img_width


        self.content_data_dir = content_data_dir
        self.style_train_data_dir = style_train_data_dir
        self.style_validation_data_dir = style_validation_data_dir
        self.file_list_txt_content = file_list_txt_content
        self.file_list_txt_style_train = file_list_txt_style_train
        self.file_list_txt_style_validation = file_list_txt_style_validation
        self.input_output_img_filter_num = channels


        self.optimization_method = optimization_method
        self.batch_size = batch_size
        self.final_learning_rate_pctg = final_learning_rate_pctg


        self.resume_training = resume_training

        self.train_data_augment = (train_data_augment==1)


        self.Discriminative_Penalty = Discriminative_Penalty + eps
        self.Discriminator_Gradient_Penalty = Discriminator_Gradient_Penalty + eps
        self.L1_Penalty = L1_Penalty + eps
        self.Feature_Penalty_True_Fake_Target = Feature_Penalty_True_Fake_Target + eps
        self.Feature_Penalty_Style_Reference = Feature_Penalty_Style_Reference + eps
        self.Feature_Penalty_Content_Prototype = Feature_Penalty_Content_Prototype  + eps
        self.Lconst_content_Penalty = Lconst_content_Penalty + eps
        self.Lconst_style_Penalty = Lconst_style_Penalty + eps
        self.lr = lr


        self.Discriminator_Categorical_Penalty = Discriminator_Categorical_Penalty + eps
        self.Generator_Categorical_Penalty = Generator_Categorical_Penalty + eps
        self.generator_weight_decay_penalty = generator_weight_decay_penalty + eps
        self.discriminator_weight_decay_penalty = discriminator_weight_decay_penalty + eps

        if self.generator_weight_decay_penalty > 10 * eps:
            self.weight_decay_generator = True
        else:
            self.weight_decay_generator = False
        if self.discriminator_weight_decay_penalty > 10 * eps:
            self.weight_decay_discriminator = True
        else:
            self.weight_decay_discriminator = False

        self.generator_devices = generator_devices
        self.discriminator_devices = discriminator_devices
        self.feature_extractor_device=feature_extractor_devices

        self.generator_residual_at_layer = generator_residual_at_layer
        self.generator_residual_blocks = generator_residual_blocks
        self.discriminator = discriminator



        self.discriminator_implementation = discriminator_dict[self.discriminator]

        if not self.Feature_Penalty_True_Fake_Target > 10*eps:
            self.extractor_true_fake_enabled = False
            self.true_fake_target_extractor_dir = 'None'
        else:
            self.extractor_true_fake_enabled = True
            self.true_fake_target_extractor_dir = true_fake_target_extractor_dir



        if not self.Feature_Penalty_Style_Reference > 10*eps:
            self.extractor_style_reference_enabled = False
            self.style_reference_extractor_dir = 'None'
        else:
            self.extractor_style_reference_enabled = True
            self.style_reference_extractor_dir = style_reference_extractor_dir

        if not self.Feature_Penalty_Content_Prototype > 10*eps:
            self.extractor_content_prototype_enabled = False
            self.content_prototype_extractor_dir = 'None'
        else:
            self.extractor_content_prototype_enabled = True
            self.content_prototype_extractor_dir = content_prototype_extractor_dir

        self.accuracy_k=[1,3,5,10,20,50]

        # properties for inferring
        self.model_dir=model_dir
        self.infer_dir=infer_dir
        if os.path.exists(self.infer_dir) and not (self.infer_dir == './'):
            shutil.rmtree(self.infer_dir)
        if not self.infer_dir == './':
            os.makedirs(self.infer_dir)

        # for styleadd infer only
        self.targeted_chars_txt=targeted_chars_txt
        self.target_file_path=target_file_path
        self.save_mode=save_mode
        self.style_add_target_model_dir=style_add_target_model_dir
        self.style_add_target_file_list=style_add_target_file_list
        self.target_label1_selection=target_label1_selection
        self.style_input_number=style_input_number
        self.source_font=source_font

        self.softmax_temperature=softmax_temperature


        # init all the directories
        self.sess = None
        self.print_separater = "#################################################################"

    def find_bn_avg_var(self,var_list):
        var_list_new = list()
        for ii in var_list:
            var_list_new.append(ii)

        all_vars = tf.global_variables()
        bn_var_list = [var for var in var_list if 'bn' in var.name]
        output_avg_var = list()
        for bn_var in bn_var_list:
            if 'gamma' in bn_var.name:
                continue
            bn_var_name = bn_var.name
            variance_name = bn_var_name.replace('beta', 'moving_variance')
            average_name = bn_var_name.replace('beta', 'moving_mean')
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

    def get_model_id_and_dir_for_train(self):
        encoder_decoder_layer_num = int(np.floor(math.log(self.img2img_width) / math.log(2)))
        model_id = "Exp%s_GenEncDec%d-Res%d@Lyr%d_%s" % (self.experiment_id,
                                                         encoder_decoder_layer_num,
                                                         self.generator_residual_blocks,
                                                         self.generator_residual_at_layer,
                                                         self.discriminator)

        model_ckpt_dir = os.path.join(self.checkpoint_dir, model_id)
        model_log_dir = os.path.join(self.log_dir, model_id)
        model_infer_dir = os.path.join(self.inf_data_dir, model_id)
        return model_id, model_ckpt_dir, model_log_dir, model_infer_dir

    def checkpoint(self, saver, model_dir,global_step):
        model_name = "img2img.model"
        step = global_step.eval(session=self.sess)
        if step==0:
            step=1
        print(os.path.join(model_dir, model_name))
        saver.save(self.sess, os.path.join(model_dir, model_name), global_step=int(step))

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

    def generate_fake_samples(self,prototype,reference):

        def feed_dict_for_g_tmp(reference,prototype):
            output_dict = {}
            generator_handle = getattr(self, "generator_handle")

            output_dict.update({generator_handle.content_prototype:prototype})
            for ii in range(self.style_input_number):
                output_dict.update({generator_handle.style_reference_list[ii]:
                                        np.reshape(reference[:, :, :, ii],
                                                   [self.batch_size,
                                                    self.img2img_width,
                                                    self.img2img_width,1])})
            return output_dict

        evalHandle = getattr(self, "eval_handle")
        fake_images = self.sess.run(evalHandle.inferred_generated_images,
                                    feed_dict=feed_dict_for_g_tmp(reference=reference,
                                                                  prototype=prototype))
        return fake_images


    def evaluate_samples(self,source,target,input_style):

        def feed_dict_for_d_tmp():
            output_dict = {}
            discriminator_handle = getattr(self, "discriminator_handle")
            rnd_index = rnd.sample(range(self.style_input_number),1)[0]

            output_dict.update({discriminator_handle.infer_source:source})
            output_dict.update({discriminator_handle.infer_true_or_generated_target:target})
            output_dict.update({discriminator_handle.infer_input_target: np.reshape(input_style[:,:,:,rnd_index],
                                                                                    [self.batch_size,
                                                                                     self.img2img_width,
                                                                                     self.img2img_width,
                                                                                     1])})
            return output_dict

        evalHandle = getattr(self,'eval_handle')
        c_logits = self.sess.run(evalHandle.inferred_categorical_logits,
                                 feed_dict=feed_dict_for_d_tmp())

        return c_logits


    def validate_model(self,
                       batch_label0_one_hot,
                       batch_label1_one_hot,
                       batch_packs,
                       train_mark,
                       summary_writer, global_step,
                       training_img_list=list()):

        def feed_dict_tmp(current_specifying_target):

            output_dict = {}
            discriminator_handle = getattr(self, "discriminator_handle")

            generator_handle = getattr(self, "generator_handle")

            output_dict.update({generator_handle.content_prototype:content_prototypes})
            for ii in range(self.style_input_number):
                output_dict.update({generator_handle.style_reference_list[ii]:np.reshape(style_references[:,:,:,ii],
                                                                                      [self.batch_size,
                                                                                       self.img2img_width,
                                                                                       self.img2img_width,1])})
            output_dict.update({generator_handle.true_label0: batch_label0_one_hot})
            output_dict.update({generator_handle.true_label1: batch_label1_one_hot})


            rnd_index_style = rnd.sample(range(self.style_input_number), 1)[0]
            rnd_index_content = rnd.sample(range(self.content_input_num), 1)[0]

            output_dict.update({discriminator_handle.infer_content_prototype:
                                    np.reshape(content_prototypes[:,:,:,rnd_index_content],
                                               [self.batch_size,
                                                self.img2img_width,
                                                self.img2img_width, 1])})
            output_dict.update({discriminator_handle.infer_style_reference:
                                    np.reshape(style_references[:,:,:,rnd_index_style],
                                               [self.batch_size,
                                                self.img2img_width,
                                                self.img2img_width, 1])})
            output_dict.update({discriminator_handle.discriminator_infer_true_or_generated_style: current_specifying_target})
            output_dict.update({discriminator_handle.discriminator_true_label: batch_label1_one_hot})

            feature_extractor_handle = getattr(self, "feature_extractor_handle")
            output_dict.update({feature_extractor_handle.infer_input_img: current_specifying_target})

            return output_dict


        summary_handle = getattr(self,"summary_handle")
        feature_extractor_handle = getattr(self,"feature_extractor_handle")
        if train_mark:
            merged_real_summaries = summary_handle.trn_real_summaries
            merged_fake_summaries = summary_handle.trn_fake_summaries
            check_image = summary_handle.check_train_image_summary
            check_image_input = summary_handle.check_train_image

        else:
            merged_real_summaries = summary_handle.val_real_summaries
            merged_fake_summaries = summary_handle.val_fake_summaries
            check_image = summary_handle.check_validate_image_summary
            check_image_input = summary_handle.check_validate_image

        channels = self.input_output_img_filter_num
        true_style = batch_packs[:, :, :, 0:channels]
        content_prototypes = batch_packs[:, :, :, channels:channels + self.content_input_num * channels]
        style_references = batch_packs[:, :, :, 1 + self.content_input_num * channels:]


        generated_batch = self.generate_fake_samples(prototype=content_prototypes,
                                                     reference=style_references)


        summary_real_output = self.sess.run(merged_real_summaries,
                                            feed_dict=feed_dict_tmp(current_specifying_target=true_style))
        summary_fake_output = self.sess.run(merged_fake_summaries,
                                            feed_dict=feed_dict_tmp(current_specifying_target=generated_batch))


        generated_style = scale_back_for_img(images=generated_batch)
        content_prototypes = scale_back_for_img(images=content_prototypes)
        style_references = scale_back_for_img(images=style_references)
        true_style = scale_back_for_img(images=true_style)

        if train_mark:
            generated_train_style = scale_back_for_img(images=training_img_list[0])
            extracted_training_content = scale_back_for_img(images=training_img_list[1])
            extracted_training_style = scale_back_for_img(images=training_img_list[2])
        diff_between_generated_and_true = scale_back_for_dif(generated_style - true_style)

        generated_style = merge(generated_style, [self.batch_size, 1])
        new_content_prototype = np.zeros(shape=[self.batch_size * self.img2img_width,
                                                self.img2img_width, 3,
                                                self.content_input_num],
                                         dtype=np.float32)
        for ii in range(self.content_input_num):
            new_content_prototype[:, :, :, ii] = merge(np.reshape(content_prototypes[:, :, :, ii],
                                                                  [content_prototypes[:, :, :, ii].shape[0],
                                                                   content_prototypes[:, :, :, ii].shape[1],
                                                                   content_prototypes[:, :, :, ii].shape[2], 1]),
                                                       [self.batch_size, 1])

        new_style_reference = np.zeros(shape=[self.batch_size*self.img2img_width,self.img2img_width,3,self.style_input_number],
                                       dtype=np.float32)
        for ii in range(self.style_input_number):
            new_style_reference[:,:,:,ii] = merge(np.reshape(style_references[:,:,:,ii],
                                                             [style_references[:,:,:,ii].shape[0],
                                                              style_references[:, :, :, ii].shape[1],
                                                              style_references[:, :, :, ii].shape[2],1]),
                                                  [self.batch_size, 1])
        true_style = merge(true_style, [self.batch_size, 1])
        diff_between_generated_and_true = merge(diff_between_generated_and_true, [self.batch_size, 1])

        current_display_content_prototype_indices = rnd.sample(range(self.content_input_num),
                                                               self.display_content_reference_num)

        for ii in range(len(current_display_content_prototype_indices)):
            idx = current_display_content_prototype_indices[ii]
            curt_disp = new_content_prototype[:, :, :, idx]
            if ii == 0:
                content_disp = curt_disp
            else:
                content_disp = np.concatenate([content_disp,curt_disp],axis=1)


        current_display_style_reference_indices = rnd.sample(range(self.style_input_number),
                                                             self.display_style_reference_num)
        for ii in range(len(current_display_style_reference_indices)):
            idx = current_display_style_reference_indices[ii]
            curt_disp = new_style_reference[:, :, :, idx]
            if ii == 0:
                style_disp = curt_disp
            else:
                style_disp = np.concatenate([style_disp, curt_disp], axis=1)





        if train_mark:
            generated_train_style = merge(generated_train_style,[self.batch_size,1])
            extracted_training_content = merge(extracted_training_content, [self.batch_size, 1])
            extracted_training_style = merge(extracted_training_style, [self.batch_size, 1])
            diff_between_generated_train_and_true = scale_back_for_dif(generated_train_style - true_style)

            merged_disp = np.concatenate([content_disp,
                                          generated_style,
                                          diff_between_generated_and_true,
                                          generated_train_style,
                                          diff_between_generated_train_and_true,
                                          true_style,
                                          style_disp], axis=1)
            if self.extractor_content_prototype_enabled:
                merged_disp = np.concatenate([extracted_training_content, merged_disp], axis=1)
            if self.extractor_style_reference_enabled:
                merged_disp = np.concatenate([merged_disp, extracted_training_style], axis=1)

        else:
            merged_disp = np.concatenate([content_disp,
                                          generated_style,
                                          diff_between_generated_and_true,
                                          true_style,
                                          style_disp], axis=1)


        summray_img = self.sess.run(check_image,
                                    feed_dict={check_image_input:
                                                   np.reshape(merged_disp, (1, merged_disp.shape[0],
                                                                            merged_disp.shape[1],
                                                                            merged_disp.shape[2]))})
        summary_writer.add_summary(summary_real_output, global_step.eval(session=self.sess))
        summary_writer.add_summary(summary_fake_output, global_step.eval(session=self.sess))
        summary_writer.add_summary(summray_img, global_step.eval(session=self.sess))









    def summary_finalization(self,
                             g_loss_summary,
                             d_loss_summary,
                             trn_real_summaries, val_real_summaries,
                             trn_fake_summaries, val_fake_summaries,
                             learning_rate):
        train_img_num = 5
        if self.extractor_content_prototype_enabled:
            train_img_num+=1
        if self.extractor_style_reference_enabled:
            train_img_num+=1
        check_train_image = tf.placeholder(tf.float32, [1, self.batch_size * self.img2img_width,
                                                        self.img2img_width * (self.display_content_reference_num+self.display_style_reference_num+train_img_num),
                                                        3])

        check_validate_image = tf.placeholder(tf.float32, [1, self.batch_size * self.img2img_width,
                                                           self.img2img_width * (self.display_content_reference_num+self.display_style_reference_num+3),
                                                           3])



        check_train_image_summary = tf.summary.image('TrnImg', check_train_image)
        check_validate_image_summary = tf.summary.image('ValImg', check_validate_image)


        learning_rate_summary = tf.summary.scalar('LearningRate', learning_rate)

        summary_handle = SummaryHandle(d_merged=d_loss_summary,
                                       g_merged=g_loss_summary,
                                       check_validate_image_summary=check_validate_image_summary,
                                       check_train_image_summary=check_train_image_summary,
                                       check_validate_image=check_validate_image,
                                       check_train_image=check_train_image,
                                       learning_rate=learning_rate_summary,
                                       trn_real_summaries=trn_real_summaries,
                                       val_real_summaries=val_real_summaries,
                                       trn_fake_summaries=trn_fake_summaries,
                                       val_fake_summaries=val_fake_summaries,
                                       )
        setattr(self, "summary_handle", summary_handle)


    def styleadd_infer_procedures_mode0(self):

        charset_level1, character_label_level1 = \
            inf_tools.get_chars_set_from_level1_2(path='../FontAndChars/GB2312_Level_1.txt',
                                                  level=1)
        charset_level2, character_label_level2 = \
            inf_tools.get_chars_set_from_level1_2(path='../FontAndChars/GB2312_Level_2.txt',
                                                  level=2)

        targeted_charset, targeted_label0_list = \
            inf_tools.get_chars_set_from_searching(path=self.targeted_chars_txt,
                                                   level1_charlist=charset_level1,
                                                   level2_charlist=charset_level2,
                                                   level1_labellist=character_label_level1,
                                                   level2_labellist=character_label_level2)

        source_imgs = inf_tools.find_source_char_img(charset=targeted_charset,
                                                     fontpath=self.source_font,
                                                     img_width=self.source_img_width,
                                                     img_filters=self.input_output_img_filter_num,
                                                     batch_size=self.batch_size)

        output_paper_shape = self.save_mode.split(':')
        output_paper_rows = int(output_paper_shape[0])
        output_paper_cols = int(output_paper_shape[1])
        if output_paper_rows * output_paper_cols < len(targeted_charset):
            print('Incorrect Paper Size !@!~!@~!~@~!@~')
            return



        with tf.Graph().as_default():
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            # build embedder
            with tf.device(self.generator_devices):




                input_batch = tf.placeholder(tf.float32,
                                             [self.batch_size,
                                              self.source_img_width,
                                              self.source_img_width,
                                              self.input_output_img_filter_num],
                                             name='input_batch')
                _, _, encoded_list_obtained, residual_list_obtained, _ = \
                    encoder_implementation(images=input_batch,
                                           is_training=False,
                                           encoder_device=self.generator_devices,
                                           residual_at_layer=self.generator_residual_at_layer,
                                           residual_connection_mode='Single',
                                           scope='generator/target_encoder',
                                           reuse=False,
                                           initializer='XavierInit',
                                           weight_decay=False,
                                           weight_decay_rate=0,
                                           final_layer_logit_length=-1)




            with tf.device(self.generator_devices):
                source = tf.placeholder(tf.float32,
                                        [self.batch_size,
                                         self.source_img_width,
                                         self.source_img_width,
                                         self.input_output_img_filter_num],
                                        name='geneartor_source_image')

                encoded_list_input = list()
                for ii in range(len(encoded_list_obtained)):
                    tmp = tf.placeholder(shape=encoded_list_obtained[ii].shape,
                                         dtype=encoded_list_obtained[ii].dtype)
                    encoded_list_input.append(tmp)
                residual_list_input = list()
                for ii in range(len(residual_list_obtained)):
                    tmp = tf.placeholder(shape=residual_list_obtained[ii].shape,
                                         dtype=residual_list_obtained[ii].dtype)
                    residual_list_input.append(tmp)


                generated_infer = generator_inferring(source=source,
                                                      batch_size=self.batch_size,
                                                      generator_device=self.generator_devices,
                                                      residual_at_layer=self.generator_residual_at_layer,
                                                      residual_block_num=self.generator_residual_blocks,
                                                      target_encoded_list=encoded_list_input,
                                                      residual_input_list=residual_list_input)

                gen_vars_save = self.find_bn_avg_var([var for var in tf.trainable_variables() if 'generator' in var.name])



            # model load
            saver_generator = tf.train.Saver(max_to_keep=1, var_list=gen_vars_save)
            generator_restored = self.restore_model(saver=saver_generator,
                                                    model_dir=self.model_dir,
                                                    model_name='Generator')

            if not generator_restored:
                return
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)





        # implementation
        save_dir_true = os.path.join(self.infer_dir, 'TrueChars')
        save_dir_generated = os.path.join(self.infer_dir,'GeneratedChars')
        if os.path.exists(save_dir_true):
            shutil.rmtree(save_dir_true)
        if os.path.exists(save_dir_generated):
            shutil.rmtree(save_dir_generated)
        os.makedirs(save_dir_true)
        os.makedirs(save_dir_generated)

        for target_label1 in self.target_label1_selection:

            transfer_targets = inf_tools.find_transfer_targets(data_dir=self.style_add_target_model_dir,
                                                               txt_path=self.style_add_target_file_list,
                                                               selected_label1=int(target_label1),
                                                               style_input_number=self.style_input_number,
                                                               img_width=self.source_img_width,
                                                               filter_num=self.input_output_img_filter_num,
                                                               batch_size=self.batch_size)

            true_targets = inf_tools.find_true_targets(data_dir=self.style_add_target_model_dir,
                                                       txt_path=self.style_add_target_file_list,
                                                       selected_label1=int(target_label1),
                                                       selected_label0_list=targeted_label0_list,
                                                       charset=targeted_charset, font=self.source_font,
                                                       img_width=self.source_img_width,
                                                       filter_num=self.input_output_img_filter_num,
                                                       batch_size=self.batch_size)

            # calculate ebdd_vec
            style_calculate_iters = transfer_targets.shape[0]/self.batch_size
            encoded_list_input_calculated = list()
            for ii in range(len(encoded_list_obtained)):
                tmp = np.zeros(shape=[style_calculate_iters*self.batch_size,
                                      int(encoded_list_obtained[ii].shape[1]),
                                      int(encoded_list_obtained[ii].shape[2]),
                                      int(encoded_list_obtained[ii].shape[3])],
                               dtype=np.float32)
                encoded_list_input_calculated.append(tmp)
            residual_list_input_calculated = list()
            for ii in range(len(residual_list_obtained)):
                tmp = np.zeros(shape=[style_calculate_iters * self.batch_size,
                                      int(residual_list_obtained[ii].shape[1]),
                                      int(residual_list_obtained[ii].shape[2]),
                                      int(residual_list_obtained[ii].shape[3])],
                               dtype=np.float32)
                residual_list_input_calculated.append(tmp)

            for iter in range(style_calculate_iters):
                this_batch_encoded_list, this_batch_residual_list\
                    =self.sess.run([encoded_list_obtained,residual_list_obtained],
                                   feed_dict={input_batch:transfer_targets[iter*self.batch_size:(iter+1)*self.batch_size,:,:,:]})
                for ii in range(len(encoded_list_obtained)):
                    encoded_list_input_calculated[ii][iter*self.batch_size:(iter+1)*self.batch_size,:,:,:] = this_batch_encoded_list[ii]
                for ii in range(len(residual_list_obtained)):
                    residual_list_input_calculated[ii][iter*self.batch_size:(iter+1)*self.batch_size,:,:,:] = this_batch_residual_list[ii]

            for ii in range(len(encoded_list_obtained)):
                encoded_list_input_calculated[ii] = encoded_list_input_calculated[ii][0:self.style_input_number,:,:,:]
                encoded_list_input_calculated[ii] = np.mean(encoded_list_input_calculated[ii],axis=0)
                encoded_list_input_calculated[ii] = np.tile(np.reshape(encoded_list_input_calculated[ii],
                                                                       [1,encoded_list_input_calculated[ii].shape[0],
                                                                        encoded_list_input_calculated[ii].shape[1],
                                                                        encoded_list_input_calculated[ii].shape[2]]),
                                                            [self.batch_size,1,1,1])
            for ii in range(len(residual_list_obtained)):
                residual_list_input_calculated[ii] = residual_list_input_calculated[ii][0:self.style_input_number, :, :,:]
                residual_list_input_calculated[ii] = np.mean(residual_list_input_calculated[ii], axis=0)
                residual_list_input_calculated[ii] = np.tile(np.reshape(residual_list_input_calculated[ii],
                                                                       [1, residual_list_input_calculated[ii].shape[0],
                                                                        residual_list_input_calculated[ii].shape[1],
                                                                        residual_list_input_calculated[ii].shape[2]]),
                                                            [self.batch_size, 1, 1, 1])





            # build feed_dictionary
            output_dict = {}
            for ii in range(len(encoded_list_obtained)):
                output_dict.update({encoded_list_input[ii]:encoded_list_input_calculated[ii]})
            for ii in range(len(residual_list_obtained)):
                output_dict.update({residual_list_input[ii]:residual_list_input_calculated[ii]})




            # run generator
            generator_infer_iter_num = source_imgs.shape[0] / self.batch_size
            generated_target = np.zeros([source_imgs.shape[0],self.img2img_width,self.img2img_width,self.input_output_img_filter_num])
            all_start_time=time.time()
            for iter in range(generator_infer_iter_num):
                iter_start_time=time.time()
                current_feed_dict = output_dict
                current_feed_dict.update({source:source_imgs[iter*self.batch_size:(iter+1)*self.batch_size,:,:,:]})
                generated_target[iter*self.batch_size:(iter+1)*self.batch_size,:,:,:] \
                    = self.sess.run(generated_infer,
                                    feed_dict=current_feed_dict)

                print("Style:%s,Iter:%d/%d,Elapsed:%.3f/%.3f" %  (target_label1,iter+1,generator_infer_iter_num,
                                                                  time.time()-iter_start_time,time.time()-all_start_time))


            # for save

            generated_target=generated_target[0:len(targeted_charset),:,:,:]
            true_targets = true_targets[0:len(targeted_charset), :, :, :]

            output_paper_generated = inf_tools.matrix_paper_generation(images=generated_target,
                                                                       rows=output_paper_rows,
                                                                       columns=output_paper_cols)
            output_paper_true = inf_tools.matrix_paper_generation(images=true_targets,
                                                                  rows=output_paper_rows,
                                                                  columns=output_paper_cols)



            output_paper_generated.save(os.path.join(save_dir_generated,'GeneratedFont:%s.png' % target_label1))
            output_paper_true.save(os.path.join(save_dir_true, 'RealFont:%s.png' % target_label1))
            print("ImgSaved:%s" % target_label1)
            print(self.print_separater)
            print(self.print_separater)


    def styleadd_infer_procedures_mode1(self):

        self.batch_size = 1

        charset_level1, character_label_level1 = \
            inf_tools.get_chars_set_from_level1_2(path='../FontAndChars/GB2312_Level_1.txt',
                                                  level=1)
        charset_level2, character_label_level2 = \
            inf_tools.get_chars_set_from_level1_2(path='../FontAndChars/GB2312_Level_2.txt',
                                                  level=2)

        targeted_charset, targeted_label0_list = \
            inf_tools.get_chars_set_from_searching(path=self.targeted_chars_txt,
                                                   level1_charlist=charset_level1,
                                                   level2_charlist=charset_level2,
                                                   level1_labellist=character_label_level1,
                                                   level2_labellist=character_label_level2)

        source_imgs = inf_tools.find_source_char_img(charset=targeted_charset,
                                                     fontpath=self.source_font,
                                                     img_width=self.source_img_width,
                                                     img_filters=self.input_output_img_filter_num,
                                                     batch_size=self.batch_size)

        output_paper_shape = self.save_mode.split(':')
        output_paper_rows = int(output_paper_shape[0])
        output_paper_cols = int(output_paper_shape[1])
        if output_paper_rows * output_paper_cols < len(targeted_charset):
            print('Incorrect Paper Size !@!~!@~!~@~!@~')
            return



        with tf.Graph().as_default():
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            # build embedder
            with tf.device(self.generator_devices):




                input_batch = tf.placeholder(tf.float32,
                                             [self.batch_size,
                                              self.source_img_width,
                                              self.source_img_width,
                                              self.input_output_img_filter_num],
                                             name='input_batch')
                _, _, encoded_list_obtained, residual_list_obtained, _ = \
                    encoder_implementation(images=input_batch,
                                           is_training=False,
                                           encoder_device=self.generator_devices,
                                           residual_at_layer=self.generator_residual_at_layer,
                                           residual_connection_mode='Single',
                                           scope='generator/target_encoder',
                                           reuse=False,
                                           initializer='XavierInit',
                                           weight_decay=False,
                                           weight_decay_rate=0,
                                           final_layer_logit_length=-1)




            with tf.device(self.generator_devices):
                source = tf.placeholder(tf.float32,
                                        [self.batch_size,
                                         self.source_img_width,
                                         self.source_img_width,
                                         self.input_output_img_filter_num],
                                        name='geneartor_source_image')

                encoded_list_input = list()
                for ii in range(len(encoded_list_obtained)):
                    tmp = tf.placeholder(shape=encoded_list_obtained[ii].shape,
                                         dtype=encoded_list_obtained[ii].dtype)
                    encoded_list_input.append(tmp)
                residual_list_input = list()
                for ii in range(len(residual_list_obtained)):
                    tmp = tf.placeholder(shape=residual_list_obtained[ii].shape,
                                         dtype=residual_list_obtained[ii].dtype)
                    residual_list_input.append(tmp)


                generated_infer = generator_inferring(source=source,
                                                      batch_size=self.batch_size,
                                                      generator_device=self.generator_devices,
                                                      residual_at_layer=self.generator_residual_at_layer,
                                                      residual_block_num=self.generator_residual_blocks,
                                                      target_encoded_list=encoded_list_input,
                                                      residual_input_list=residual_list_input)

                gen_vars_save = self.find_bn_avg_var([var for var in tf.trainable_variables() if 'generator' in var.name])



            # model load
            saver_generator = tf.train.Saver(max_to_keep=1, var_list=gen_vars_save)
            generator_restored = self.restore_model(saver=saver_generator,
                                                    model_dir=self.model_dir,
                                                    model_name='Generator')

            if not generator_restored:
                return
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)



        for target_label1 in self.target_label1_selection:

            true_targets = inf_tools.find_true_targets(data_dir=self.style_add_target_model_dir,
                                                       txt_path=self.style_add_target_file_list,
                                                       selected_label1=int(target_label1),
                                                       selected_label0_list=targeted_label0_list,
                                                       charset=targeted_charset, font=self.source_font,
                                                       img_width=self.source_img_width,
                                                       filter_num=self.input_output_img_filter_num,
                                                       batch_size=self.batch_size)

            transfer_target_counter=0
            for current_transfer_target in true_targets:

                if transfer_target_counter == 10:
                    transfer_target_counter +=1
                    continue
                # transfer_targets = np.zeros(shape=transfer_targets.shape,
                #                             dtype=transfer_targets.dtype)
                transfer_targets = current_transfer_target
                transfer_targets = np.expand_dims(transfer_targets,axis=0)

                # calculate ebdd_vec
                style_calculate_iters = 1
                encoded_list_input_calculated = list()
                for ii in range(len(encoded_list_obtained)):
                    tmp = np.zeros(shape=[style_calculate_iters*self.batch_size,
                                          int(encoded_list_obtained[ii].shape[1]),
                                          int(encoded_list_obtained[ii].shape[2]),
                                          int(encoded_list_obtained[ii].shape[3])],
                                   dtype=np.float32)
                    encoded_list_input_calculated.append(tmp)
                residual_list_input_calculated = list()
                for ii in range(len(residual_list_obtained)):
                    tmp = np.zeros(shape=[style_calculate_iters * self.batch_size,
                                          int(residual_list_obtained[ii].shape[1]),
                                          int(residual_list_obtained[ii].shape[2]),
                                          int(residual_list_obtained[ii].shape[3])],
                                   dtype=np.float32)
                    residual_list_input_calculated.append(tmp)

                for iter in range(style_calculate_iters):
                    this_batch_encoded_list, this_batch_residual_list\
                        =self.sess.run([encoded_list_obtained,residual_list_obtained],
                                       feed_dict={input_batch:transfer_targets})
                    for ii in range(len(encoded_list_obtained)):
                        encoded_list_input_calculated[ii][iter*self.batch_size:(iter+1)*self.batch_size,:,:,:] = this_batch_encoded_list[ii]
                    for ii in range(len(residual_list_obtained)):
                        residual_list_input_calculated[ii][iter*self.batch_size:(iter+1)*self.batch_size,:,:,:] = this_batch_residual_list[ii]

                for ii in range(len(encoded_list_obtained)):
                    encoded_list_input_calculated[ii] = encoded_list_input_calculated[ii][0:self.style_input_number,:,:,:]
                    encoded_list_input_calculated[ii] = np.mean(encoded_list_input_calculated[ii],axis=0)
                    encoded_list_input_calculated[ii] = np.tile(np.reshape(encoded_list_input_calculated[ii],
                                                                           [1,encoded_list_input_calculated[ii].shape[0],
                                                                            encoded_list_input_calculated[ii].shape[1],
                                                                            encoded_list_input_calculated[ii].shape[2]]),
                                                                [self.batch_size,1,1,1])
                for ii in range(len(residual_list_obtained)):
                    residual_list_input_calculated[ii] = residual_list_input_calculated[ii][0:self.style_input_number, :, :,:]
                    residual_list_input_calculated[ii] = np.mean(residual_list_input_calculated[ii], axis=0)
                    residual_list_input_calculated[ii] = np.tile(np.reshape(residual_list_input_calculated[ii],
                                                                           [1, residual_list_input_calculated[ii].shape[0],
                                                                            residual_list_input_calculated[ii].shape[1],
                                                                            residual_list_input_calculated[ii].shape[2]]),
                                                                [self.batch_size, 1, 1, 1])





                # build feed_dictionary
                output_dict = {}
                for ii in range(len(encoded_list_obtained)):
                    output_dict.update({encoded_list_input[ii]:encoded_list_input_calculated[ii]})
                for ii in range(len(residual_list_obtained)):
                    output_dict.update({residual_list_input[ii]:residual_list_input_calculated[ii]})




                current_feed_dict = output_dict
                current_feed_dict.update(
                    {source: source_imgs[transfer_target_counter * self.batch_size:(transfer_target_counter + 1) * self.batch_size, :, :, :]})
                current_generated = self.sess.run(generated_infer, feed_dict=current_feed_dict)
                current_true = transfer_targets
                current_prototype = source_imgs[transfer_target_counter * self.batch_size:(transfer_target_counter + 1) * self.batch_size, :, :, :]
                current_prototype = np.squeeze(current_prototype)
                current_true_fake = np.concatenate([np.squeeze(current_generated), np.squeeze(current_true)], axis=0)

                if transfer_target_counter == 0:
                    full_save = current_true_fake
                    full_prototype = current_prototype
                else:
                    full_save = np.concatenate([full_save, current_true_fake], axis=1)
                    full_prototype = np.concatenate([full_prototype, current_prototype], axis=1)

                transfer_target_counter += 1



            misc.imsave(os.path.join(self.infer_dir, 'Comparsion:%s.png' % target_label1),full_save)
            print("ImgSaved:%s" % target_label1)
            print(self.print_separater)
            print(self.print_separater)



    def framework_building(self):
        # for model base frameworks
        with tf.device('/device:CPU:0'):
            global_step = tf.get_variable('global_step',
                                          [],
                                          initializer=tf.constant_initializer(0),
                                          trainable=False,
                                          dtype=tf.int32)
            epoch_step = tf.get_variable('epoch_step',
                                         [],
                                         initializer=tf.constant_initializer(0),
                                         trainable=False,
                                         dtype=tf.int32)
            epoch_step_increase_one_op = tf.assign(epoch_step, epoch_step + 1)
            learning_rate = tf.placeholder(tf.float32, name="learning_rate")

            framework_var_list = list()
            framework_var_list.append(global_step)
            framework_var_list.append(epoch_step)

        saver_frameworks = tf.train.Saver(max_to_keep=self.model_save_epochs, var_list=framework_var_list)


        print("Framework built @%s." % '/device:CPU:0')
        return epoch_step_increase_one_op, learning_rate, global_step, epoch_step, saver_frameworks







    def feature_extractor_build(self, g_loss, g_merged_summary):

        def calculate_high_level_feature_loss(feature1,feature2):
            for counter in range(len(feature1)):

                feature_diff = feature1[counter] - feature2[counter]
                if not feature_diff.shape.ndims==4:
                    feature_diff = tf.reshape(feature_diff,[int(feature_diff.shape[0]),int(feature_diff.shape[1]),1,1])
                squared_feature_diff = feature_diff**2
                mean_squared_feature_diff = tf.reduce_mean(squared_feature_diff,axis=[1,2,3])
                square_root_mean_squared_feature_diff = tf.sqrt(eps+mean_squared_feature_diff)
                this_feature_loss = tf.reduce_mean(square_root_mean_squared_feature_diff)

                if counter == 0:
                    final_loss = this_feature_loss
                else:
                    final_loss += this_feature_loss
            final_loss = final_loss / len(feature1)

            return final_loss

        def build_feature_extractor(input_target_infer,input_true_img,
                                    label0_length, label1_length,
                                    extractor_usage):
            generator_handle = getattr(self,'generator_handle')
            output_logit_list = list()


            with tf.variable_scope(tf.get_variable_scope()):
                with tf.device(self.feature_extractor_device):


                    _, _, real_features,network_info = \
                        feature_extractor_network(image=input_true_img,
                                                  batch_size=self.batch_size,
                                                  device=self.feature_extractor_device,
                                                  label0_length=label0_length,
                                                  label1_length=label1_length,
                                                  reuse=False,
                                                  keep_prob=1,
                                                  initializer=self.initializer,
                                                  network_usage=extractor_usage)
                    _, _, fake_features, _ = \
                        feature_extractor_network(image=generator_handle.generated_target_train,
                                                  batch_size=self.batch_size,
                                                  device=self.feature_extractor_device,
                                                  label0_length=label0_length,
                                                  label1_length=label1_length,
                                                  reuse=True,
                                                  keep_prob=1,
                                                  initializer=self.initializer,
                                                  network_usage=extractor_usage)

                    label1_logits, label0_logits, _, _ = \
                        feature_extractor_network(image=input_target_infer,
                                                  batch_size=self.batch_size,
                                                  device=self.feature_extractor_device,
                                                  label0_length=label0_length,
                                                  label1_length=label1_length,
                                                  reuse=True,
                                                  keep_prob=1,
                                                  initializer=self.initializer,
                                                  network_usage=extractor_usage)
            if not label0_length==-1:
                output_logit_list.append(label0_logits)
            if not label1_length==-1:
                output_logit_list.append(label1_logits)

            feature_loss = calculate_high_level_feature_loss(feature1=real_features,
                                                             feature2=fake_features)

            return output_logit_list, feature_loss, network_info

        def define_entropy_accuracy_calculation_op(true_labels, infer_logits, summary_name):
            extr_prdt = tf.argmax(infer_logits, axis=1)
            extr_true = tf.argmax(true_labels, axis=1)

            correct = tf.equal(extr_prdt, extr_true)
            acry = tf.reduce_mean(tf.cast(correct, tf.float32)) * 100

            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=infer_logits, labels=tf.nn.softmax(infer_logits))
            entropy = tf.reduce_mean(entropy)

            trn_acry_real = tf.summary.scalar("Accuracy_" + summary_name + "/TrainReal", acry)
            val_acry_real = tf.summary.scalar("Accuracy_" + summary_name + "/TestReal", acry)
            trn_acry_fake = tf.summary.scalar("Accuracy_" + summary_name + "/TrainFake", acry)
            val_acry_fake = tf.summary.scalar("Accuracy_" + summary_name + "/TestFake", acry)


            trn_enpy_real = tf.summary.scalar("Entropy_" + summary_name + "/TrainReal", entropy)
            val_enpy_real = tf.summary.scalar("Entropy_" + summary_name + "/TestReal", entropy)
            trn_enpy_fake = tf.summary.scalar("Entropy_" + summary_name + "/TrainFake", entropy)
            val_enpy_fake = tf.summary.scalar("Entropy_" + summary_name + "/TestFake", entropy)

            trn_real_merged = tf.summary.merge([trn_acry_real, trn_enpy_real])
            trn_fake_merged = tf.summary.merge([trn_acry_fake, trn_enpy_fake])

            val_real_merged = tf.summary.merge([val_acry_real, val_enpy_real])
            val_fake_merged = tf.summary.merge([val_acry_fake, val_enpy_fake])

            return trn_real_merged, trn_fake_merged, val_real_merged, val_fake_merged

        extr_trn_real_merged = []
        extr_trn_fake_merged = []
        extr_val_real_merged = []
        extr_val_fake_merged = []
        saver_list = list()

        input_target_infer = tf.placeholder(tf.float32,
                                            [self.batch_size,
                                             self.source_img_width,
                                             self.source_img_width,
                                             self.input_output_img_filter_num],
                                            name='extractor_infer_img_input')
        generator_handle = getattr(self, 'generator_handle')

        if self.extractor_true_fake_enabled:
            true_fake_infer_list, true_fake_feature_loss, network_info = \
                build_feature_extractor(input_target_infer=input_target_infer,
                                        input_true_img=generator_handle.true_style,
                                        label0_length=len(self.involved_label0_list),
                                        label1_length=len(self.involved_label1_list),
                                        extractor_usage='TrueFake_FeatureExtractor')
            g_loss += true_fake_feature_loss
            feature_true_fake_loss_summary = tf.summary.scalar("Loss_FeatureExtractor/TrueFakeL2",
                                                               tf.abs(true_fake_feature_loss) / self.Feature_Penalty_True_Fake_Target)
            g_merged_summary = tf.summary.merge([g_merged_summary, feature_true_fake_loss_summary])

            extr_vars_true_fake = [var for var in tf.trainable_variables() if 'TrueFake_FeatureExtractor' in var.name]
            extr_vars_true_fake = self.find_bn_avg_var(extr_vars_true_fake)
            extr_vars_true_fake = self.variable_dict(var_input=extr_vars_true_fake, delete_name_from_character='/')
            saver_extractor_true_fake = tf.train.Saver(max_to_keep=1, var_list=extr_vars_true_fake)

            summary_train_real_merged_label0, \
            summary_train_fake_merged_label0, \
            summary_val_real_merged_label0, \
            summary_val_fake_merged_label0 = \
                define_entropy_accuracy_calculation_op(true_labels=generator_handle.true_label0,
                                                       infer_logits=true_fake_infer_list[0],
                                                       summary_name="Extractor_TrueFake/Lb0")

            summary_train_real_merged_label1, \
            summary_train_fake_merged_label1, \
            summary_val_real_merged_label1, \
            summary_val_fake_merged_label1 = \
                define_entropy_accuracy_calculation_op(true_labels=generator_handle.true_label1,
                                                       infer_logits=true_fake_infer_list[1],
                                                       summary_name="Extractor_TrueFake/Lb1")

            extr_trn_real_merged = tf.summary.merge([extr_trn_real_merged, summary_train_real_merged_label0, summary_train_real_merged_label1])
            extr_trn_fake_merged = tf.summary.merge([extr_trn_fake_merged, summary_train_fake_merged_label0, summary_train_fake_merged_label1])
            extr_val_real_merged = tf.summary.merge([extr_val_real_merged, summary_val_real_merged_label0, summary_val_real_merged_label1])
            extr_val_fake_merged = tf.summary.merge([extr_val_fake_merged, summary_val_fake_merged_label0, summary_val_fake_merged_label1])

            print("TrueFakeExtractor @ %s with %s;" % (self.feature_extractor_device, network_info))



        else:
            true_fake_infer_list=list()
            true_fake_infer_list.append(-1)
            true_fake_infer_list.append(-1)
            saver_extractor_true_fake = None


        if self.extractor_content_prototype_enabled:
            content_prototype = generator_handle.content_prototype
            selected_index = tf.random_uniform(shape=[],minval=0,maxval=int(content_prototype.shape[3]),dtype=tf.int64)
            selected_content_prototype = tf.expand_dims(content_prototype[:,:,:,selected_index], axis=3)
            content_prototype_infer_list,content_prototype_feature_loss, network_info = \
                build_feature_extractor(input_target_infer=input_target_infer,
                                        input_true_img=selected_content_prototype,
                                        label0_length=len(self.involved_label0_list),
                                        label1_length=-1,
                                        extractor_usage='ContentPrototype_FeatureExtractor')
            g_loss += content_prototype_feature_loss
            feature_content_prototype_loss_summary = tf.summary.scalar("Loss_FeatureExtractor/ContentPrototypeL2",
                                                               tf.abs(content_prototype_feature_loss) / self.Feature_Penalty_Content_Prototype)
            g_merged_summary = tf.summary.merge([g_merged_summary, feature_content_prototype_loss_summary])

            extr_vars_content_prototype = [var for var in tf.trainable_variables() if 'ContentPrototype_FeatureExtractor' in var.name]
            extr_vars_content_prototype = self.find_bn_avg_var(extr_vars_content_prototype)
            extr_vars_content_prototype = self.variable_dict(var_input=extr_vars_content_prototype, delete_name_from_character='/')
            saver_extractor_content_prototype = tf.train.Saver(max_to_keep=1, var_list=extr_vars_content_prototype)

            summary_train_real_merged_label0, \
            summary_train_fake_merged_label0, \
            summary_val_real_merged_label0, \
            summary_val_fake_merged_label0 = \
                define_entropy_accuracy_calculation_op(true_labels=generator_handle.true_label0,
                                                       infer_logits=content_prototype_infer_list[0],
                                                       summary_name="Extractor_ContentPrototype/Lb0")

            extr_trn_real_merged = tf.summary.merge([extr_trn_real_merged, summary_train_real_merged_label0])
            extr_trn_fake_merged = tf.summary.merge([extr_trn_fake_merged, summary_train_fake_merged_label0])
            extr_val_real_merged = tf.summary.merge([extr_val_real_merged, summary_val_real_merged_label0])
            extr_val_fake_merged = tf.summary.merge([extr_val_fake_merged, summary_val_fake_merged_label0])

            print("ContentPrototypeExtractor @ %s with %s;" % (self.feature_extractor_device, network_info))


        else:
            content_prototype_infer_list=list()
            content_prototype_infer_list.append(-1)
            saver_extractor_content_prototype = None
            selected_content_prototype = -1

        if self.extractor_style_reference_enabled:
            style_reference = generator_handle.style_reference_list
            selected_index = tf.random_uniform(shape=[], minval=0, maxval=self.style_input_number, dtype=tf.int64)
            selected_style_reference = tf.gather(style_reference,selected_index)
            style_reference_infer_list, style_reference_feature_loss, network_info = \
                build_feature_extractor(input_target_infer=input_target_infer,
                                        input_true_img = selected_style_reference,
                                        label0_length=-1,
                                        label1_length=len(self.involved_label1_list),
                                        extractor_usage='StyleReference_FeatureExtractor')
            g_loss += style_reference_feature_loss
            feature_style_reference_loss_summary = tf.summary.scalar("Loss_FeatureExtractor/StyleReferenceL2",
                                                                     tf.abs(style_reference_feature_loss) / self.Feature_Penalty_Style_Reference)
            g_merged_summary = tf.summary.merge([g_merged_summary, feature_style_reference_loss_summary])

            extr_vars_style_reference = [var for var in tf.trainable_variables() if 'StyleReference_FeatureExtractor' in var.name]
            extr_vars_style_reference = self.find_bn_avg_var(extr_vars_style_reference)
            extr_vars_style_reference = self.variable_dict(var_input=extr_vars_style_reference, delete_name_from_character='/')
            saver_extractor_style_reference = tf.train.Saver(max_to_keep=1, var_list=extr_vars_style_reference)

            summary_train_real_merged_label1, \
            summary_train_fake_merged_label1, \
            summary_val_real_merged_label1, \
            summary_val_fake_merged_label1 = \
                define_entropy_accuracy_calculation_op(true_labels=generator_handle.true_label1,
                                                       infer_logits=style_reference_infer_list[0],
                                                       summary_name="Extractor_StyleReference/Lb1")

            extr_trn_real_merged = tf.summary.merge([extr_trn_real_merged, summary_train_real_merged_label1])
            extr_trn_fake_merged = tf.summary.merge([extr_trn_fake_merged, summary_train_fake_merged_label1])
            extr_val_real_merged = tf.summary.merge([extr_val_real_merged, summary_val_real_merged_label1])
            extr_val_fake_merged = tf.summary.merge([extr_val_fake_merged, summary_val_fake_merged_label1])
            print("StyleReferenceExtractor @ %s with %s;" % (self.feature_extractor_device, network_info))

        else:
            style_reference_infer_list=list()
            style_reference_infer_list.append(-1)
            saver_extractor_style_reference = None
            selected_style_reference = -1


        feature_extractor_handle = FeatureExtractorHandle(infer_input_img=input_target_infer,
                                                          selected_content_prototype=selected_content_prototype,
                                                          selected_style_reference=selected_style_reference)
        setattr(self, "feature_extractor_handle", feature_extractor_handle)


        saver_list.append(saver_extractor_true_fake)
        saver_list.append(saver_extractor_content_prototype)
        saver_list.append(saver_extractor_style_reference)


        return g_loss, g_merged_summary,saver_list,\
               extr_trn_real_merged,extr_trn_fake_merged,extr_val_real_merged,extr_val_fake_merged



    def generator_build(self):

        name_prefix = 'generator'

        # network architechture
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(self.generator_devices):

                # input definitions of the generator
                content_prototype = tf.placeholder(tf.float32,
                                                   [self.batch_size,
                                                    self.source_img_width,
                                                    self.source_img_width,
                                                    self.input_output_img_filter_num * self.content_input_num],
                                                   name='generator_content_prototype')

                style_reference_list = list()
                for ii in range(self.style_input_number):
                    style_reference = tf.placeholder(tf.float32,
                                                     [self.batch_size,
                                                      self.source_img_width,
                                                      self.source_img_width,
                                                      self.input_output_img_filter_num],
                                                     name='generator_style_reference_%d' % ii)
                    style_reference_list.append(style_reference)



                true_style = tf.placeholder(tf.float32,
                                            [self.batch_size,
                                             self.source_img_width,
                                             self.source_img_width,
                                             self.input_output_img_filter_num],
                                            name='generator_true_target')
                true_label0 = tf.placeholder(tf.float32,
                                             [self.batch_size,
                                              len(self.involved_label0_list)],
                                             name='generator_true_label0')
                true_label1 = tf.placeholder(tf.float32,
                                             [self.batch_size,
                                              len(self.involved_label1_list)],
                                             name='generator_true_label1')






                # build the generator
                generated_target_train, encoded_content_prototype_train, encoded_style_reference_train,\
                content_category_logits_train, style_category_logits_train, network_info = \
                    generator_implementation(content_prototype=content_prototype,
                                             style_reference=style_reference_list,
                                             is_training=True,
                                             batch_size=self.batch_size,
                                             generator_device=self.generator_devices,
                                             residual_at_layer=self.generator_residual_at_layer,
                                             residual_block_num=self.generator_residual_blocks,
                                             label0_length=len(self.involved_label0_list),
                                             label1_length=len(self.involved_label1_list),
                                             scope=name_prefix,
                                             reuse=False,
                                             initializer=self.initializer,
                                             weight_decay=self.weight_decay_generator,
                                             style_input_number=self.style_input_number,
                                             content_prototype_number=self.content_input_num,
                                             weight_decay_rate=self.generator_weight_decay_penalty)

                # encoded of the generated target on the content prototype encoder
                encoded_content_prototype_generated_target = \
                    encoder_implementation(images=tf.tile(generated_target_train,
                                                          [1,1,1,self.content_input_num]),
                                           is_training=True,
                                           encoder_device=self.generator_devices,
                                           residual_at_layer=self.generator_residual_at_layer,
                                           residual_connection_mode='Multi',
                                           scope=name_prefix+'/content_encoder',
                                           reuse=True,
                                           initializer=self.initializer,
                                           weight_decay=False,
                                           weight_decay_rate=self.generator_weight_decay_penalty,
                                           final_layer_logit_length=len(self.involved_label0_list))[0]

                # encoded of the generated target on the style reference encoder
                encoded_style_reference_generated_target = \
                        encoder_implementation(images=generated_target_train,
                                               is_training=True,
                                               encoder_device=self.generator_devices,
                                               residual_at_layer=self.generator_residual_at_layer,
                                               residual_connection_mode='Single',
                                               scope=name_prefix + '/style_encoder',
                                               reuse=True,
                                               initializer=self.initializer,
                                               weight_decay=False,
                                               weight_decay_rate=self.generator_weight_decay_penalty,
                                               final_layer_logit_length=len(self.involved_label1_list))[0]


                # for inferring
                generated_target_infer = \
                    generator_implementation(content_prototype=content_prototype,
                                             style_reference=style_reference_list,
                                             is_training=False,
                                             batch_size=self.batch_size,
                                             generator_device=self.generator_devices,
                                             residual_at_layer=self.generator_residual_at_layer,
                                             residual_block_num=self.generator_residual_blocks,
                                             label0_length=len(self.involved_label0_list),
                                             label1_length=len(self.involved_label1_list),
                                             scope=name_prefix,
                                             reuse=True,
                                             initializer=self.initializer,
                                             weight_decay=False,
                                             style_input_number=self.style_input_number,
                                             content_prototype_number=self.content_input_num,
                                             weight_decay_rate=eps)[0]

                content_prototype_category_infer = \
                        encoder_implementation(images=content_prototype,
                                               is_training=False,
                                               encoder_device=self.generator_devices,
                                               residual_at_layer=self.generator_residual_at_layer,
                                               residual_connection_mode='Multi',
                                               scope=name_prefix + '/content_encoder',
                                               reuse=True,
                                               initializer=self.initializer,
                                               weight_decay=False,
                                               weight_decay_rate=self.generator_weight_decay_penalty,
                                               final_layer_logit_length=len(self.involved_label0_list))[1]

                style_reference_category_list_infer = list()
                for ii in range(self.style_input_number):
                    encoded_style_category_logit = \
                            encoder_implementation(images=style_reference_list[ii],
                                                   is_training=False,
                                                   encoder_device=self.generator_devices,
                                                   residual_at_layer=self.generator_residual_at_layer,
                                                   residual_connection_mode='Single',
                                                   scope=name_prefix + '/style_encoder',
                                                   reuse=True,
                                                   initializer=self.initializer,
                                                   weight_decay=False,
                                                   weight_decay_rate=self.generator_weight_decay_penalty,
                                                   final_layer_logit_length=len(self.involved_label1_list))[1]
                    style_reference_category_list_infer.append(encoded_style_category_logit)
                style_reference_category_infer = 0
                for ii in range(self.style_input_number):
                    style_reference_category_infer += style_reference_category_list_infer[ii]
                style_reference_category_infer = style_reference_category_infer / self.style_input_number




                curt_generator_handle = GeneratorHandle(content_prototype=content_prototype,
                                                        true_style=true_style,
                                                        style_reference_list=style_reference_list,
                                                        true_label0=true_label0,
                                                        true_label1=true_label1,
                                                        generated_target_train=generated_target_train,
                                                        generated_target_infer=generated_target_infer)
                setattr(self, "generator_handle", curt_generator_handle)



        # loss build
        g_loss=0
        g_merged_summary = []

        # weight_decay_loss
        generator_weight_decay_loss = tf.get_collection('generator_weight_decay')
        weight_decay_loss = 0
        if generator_weight_decay_loss:
            for ii in generator_weight_decay_loss:
                weight_decay_loss = ii + weight_decay_loss
            weight_decay_loss = weight_decay_loss / len(generator_weight_decay_loss)
            generator_weight_decay_loss_summary = tf.summary.scalar("Loss_Generator/WeightDecay",
                                                                    tf.abs(weight_decay_loss)/self.generator_weight_decay_penalty)
            g_loss += weight_decay_loss
            g_merged_summary = tf.summary.merge([g_merged_summary, generator_weight_decay_loss_summary])


        # const loss for both source and real target
        if self.Lconst_content_Penalty > eps * 10:
            const_loss_content = tf.square(encoded_content_prototype_generated_target - encoded_content_prototype_train)
            const_loss_content = tf.reduce_mean(const_loss_content) * self.Lconst_content_Penalty
            const_loss_content = const_loss_content / self.content_input_num
            g_loss += const_loss_content
            const_content_loss_summary = tf.summary.scalar("Loss_Generator/ConstContentPrototype",
                                                           tf.abs(const_loss_content) / self.Lconst_content_Penalty)
            g_merged_summary=tf.summary.merge([g_merged_summary, const_content_loss_summary])
        if self.Lconst_style_Penalty > eps * 10:
            current_const_loss_style = tf.square(encoded_style_reference_train - encoded_style_reference_generated_target)
            current_const_loss_style = tf.reduce_mean(current_const_loss_style) * self.Lconst_style_Penalty
            g_loss += current_const_loss_style
            const_style_loss_summary = tf.summary.scalar("Loss_Generator/ConstStyleReference",
                                                         tf.abs(current_const_loss_style) / self.Lconst_style_Penalty)
            g_merged_summary=tf.summary.merge([g_merged_summary, const_style_loss_summary])



        # l1 loss
        if self.L1_Penalty > eps * 10:
            l1_loss = tf.abs(generated_target_train - true_style)
            l1_loss = tf.reduce_mean(l1_loss) * self.L1_Penalty
            l1_loss_summary = tf.summary.scalar("Loss_Generator/L1_Pixel",
                                                tf.abs(l1_loss) / self.L1_Penalty)
            g_loss+=l1_loss
            g_merged_summary = tf.summary.merge([g_merged_summary, l1_loss_summary])

        # category loss
        if self.Generator_Categorical_Penalty > 10 * eps:
            category_loss_content = tf.nn.softmax_cross_entropy_with_logits(logits=content_category_logits_train,
                                                                            labels=true_label0)
            category_loss_content = tf.reduce_mean(category_loss_content) * self.Generator_Categorical_Penalty
            category_loss_content = category_loss_content / self.content_input_num

            category_loss_style = tf.nn.softmax_cross_entropy_with_logits(logits=style_category_logits_train,
                                                                          labels=true_label1)
            category_loss_style = tf.reduce_mean(category_loss_style) * self.Generator_Categorical_Penalty
            category_loss = (category_loss_content + category_loss_style) / 2.0

            content_category_loss_summary = tf.summary.scalar("Loss_Generator/CategoryContentPrototype",
                                                             tf.abs(category_loss_content) / self.Generator_Categorical_Penalty)
            style_category_loss_summary = tf.summary.scalar("Loss_Generator/CategoryStyleReference",
                                                             tf.abs(category_loss_style) / self.Generator_Categorical_Penalty)
            category_loss_summary = tf.summary.scalar("Loss_Generator/Category",
                                                      tf.abs(category_loss) / self.Generator_Categorical_Penalty)
            generator_category_loss_summary_merged = tf.summary.merge([category_loss_summary,
                                                                       content_category_loss_summary,
                                                                       style_category_loss_summary])
            g_loss += category_loss
            g_merged_summary = tf.summary.merge([g_merged_summary,
                                                 content_category_loss_summary,
                                                 style_category_loss_summary,
                                                 category_loss_summary])




        # build accuracy and entropy calculation
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(self.generator_devices):
                content_true = tf.argmax(true_label0,axis=1)
                content_prdt = tf.argmax(content_prototype_category_infer, axis=1)
                content_accuracy = tf.equal(content_true,content_prdt)
                content_accuracy = tf.reduce_mean(tf.cast(content_accuracy,tf.float32)) * 100
                content_accuracy = content_accuracy / self.content_input_num

                content_entropy = \
                    tf.nn.softmax_cross_entropy_with_logits(logits=content_prototype_category_infer,
                                                            labels=tf.nn.softmax(content_prototype_category_infer))
                content_entropy = tf.reduce_mean(content_entropy)
                content_entropy = content_entropy / self.content_input_num

                style_true = tf.argmax(true_label1, axis=1)
                style_prdt = tf.argmax(style_reference_category_infer, axis=1)
                style_accuracy = tf.equal(style_true, style_prdt)
                style_accuracy = tf.reduce_mean(tf.cast(style_accuracy, tf.float32)) * 100
                style_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=style_reference_category_infer,
                                                                           labels=tf.nn.softmax(style_reference_category_infer))
                style_entropy = tf.reduce_mean(style_entropy)


        train_content_accuracy = tf.summary.scalar("Accuracy_Generator/ContentPrototype_Train", content_accuracy)
        test_content_accuracy = tf.summary.scalar("Accuracy_Generator/ContentPrototype_Test", content_accuracy)
        train_style_accuracy = tf.summary.scalar("Accuracy_Generator/StyleReference_Train", style_accuracy)
        test_style_accuracy = tf.summary.scalar("Accuracy_Generator/StyleReference_Test", style_accuracy)
        train_content_entropy = tf.summary.scalar("Entropy_Generator/ContentPrototype_Train", content_entropy)
        test_content_entropy = tf.summary.scalar("Entropy_Generator/ContentPrototype_Test", content_entropy)
        train_style_entropy = tf.summary.scalar("Entropy_Generator/StyleReference_Train", style_entropy)
        test_style_entropy = tf.summary.scalar("Entropy_Generator/StyleReference_Test", style_entropy)

        train_summary = tf.summary.merge([train_content_accuracy,train_style_accuracy,train_content_entropy,train_style_entropy])
        test_summary = tf.summary.merge([test_content_accuracy, test_style_accuracy, test_content_entropy, test_style_entropy])

        gen_vars_train = [var for var in tf.trainable_variables() if 'generator' in var.name]
        gen_vars_save = self.find_bn_avg_var(gen_vars_train)

        saver_generator = tf.train.Saver(max_to_keep=self.model_save_epochs, var_list=gen_vars_save)


        print(
            "Generator @%s with %s;" % (self.generator_devices, network_info))
        return generated_target_infer, g_loss, g_merged_summary, \
               gen_vars_train, saver_generator,\
               category_loss,generator_category_loss_summary_merged,train_summary, test_summary


    def discriminator_build(self,
                            g_loss,
                            g_merged_summary,
                            generator_category_loss,
                            generator_category_loss_summary):

        generator_handle = getattr(self,'generator_handle')
        fake_style = generator_handle.generated_target_train

        name_prefix = 'discriminator'

        discriminator_category_logit_length = len(self.involved_label1_list)

        critic_logit_length = int(np.floor(math.log(discriminator_category_logit_length) / math.log(2)))
        critic_logit_length = np.power(2,critic_logit_length+1)

        current_critic_logit_penalty = tf.placeholder(tf.float32, [], name='current_critic_logit_penalty')

        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(self.discriminator_devices):


                discriminator_true_label = tf.placeholder(tf.float32,
                                                          [self.batch_size,
                                                           discriminator_category_logit_length],
                                                          name='discriminator_label')
                content_prototype = tf.placeholder(tf.float32,
                                                   [self.batch_size,
                                                    self.source_img_width,
                                                    self.source_img_width,
                                                    self.input_output_img_filter_num],
                                                   name='discriminator_content_prototype_image')
                true_style = tf.placeholder(tf.float32,
                                             [self.batch_size,
                                              self.source_img_width,
                                              self.source_img_width,
                                              self.input_output_img_filter_num],
                                             name='discriminator_true_style_image')
                style_reference = tf.placeholder(tf.float32,
                                              [self.batch_size,
                                               self.source_img_width,
                                               self.source_img_width,
                                               self.input_output_img_filter_num],
                                              name='discriminator_style_reference_image')


                real_pack = tf.concat([content_prototype, true_style, style_reference], axis=3)
                fake_pack = tf.concat([content_prototype, fake_style, style_reference], axis=3)

                real_C_logits,real_Discriminator_logits,network_info = \
                    self.discriminator_implementation(image=real_pack,
                                                      is_training=True,
                                                      parameter_update_device=self.discriminator_devices,
                                                      category_logit_num=discriminator_category_logit_length,
                                                      batch_size=self.batch_size,
                                                      critic_length=critic_logit_length,
                                                      reuse=False,
                                                      initializer=self.initializer,
                                                      weight_decay=self.weight_decay_discriminator,
                                                      scope=name_prefix,
                                                      weight_decay_rate=self.discriminator_weight_decay_penalty)

                fake_C_logits,fake_Discriminator_logits,_ = \
                    self.discriminator_implementation(image=fake_pack,
                                                      is_training=True,
                                                      parameter_update_device=self.discriminator_devices,
                                                      category_logit_num=discriminator_category_logit_length,
                                                      batch_size=self.batch_size,
                                                      critic_length=critic_logit_length,
                                                      reuse=True,
                                                      initializer=self.initializer, 
                                                      weight_decay=self.weight_decay_discriminator,
                                                      scope=name_prefix,
                                                      weight_decay_rate=self.discriminator_weight_decay_penalty)

                epsilon = tf.random_uniform([], 0.0, 1.0)
                interpolated_pair = real_pack*epsilon + (1-epsilon)*fake_pack
                _,intepolated_Cr_logits,_ = self.discriminator_implementation(image=interpolated_pair,
                                                                              is_training=True,
                                                                              parameter_update_device=self.discriminator_devices,
                                                                              category_logit_num=discriminator_category_logit_length,
                                                                              batch_size=self.batch_size,
                                                                              critic_length=critic_logit_length,
                                                                              reuse=True,
                                                                              initializer=self.initializer,
                                                                              weight_decay=self.weight_decay_discriminator,
                                                                              scope=name_prefix,
                                                                              weight_decay_rate=self.discriminator_weight_decay_penalty)
                discriminator_gradients = tf.gradients(intepolated_Cr_logits,interpolated_pair)[0]
                discriminator_slopes = tf.sqrt(eps+tf.reduce_sum(tf.square(discriminator_gradients),reduction_indices=[1]))
                discriminator_slopes = (discriminator_slopes-1.0)**2

                discriminator_infer_content = tf.placeholder(tf.float32,
                                                            [self.batch_size,
                                                             self.source_img_width,
                                                             self.source_img_width,
                                                             self.input_output_img_filter_num],
                                                            name='discriminator_infer_source')
                discriminator_infer_true_or_generated_style = tf.placeholder(tf.float32,
                                                                              [self.batch_size,
                                                                               self.source_img_width,
                                                                               self.source_img_width,
                                                                               self.input_output_img_filter_num],
                                                                              name='discriminator_infer_true_or_generated_target')
                discriminator_infer_style = tf.placeholder(tf.float32,
                                                                  [self.batch_size,
                                                                   self.source_img_width,
                                                                   self.source_img_width,
                                                                   self.input_output_img_filter_num],
                                                                  name='discriminator_infer_input_target')

                discriminator_input_pack = tf.concat([discriminator_infer_content,
                                                      discriminator_infer_true_or_generated_style,
                                                      discriminator_infer_style],
                                                     axis=3)

                infer_C_logits,infer_Cr_logits,_ = \
                    self.discriminator_implementation(image=discriminator_input_pack,
                                                      is_training=False,
                                                      parameter_update_device=self.discriminator_devices,
                                                      category_logit_num=discriminator_category_logit_length,
                                                      batch_size=self.batch_size,
                                                      critic_length=critic_logit_length,
                                                      reuse=True,
                                                      initializer=self.initializer,
                                                      weight_decay=False,
                                                      weight_decay_rate=eps,
                                                      scope=name_prefix)
                infer_C_logits = tf.nn.softmax(infer_C_logits)

                curt_discriminator_handle = DiscriminatorHandle(discriminator_true_label=discriminator_true_label,
                                                                content_prototype=content_prototype,
                                                                true_style=true_style,
                                                                style_reference=style_reference,
                                                                infer_content_prototype=discriminator_infer_content,
                                                                discriminator_infer_true_or_generated_style=discriminator_infer_true_or_generated_style,
                                                                infer_style_reference=discriminator_infer_style,
                                                                infer_C_logits=infer_C_logits,
                                                                infer_Cr_logits=infer_Cr_logits,
                                                                current_critic_logit_penalty=current_critic_logit_penalty)

                setattr(self, "discriminator_handle", curt_discriminator_handle)



        # loss build
        d_loss = 0
        d_merged_summary=[]

        # generator_category_loss
        d_loss += generator_category_loss
        d_merged_summary = tf.summary.merge([d_merged_summary,
                                             generator_category_loss_summary])

        # weight_decay_loss
        discriminator_weight_decay_loss = tf.get_collection('discriminator_weight_decay')
        weight_decay_loss = 0
        if discriminator_weight_decay_loss:
            for ii in discriminator_weight_decay_loss:
                weight_decay_loss += ii
            weight_decay_loss = weight_decay_loss / len(discriminator_weight_decay_loss)
            discriminator_weight_decay_loss_summary = tf.summary.scalar("Loss_Discriminator/WeightDecay",
                                                                        tf.abs(weight_decay_loss)/self.discriminator_weight_decay_penalty)
            d_loss += weight_decay_loss
            d_merged_summary = tf.summary.merge([d_merged_summary,
                                                 discriminator_weight_decay_loss_summary])

        # category loss
        if self.Discriminator_Categorical_Penalty > 10 * eps:
            real_category_loss = tf.nn.softmax_cross_entropy_with_logits(logits=real_C_logits,
                                                                         labels=discriminator_true_label)
            fake_category_loss = tf.nn.softmax_cross_entropy_with_logits(logits=fake_C_logits,
                                                                         labels=discriminator_true_label)

            real_category_loss = tf.reduce_mean(real_category_loss) * self.Discriminator_Categorical_Penalty
            fake_category_loss = tf.reduce_mean(fake_category_loss) * self.Discriminator_Categorical_Penalty
            category_loss = (real_category_loss + fake_category_loss) / 2.0


            real_category_loss_summary = tf.summary.scalar("Loss_Discriminator/CategoryReal",
                                                           tf.abs(real_category_loss) / self.Discriminator_Categorical_Penalty)
            fake_category_loss_summary = tf.summary.scalar("Loss_Discriminator/CategoryFake",
                                                           tf.abs(fake_category_loss) / self.Discriminator_Categorical_Penalty)
            category_loss_summary = tf.summary.scalar("Loss_Discriminator/Category", tf.abs(category_loss) / self.Discriminator_Categorical_Penalty)

            d_loss += category_loss
            d_merged_summary = tf.summary.merge([d_merged_summary,
                                                 real_category_loss_summary,
                                                 fake_category_loss_summary,
                                                 category_loss_summary])

            g_loss+=fake_category_loss
            g_merged_summary=tf.summary.merge([g_merged_summary,fake_category_loss_summary])


        # discriminative loss
        if self.Discriminative_Penalty > 10 * eps:
            d_loss_real = real_Discriminator_logits
            d_loss_fake = -fake_Discriminator_logits

            d_norm_real_loss = tf.abs(tf.abs(d_loss_real) - 1)
            d_norm_fake_loss = tf.abs(tf.abs(d_loss_fake) - 1)

            d_norm_real_loss = tf.reduce_mean(d_norm_real_loss) * current_critic_logit_penalty
            d_norm_fake_loss = tf.reduce_mean(d_norm_fake_loss) * current_critic_logit_penalty
            d_norm_loss = (d_norm_real_loss + d_norm_fake_loss) / 2

            d_norm_real_loss_summary = tf.summary.scalar("Loss_Discriminator/CriticLogit_NormReal",
                                                         d_norm_real_loss / current_critic_logit_penalty)
            d_norm_fake_loss_summary = tf.summary.scalar("Loss_Discriminator/CriticLogit_NormFake",
                                                         d_norm_fake_loss / current_critic_logit_penalty)
            d_norm_loss_summary = tf.summary.scalar("Loss_Discriminator/CriticLogit_Norm", d_norm_loss / current_critic_logit_penalty)

            d_loss += d_norm_loss
            d_merged_summary = tf.summary.merge([d_merged_summary,
                                                 d_norm_real_loss_summary,
                                                 d_norm_fake_loss_summary,
                                                 d_norm_loss_summary])

            d_loss_real = tf.reduce_mean(d_loss_real) * self.Discriminative_Penalty
            d_loss_fake = tf.reduce_mean(d_loss_fake) * self.Discriminative_Penalty
            d_loss_real_fake_summary = tf.summary.scalar("TrainingProgress_DiscriminatorRealFakeLoss",
                                                         tf.abs(
                                                             d_loss_real + d_loss_fake) / self.Discriminative_Penalty)
            if self.Discriminator_Gradient_Penalty > 10 * eps:
                d_gradient_loss = discriminator_slopes
                d_gradient_loss = tf.reduce_mean(d_gradient_loss) * self.Discriminator_Gradient_Penalty
                d_gradient_loss_summary = tf.summary.scalar("Loss_Discriminator/D_Gradient",
                                                            tf.abs(
                                                                d_gradient_loss) / self.Discriminator_Gradient_Penalty)
                d_loss += d_gradient_loss
                d_merged_summary = tf.summary.merge([d_merged_summary,
                                                     d_gradient_loss_summary,
                                                     d_loss_real_fake_summary])

            cheat_loss = fake_Discriminator_logits


            d_loss_real_summary = tf.summary.scalar("Loss_Discriminator/AdversarialReal",
                                                    tf.abs(d_loss_real) / self.Discriminative_Penalty)
            d_loss_fake_summary = tf.summary.scalar("Loss_Discriminator/AdversarialFake",
                                                    tf.abs(d_loss_fake) / self.Discriminative_Penalty)

            d_loss += (d_loss_real+d_loss_fake)/2
            d_merged_summary = tf.summary.merge([d_merged_summary,
                                                 d_loss_fake_summary,
                                                 d_loss_real_summary])

            cheat_loss = tf.reduce_mean(cheat_loss) * self.Discriminative_Penalty
            cheat_loss_summary = tf.summary.scalar("Loss_Generator/Cheat", tf.abs(cheat_loss) / self.Discriminative_Penalty)
            g_loss+=cheat_loss
            g_merged_summary=tf.summary.merge([g_merged_summary,cheat_loss_summary])



        # d_loss_final and g_loss_final
        d_loss_summary = tf.summary.scalar("Loss_Discriminator/Total", tf.abs(d_loss))
        g_loss_summary = tf.summary.scalar("Loss_Generator/Total", tf.abs(g_loss))
        d_merged_summary=tf.summary.merge([d_merged_summary,d_loss_summary])
        g_merged_summary = tf.summary.merge([g_merged_summary, g_loss_summary])




        # build accuracy and entropy claculation
        # discriminator reference build here
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(self.discriminator_devices):
                c_labels_prdt = tf.argmax(infer_C_logits, axis=1)
                c_labels_true = tf.argmax(discriminator_true_label, axis=1)

                correct_prediction_category = tf.equal(c_labels_prdt, c_labels_true)
                accuracy_prediction_category = tf.reduce_mean(tf.cast(correct_prediction_category, tf.float32)) * 100

                shanon_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=infer_C_logits,
                                                                            labels=tf.nn.softmax(infer_C_logits ))
                shanon_entropy = tf.reduce_mean(shanon_entropy)


        trn_real_acry = tf.summary.scalar("Accuracy_Discriminator/AuxClassifier_TrainReal",accuracy_prediction_category)
        trn_fake_acry = tf.summary.scalar("Accuracy_Discriminator/AuxClassifier_TrainFake",accuracy_prediction_category)
        val_real_acry = tf.summary.scalar("Accuracy_Discriminator/AuxClassifier_TestReal",accuracy_prediction_category)
        val_fake_acry = tf.summary.scalar("Accuracy_Discriminator/AuxClassifier_TestFake",accuracy_prediction_category)

        trn_real_enty = tf.summary.scalar("Entropy_Discriminator/AuxClassifier_TrainReal", shanon_entropy)
        trn_fake_enty = tf.summary.scalar("Entropy_Discriminator/AuxClassifier_TrainFake", shanon_entropy)
        val_real_enty = tf.summary.scalar("Entropy_Discriminator/AuxClassifier_TestReal", shanon_entropy)
        val_fake_enty = tf.summary.scalar("Entropy_Discriminator/AuxClassifier_TestFake", shanon_entropy)

        trn_real_summary = tf.summary.merge([trn_real_acry, trn_real_enty])
        trn_fake_summary = tf.summary.merge([trn_fake_acry, trn_fake_enty])

        tst_real_summary = tf.summary.merge([val_real_acry, val_real_enty])
        tst_fake_summary = tf.summary.merge([val_fake_acry, val_fake_enty])

        dis_vars_train = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        dis_vars_save = self.find_bn_avg_var(dis_vars_train)



        saver_discriminator = tf.train.Saver(max_to_keep=self.model_save_epochs, var_list=dis_vars_save)


        print("Discriminator @ %s with %s;" % (self.discriminator_devices,network_info))
        return infer_C_logits,\
               g_merged_summary, d_merged_summary,\
               g_loss,d_loss, \
               trn_real_summary, trn_fake_summary, tst_real_summary, tst_fake_summary, \
               dis_vars_train,saver_discriminator


    def create_optimizer(self,
                         learning_rate,global_step,
                         gen_vars_train,generator_loss_train,
                         dis_vars_train,discriminator_loss_train):

        print(self.print_separater)

        if dis_vars_train:
            if self.optimization_method == 'adam':
                d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(discriminator_loss_train,
                                                                                        var_list=dis_vars_train,
                                                                                        global_step=global_step)
            elif self.optimization_method == 'gradient_descent':
                d_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(discriminator_loss_train,
                                                                                        var_list=dis_vars_train,
                                                                                        global_step=global_step)
            print("Optimizer Discriminator @ %s;" % (self.discriminator_devices))
        else:
            print("The discriminator is frozen.")
            d_optimizer=None

        if gen_vars_train:
            if self.optimization_method == 'adam':
                g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(generator_loss_train,
                                                                                        var_list=gen_vars_train)
            elif self.optimization_method == 'gradient_descent':
                g_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(generator_loss_train,
                                                                                        var_list=gen_vars_train)

            print(
                "Optimizer Generator @ %s;" % (self.generator_devices))

        else:
            g_optimizer = None
            print("The generator is frozen.")
        print(self.print_separater)


        return g_optimizer, d_optimizer


    def divide_variables(self,base_vars, current_model_vars):

        re_init_vars=list()
        non_init_vars=list()


        counter=0
        for model_var in current_model_vars:
            model_var_name=str(model_var.name)
            model_var_name=model_var_name.replace(':0','')




            base_var_shape=[ii for ii in base_vars if model_var_name in ii[0]][0][1]

            same_dimension=True
            for ii in range(len(base_var_shape)):
                if int(model_var.shape[ii]) != base_var_shape[ii]:
                    same_dimension=False
                    break
            if same_dimension:
                non_init_vars.append(model_var)
            else:
                re_init_vars.append(model_var)

            counter+=1
        return re_init_vars,non_init_vars

    def restore_from_previous_model(self, saver_generator, saver_discriminator):
        def list_diff(first, second):
            second = set(second)
            return [item for item in first if item not in second]

        def checking_var_consistency(checking_var, stored_var_name, stored_var_shape):
            check_name = (stored_var_name == str(checking_var.name[:len(checking_var.name) - 2]))
            check_dimension = len(checking_var.shape.dims) == len(stored_var_shape)
            checking_shape_consistent = True
            if check_dimension:
                checking_shape = checking_var.shape

                for ii in range(len(checking_shape.dims)):
                    current_checking_shape = int(checking_shape[ii])
                    current_stored_shape = stored_var_shape[ii]
                    if not current_checking_shape == current_stored_shape:
                        checking_shape_consistent = False
                        break
            return check_name and check_dimension and checking_shape_consistent

        def variable_comparison_and_restore(current_saver, restore_model_dir, model_name):
            ckpt = tf.train.get_checkpoint_state(restore_model_dir)
            output_var_tensor_list = list()
            saved_var_name_list = list()
            current_var_name_list = list()
            for var_name, var_shape in tf.contrib.framework.list_variables(ckpt.model_checkpoint_path):
                for checking_var in current_saver._var_list:
                    found_var = checking_var_consistency(checking_var=checking_var,
                                                         stored_var_name=var_name,
                                                         stored_var_shape=var_shape)
                    if found_var:
                        output_var_tensor_list.append(checking_var)
                    current_var_name_list.append(str(checking_var.name[:len(checking_var.name) - 2]))
                saved_var_name_list.append(var_name)

            ignore_var_tensor_list = list_diff(first=current_saver._var_list,
                                               second=output_var_tensor_list)
            if ignore_var_tensor_list:
                print("IgnoreVars_ForVar in current model but not in the stored model:")
                counter = 0
                for ii in ignore_var_tensor_list:
                    print("No.%d, %s" % (counter, ii))
                    counter += 1
                if not self.debug_mode == 1:
                    raw_input("Press enter to continue")

            current_var_name_list = np.unique(current_var_name_list)
            ignore_var_name_list = list_diff(first=saved_var_name_list,
                                             second=current_var_name_list)
            if ignore_var_name_list:
                print("IgnoreVars_ForVar in stored model but not in the current model:")
                counter = 0
                for ii in ignore_var_name_list:
                    print("No.%d, %s" % (counter, ii))
                    counter += 1
                if not self.debug_mode == 1:
                    raw_input("Press enter to continue")

            saver = tf.train.Saver(max_to_keep=1, var_list=output_var_tensor_list)
            self.restore_model(saver=saver,
                               model_dir=restore_model_dir,
                               model_name=model_name)

        if not self.training_from_model == None:
            variable_comparison_and_restore(current_saver=saver_generator,
                                            restore_model_dir=os.path.join(self.training_from_model, 'generator'),
                                            model_name='Generator_ForPreviousTrainedBaseModel')
            variable_comparison_and_restore(current_saver=saver_discriminator,
                                            restore_model_dir=os.path.join(self.training_from_model,
                                                                           'discriminator'),
                                            model_name='Discriminator_ForPreviousTrainedBaseModel')

    def model_initialization(self,
                             saver_generator, saver_discriminator,
                             saver_framework,
                             feature_extractor_saver_list):
        # initialization of all the variables
        self.sess.run(tf.global_variables_initializer())



        # restore of high_level feature extractor
        if self.extractor_true_fake_enabled:
            extr_restored = self.restore_model(saver=feature_extractor_saver_list[0],
                                               model_dir=self.true_fake_target_extractor_dir,
                                               model_name="TrueFakeExtractor")
        if self.extractor_content_prototype_enabled:
            extr_restored = self.restore_model(saver=feature_extractor_saver_list[1],
                                               model_dir=self.content_prototype_extractor_dir,
                                               model_name="ContentPrototypeExtractor")
        if self.extractor_style_reference_enabled:
            extr_restored = self.restore_model(saver=feature_extractor_saver_list[2],
                                               model_dir=self.style_reference_extractor_dir,
                                               model_name="StyleReferenceExtractor")






        
        # restore of the model frameworks
        if self.resume_training == 1:
            framework_restored = self.restore_model(saver=saver_framework,
                                                    model_dir=os.path.join(self.checkpoint_dir, 'frameworks'),
                                                    model_name="Frameworks")
            generator_restored = self.restore_model(saver=saver_generator,
                                                    model_dir=os.path.join(self.checkpoint_dir, 'generator'),
                                                    model_name="Generator")
            discriminator_restored = self.restore_model(saver=saver_discriminator,
                                                        model_dir=os.path.join(self.checkpoint_dir, 'discriminator'),
                                                        model_name="Discriminator")

        else:
            print("Framework initialized randomly.")
            print("Generator initialized randomly.")
            print("Discriminator initialized randomly.")
        print(self.print_separater)

    def train_procedures(self):

        if self.debug_mode == 1:
            self.sample_seconds = 5
            self.summary_seconds = 5
            self.record_seconds = 5
            self.print_info_seconds = 5
        else:
            self.summary_seconds = self.print_info_seconds * 1
            self.sample_seconds = self.print_info_seconds * 7
            self.record_seconds = self.print_info_seconds * 9

        with tf.Graph().as_default():
            # tensorflow parameters
            # DO NOT MODIFY!!!
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)


            # define the data set
            data_provider = DataProvider(batch_size=self.batch_size,
                                         epoch=self.epoch,
                                         info_print_interval=self.print_info_seconds / 10,
                                         input_width=self.source_img_width,
                                         input_filters=self.input_output_img_filter_num,
                                         augment_train_data=self.train_data_augment,
                                         style_input_num=self.style_input_number,
                                         content_data_dir=self.content_data_dir,
                                         style_train_data_dir=self.style_train_data_dir,
                                         style_validation_data_dir=self.style_validation_data_dir,
                                         file_list_txt_content=self.file_list_txt_content,
                                         file_list_txt_style_train=self.file_list_txt_style_train,
                                         file_list_txt_style_validation=self.file_list_txt_style_validation)

            self.involved_label0_list, self.involved_label1_list = data_provider.get_involved_label_list()
            self.content_input_num = data_provider.content_input_num
            self.display_style_reference_num = np.min([4, self.style_input_number])
            self.display_content_reference_num = np.min([4, self.content_input_num])

            # ignore
            delete_items=list()
            involved_label_list = self.involved_label1_list
            for ii in self.accuracy_k:
                if ii>len(involved_label_list):
                    delete_items.append(ii)
            for ii in delete_items:
                self.accuracy_k.remove(ii)
            if delete_items and (not self.accuracy_k[len(self.accuracy_k)-1] == len(involved_label_list)):
                self.accuracy_k.append(len(involved_label_list))

            self.train_data_repeat_time = 1
            learning_rate_decay_rate = np.power(self.final_learning_rate_pctg, 1.0 / (self.epoch - 1))

            # define the directory name for model saving location and log saving location
            # delete or create relevant directories
            id, \
            self.checkpoint_dir, \
            self.log_dir, \
            self.inf_data_dir = self.get_model_id_and_dir_for_train()
            if (not self.resume_training == 1) and os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir) # delete!
            if (not self.resume_training == 1) and os.path.exists(self.checkpoint_dir):
                shutil.rmtree(self.checkpoint_dir)
            if (not self.resume_training == 1) and os.path.exists(self.inf_data_dir):
                shutil.rmtree(self.inf_data_dir)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(os.path.join(self.checkpoint_dir, 'discriminator'))
                os.makedirs(os.path.join(self.checkpoint_dir, 'generator'))
                os.makedirs(os.path.join(self.checkpoint_dir, 'frameworks'))
            if not os.path.exists(self.inf_data_dir):
                os.makedirs(self.inf_data_dir)


            #######################################################################################
            #######################################################################################
            #                                model building
            #######################################################################################
            #######################################################################################

            # for framework building
            epoch_step_increase_one_op, \
            learning_rate, \
            global_step, \
            epoch_step, \
            saver_frameworks = self.framework_building()


            # for generator building
            generated_batch_infer, \
            g_loss, g_merged_summary, \
            gen_vars_train, saver_generator,\
            generator_category_loss,generator_category_loss_summary_merged,\
            generator_train_summary, generator_test_summary = self.generator_build()

            # for feature extractor
            g_loss, g_merged_summary, feature_extractor_saver_list, \
            extr_trn_real_merged, extr_trn_fake_merged, extr_val_real_merged, extr_val_fake_merged= \
                self.feature_extractor_build(g_loss=g_loss,
                                             g_merged_summary=g_merged_summary)

            # for discriminator building
            discriminator_batch_c_logits, \
            g_merged_summary, d_merged_summary, \
            g_loss, d_loss, \
            dis_trn_real_summary, dis_trn_fake_summary, dis_val_real_summary, dis_val_fake_summary, \
            dis_vars_train, saver_discriminator =\
                self.discriminator_build(g_loss=g_loss,
                                         g_merged_summary=g_merged_summary,
                                         generator_category_loss=generator_category_loss,
                                         generator_category_loss_summary=generator_category_loss_summary_merged)
            evalHandle = EvalHandle(inferred_generated_images=generated_batch_infer,
                                    inferred_categorical_logits=discriminator_batch_c_logits)
            setattr(self, "eval_handle", evalHandle)


            # # for optimizer creation
            optimizer_g, optimizer_d = \
                self.create_optimizer(learning_rate=learning_rate,
                                      global_step=global_step,
                                      gen_vars_train=gen_vars_train,
                                      generator_loss_train=g_loss,
                                      dis_vars_train=dis_vars_train,
                                      discriminator_loss_train=d_loss)

            trn_real_summary_merged = tf.summary.merge([dis_trn_real_summary, generator_train_summary, extr_trn_real_merged])
            trn_fake_summary_merged = tf.summary.merge([dis_trn_fake_summary, extr_trn_fake_merged])
            val_real_summary_merged = tf.summary.merge([dis_val_real_summary, generator_test_summary, extr_val_real_merged])
            val_fake_summary_merged = tf.summary.merge([dis_val_fake_summary, extr_val_fake_merged])

            # summaries
            self.summary_finalization(g_loss_summary=g_merged_summary,
                                      d_loss_summary=d_merged_summary,
                                      trn_real_summaries=trn_real_summary_merged,
                                      val_real_summaries=val_real_summary_merged,
                                      trn_fake_summaries=trn_fake_summary_merged,
                                      val_fake_summaries=val_fake_summary_merged,
                                      learning_rate=learning_rate)


            # model initialization
            self.model_initialization(saver_framework=saver_frameworks,
                                      saver_generator=saver_generator,
                                      saver_discriminator=saver_discriminator,
                                      feature_extractor_saver_list=feature_extractor_saver_list)
            self.restore_from_previous_model(saver_generator=saver_generator,
                                             saver_discriminator=saver_discriminator)
            print(self.print_separater)
            summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)
            self.previous_highest_arcy_label = list()
            for ii in range(len(self.accuracy_k)):
                self.previous_highest_arcy_label.append(-1)
            self.previous_highest_accuracy_info_list = list()
            self.infer_epochs = 1


            print("%d Threads to read the data" % (data_provider.thread_num))
            print("BatchSize:%d, EpochNum:%d, LearningRateDecay:%.10f Per Epoch"
                  % (self.batch_size, self.epoch, learning_rate_decay_rate))
            print("TrainingSize:%d, ValidateSize:%d, StyleLabel0_Vec:%d, StyleLabel1_Vec:%d" %
                  (len(data_provider.train_iterator.true_style.data_list),
                   len(data_provider.validate_iterator.true_style.data_list),
                   len(self.involved_label0_list),
                   len(self.involved_label1_list)))
            print("ContentLabel0_Vec:%d, ContentLabel1_Vec:%d" % (len(data_provider.content_label0_vec),len(data_provider.content_label1_vec)))
            print("PrintInfo:%ds, Summary:%ds, Sample:%ds, PrintRecord:%ds"%(self.print_info_seconds,
                                                                             self.summary_seconds,
                                                                             self.sample_seconds,
                                                                             self.record_seconds))
            print("ForInitTraining: InvolvedLabel0:%d, InvolvedLabel1:%d" % (len(self.involved_label0_list),
                                                                             len(self.involved_label1_list)))
            print("DataAugment:%d, InputStyleNum:%d" % (self.train_data_augment, self.style_input_number))
            print(self.print_separater)
            print("Penalties:")
            print("Generator: PixelL1:%.3f,ConstCP/SR:%.3f/%.3f,Cat:%.3F,Wgt:%.6f;" % (self.L1_Penalty,
                                                                                       self.Lconst_content_Penalty,
                                                                                       self.Lconst_style_Penalty,
                                                                                       self.Generator_Categorical_Penalty,
                                                                                       self.generator_weight_decay_penalty))
            print("Discriminator: Cat:%.3f,Dis:%.3f,WST-Grdt:%.3f,Wgt:%.6f;" % (self.Discriminator_Categorical_Penalty,
                                                                                self.Discriminative_Penalty,
                                                                                self.Discriminator_Gradient_Penalty,
                                                                                self.discriminator_weight_decay_penalty))
            print("FeatureExtractor: TrueFalse:%.3f, ContentPrototype:%.3f, StyleReference:%.3f;" % (self.Feature_Penalty_True_Fake_Target,
                                                                                                     self.Feature_Penalty_Content_Prototype,
                                                                                                     self.Feature_Penalty_Style_Reference))
            print("InitLearningRate:%.3f" % self.lr)
            print(self.print_separater)
            print("Initialization completed, and training started right now.")


            self.train_implementation(data_provider=data_provider,
                                      summary_writer=summary_writer,
                                      learning_rate_decay_rate=learning_rate_decay_rate, learning_rate=learning_rate,
                                      global_step=global_step, epoch_step_increase_one_op=epoch_step_increase_one_op,
                                      dis_vars_train=dis_vars_train, gen_vars_train=gen_vars_train,
                                      optimizer_d=optimizer_d, optimizer_g=optimizer_g,
                                      saver_discriminator=saver_discriminator, saver_generator=saver_generator,
                                      saver_frameworks=saver_frameworks,
                                      epoch_step=epoch_step)

    def train_implementation(self,
                             data_provider,
                             summary_writer,
                             learning_rate_decay_rate,
                             learning_rate,
                             global_step,
                             epoch_step_increase_one_op,
                             dis_vars_train,
                             gen_vars_train,
                             optimizer_d,
                             optimizer_g,
                             saver_discriminator,saver_generator,saver_frameworks,
                             epoch_step
                             ):
        def W_GAN(current_epoch):
            generator_handle = getattr(self, "generator_handle")
            feature_extractor_handle = getattr(self,"feature_extractor_handle")

            if current_epoch <= self.final_training_epochs:
                self.g_iters = 5
            else:
                self.g_iters = 3

            if global_step.eval(session=self.sess) <= self.g_iters * 5 * self.discriminator_initialization_iters:
                g_iters = self.g_iters * 5
            else:
                g_iters = self.g_iters

            info=""

            reading_data_start = time.time()
            if ((global_step.eval(session=self.sess)) % g_iters == 0
                or global_step.eval(session=self.sess) == global_step_start) and gen_vars_train:
                    batch_packs, batch_label1, batch_label0 = \
                    data_provider.train_iterator.get_next_batch(sess=self.sess)
            else:
                batch_packs, batch_label1, batch_label0 = \
                    data_provider.train_iterator.get_next_batch(sess=self.sess)
            batch_label1_one_hot = self.dense_to_one_hot(input_label=batch_label1,
                                                         involved_label_list=self.involved_label1_list)
            batch_label0_one_hot = self.dense_to_one_hot(input_label=batch_label0,
                                                         involved_label_list=self.involved_label0_list)
            reading_data_consumed = time.time() - reading_data_start


            # optimization for discriminator for all the iterations
            training_img_tensor_list = list()
            training_img_tensor_list.append(generator_handle.generated_target_train)
            if self.extractor_content_prototype_enabled:
                training_img_tensor_list.append(feature_extractor_handle.selected_content_prototype)
            else:
                training_img_tensor_list.append(generator_handle.generated_target_train)
            if self.extractor_style_reference_enabled:
                training_img_tensor_list.append(feature_extractor_handle.selected_style_reference)
            else:
                training_img_tensor_list.append(generator_handle.generated_target_train)

            optimization_start = time.time()
            if dis_vars_train \
                    or global_step.eval(session=self.sess) == global_step_start:

                _ = self.sess.run(optimizer_d,
                                  feed_dict=self.feed_dictionary_generation_for_d(batch_packs=batch_packs,
                                                                                  batch_label0=batch_label0_one_hot,
                                                                                  batch_label1=batch_label1_one_hot,
                                                                                  current_lr=current_lr_real,
                                                                                  learning_rate=learning_rate,
                                                                                  critic_penalty_input=current_critic_logit_penalty_value))

                output_training_img_list = self.sess.run(training_img_tensor_list,
                                                         feed_dict=self.feed_dictionary_generation_for_g(batch_packs=batch_packs,
                                                                                                         batch_label0=batch_label0_one_hot,
                                                                                                         batch_label1=batch_label1_one_hot,
                                                                                                         current_lr=current_lr_real,
                                                                                                         learning_rate=learning_rate))


                info=info+"OptimizeOnD"

            # optimization for generator every (g_iters) iterations
            if ((global_step.eval(session=self.sess)) % g_iters == 0
                or global_step.eval(session=self.sess) == global_step_start + 1) and gen_vars_train:
                _, \
                output_training_img_list \
                    = self.sess.run([optimizer_g,
                                     training_img_tensor_list],
                                    feed_dict=self.feed_dictionary_generation_for_g(batch_packs=batch_packs,
                                                                                    batch_label0=batch_label0_one_hot,
                                                                                    batch_label1=batch_label1_one_hot,
                                                                                    current_lr=current_lr_real,
                                                                                    learning_rate=learning_rate))



                info = info + "&&G"
            optimization_elapsed = time.time() - optimization_start

            return output_training_img_list, \
                   batch_packs,batch_label0_one_hot,batch_label1_one_hot,\
                   reading_data_consumed, optimization_elapsed,\
                   info



        summary_start = time.time()
        sample_start = time.time()
        print_info_start = time.time()
        record_start = time.time()
        training_start_time = time.time()

        

        if self.resume_training==1:
            ei_start = epoch_step.eval(self.sess)
            current_lr = self.lr * np.power(learning_rate_decay_rate, ei_start)
            #current_lr = max(current_lr, 0.00009)
        else:
            ei_start = 0
            current_lr = self.lr
        global_step_start = global_step.eval(session=self.sess)
        print("InitTrainingEpochs:%d, FinalTrainingEpochStartAt:%d" % (self.init_training_epochs,self.final_training_epochs))
        print("TrainingStart:Epoch:%d, GlobalStep:%d, LearnRate:%.5f" % (ei_start+1,global_step_start+1,current_lr))


        if self.debug_mode == 0:
            raw_input("Press enter to continue")
        print(self.print_separater)


        self.found_new_record_on_the_previous_epoch = True
        self.previous_inf_dir = './'
        summary_handle = getattr(self, "summary_handle")
        training_start_time=time.time()

        training_epoch_list = range(ei_start,self.epoch,1)

        for ei in training_epoch_list:

            init_val=False
            if ei==ei_start:
                init_val=True
            data_provider.dataset_reinitialization(sess=self.sess, init_for_val=init_val,
                                                   info_interval=self.print_info_seconds/10)
            self.itrs_for_current_epoch = data_provider.compute_total_batch_num()

            print(self.print_separater)
            print("Epoch:%d/%d with Iters:%d is now commencing" % (ei + 1, self.epoch, self.itrs_for_current_epoch))
            print(self.print_separater)

            if not ei == ei_start:
                update_lr = current_lr * learning_rate_decay_rate
                # update_lr = max(update_lr, 0.00009)
                print("decay learning rate from %.7f to %.7f" % (current_lr, update_lr))
                print(self.print_separater)
                current_lr = update_lr

            for bid in range(self.itrs_for_current_epoch):

                if time.time() - training_start_time <= 600:
                    summary_seconds = 60
                    sample_seconds = 60
                    print_info_seconds = 60
                else:
                    summary_seconds = self.summary_seconds
                    sample_seconds = self.sample_seconds
                    print_info_seconds = self.print_info_seconds
                record_seconds = self.record_seconds


                this_itr_start = time.time()


                if epoch_step.eval(session=self.sess) < self.init_training_epochs:
                    current_critic_logit_penalty_value = (float(global_step.eval(session=self.sess))/float(self.init_training_epochs*self.itrs_for_current_epoch))*self.Discriminative_Penalty
                    current_lr_real = current_lr * 0.1
                else:
                    current_critic_logit_penalty_value = self.Discriminative_Penalty
                    current_lr_real = current_lr

                output_training_img_list,\
                batch_packs_curt_train,batch_label0_one_hot_curt_train,batch_label1_one_hot_curt_train,\
                reading_data_consumed, optimization_consumed, \
                info = W_GAN(current_epoch=epoch_step.eval(session=self.sess))

                passed_full = time.time() - training_start_time
                passed_itr = time.time() - this_itr_start

                if time.time()-print_info_start>print_info_seconds or global_step.eval(session=self.sess)==global_step_start+1:
                    print_info_start = time.time()
                    current_time = time.strftime('%Y-%m-%d@%H:%M:%S', time.localtime())
                    print("Time:%s,Epoch:%d/%d,Itr:%d/%d;" %
                          (current_time,
                           ei + 1, self.epoch,
                           bid + 1, self.itrs_for_current_epoch))


                    print("ItrDuration:%.2fses,FullDuration:%.2fhrs(%.2fdays);" %
                          (passed_itr, passed_full / 3600, passed_full / (3600 * 24)))
                    print("ReadingData:%f,Optimization:%f;" %(reading_data_consumed,
                                                              optimization_consumed))



                    percentage_completed = float(global_step.eval(session=self.sess)) / float(
                        self.epoch * self.itrs_for_current_epoch) * 100
                    percentage_to_be_fulfilled = 100 - percentage_completed
                    hrs_estimated_remaining = (float(passed_full) / (
                            percentage_completed + eps)) * percentage_to_be_fulfilled / 3600
                    print("CompletePctg:%.2f,TimeRemainingEstm:%.2fhrs(%.2fdays)" % (
                        percentage_completed, hrs_estimated_remaining,
                        hrs_estimated_remaining / 24))
                    print("CriticPenalty:%.5f/%.3f;" % (current_critic_logit_penalty_value,
                                                        self.Discriminative_Penalty))



                    print("TrainingInfo:%s" % info)
                    print(self.print_separater)

                if time.time()-record_start>record_seconds or global_step.eval(session=self.sess)==global_step_start+1:
                    record_start=time.time()
                    if not len(self.previous_highest_accuracy_info_list)==0:
                        for info in self.previous_highest_accuracy_info_list:
                            print(info)
                        print(self.print_separater)


                if time.time()-summary_start>summary_seconds or global_step.eval(session=self.sess)==global_step_start+300:
                    summary_start = time.time()

                    if dis_vars_train:
                        batch_packs, batch_label1, batch_label0 = \
                            data_provider.train_iterator.get_next_batch(sess=self.sess)
                        batch_label1_one_hot = self.dense_to_one_hot(input_label=batch_label1,
                                                                     involved_label_list=self.involved_label1_list)
                        batch_label0_one_hot = self.dense_to_one_hot(input_label=batch_label0,
                                                                     involved_label_list=self.involved_label0_list)

                        d_summary = self.sess.run(
                            summary_handle.d_merged,
                            feed_dict=self.feed_dictionary_generation_for_d(batch_packs=batch_packs,
                                                                            batch_label0=batch_label0_one_hot,
                                                                            batch_label1=batch_label1_one_hot,
                                                                            current_lr=current_lr_real,
                                                                            learning_rate=learning_rate,
                                                                            critic_penalty_input=current_critic_logit_penalty_value))
                        summary_writer.add_summary(d_summary, global_step.eval(session=self.sess))
                    if gen_vars_train:
                        batch_packs, batch_label1, batch_label0 = \
                            data_provider.train_iterator.get_next_batch(sess=self.sess)
                        batch_label1_one_hot = self.dense_to_one_hot(input_label=batch_label1,
                                                                     involved_label_list=self.involved_label1_list)
                        batch_label0_one_hot = self.dense_to_one_hot(input_label=batch_label0,
                                                                     involved_label_list=self.involved_label0_list)
                        g_summary = self.sess.run(summary_handle.g_merged,
                                                  feed_dict=self.feed_dictionary_generation_for_g(batch_packs=batch_packs,
                                                                                                  batch_label0=batch_label0_one_hot,
                                                                                                  batch_label1=batch_label1_one_hot,
                                                                                                  current_lr=current_lr_real,
                                                                                                  learning_rate=learning_rate))
                        summary_writer.add_summary(g_summary, global_step.eval(session=self.sess))

                    learning_rate_summary = self.sess.run(summary_handle.learning_rate,
                                                          feed_dict={learning_rate: current_lr_real})
                    summary_writer.add_summary(learning_rate_summary, global_step.eval(session=self.sess))
                    summary_writer.flush()

                if time.time()-sample_start>sample_seconds or global_step.eval(session=self.sess)==global_step_start+300:
                    sample_start = time.time()

                    # check for train set
                    self.validate_model(batch_packs=batch_packs_curt_train,
                                        batch_label0_one_hot=batch_label0_one_hot_curt_train,
                                        batch_label1_one_hot=batch_label1_one_hot_curt_train,
                                        train_mark=True,
                                        training_img_list=output_training_img_list,
                                        summary_writer=summary_writer,
                                        global_step=global_step)

                    # check for validation set
                    batch_packs_val,  batch_label1_val, batch_label0_val = \
                        data_provider.validate_iterator.get_next_batch(sess=self.sess)

                    batch_label1_val_one_hot = self.dense_to_one_hot(input_label=batch_label1_val,
                                                                     involved_label_list=self.involved_label1_list)
                    batch_label0_val_one_hot = self.dense_to_one_hot(input_label=batch_label0_val,
                                                                     involved_label_list=self.involved_label0_list)

                    self.validate_model(batch_packs=batch_packs_val,
                                        batch_label0_one_hot=batch_label0_val_one_hot,
                                        batch_label1_one_hot=batch_label1_val_one_hot,
                                        train_mark=False,
                                        summary_writer=summary_writer,
                                        global_step=global_step)

                    summary_writer.flush()

            # self-increase the epoch number
            self.sess.run(epoch_step_increase_one_op)
            current_time = time.strftime('%Y-%m-%d @ %H:%M:%S', time.localtime())
            print("Time:%s,Checkpoint:SaveCheckpoint@step:%d" % (current_time, global_step.eval(session=self.sess)))
            self.checkpoint(saver=saver_discriminator,
                            model_dir=os.path.join(self.checkpoint_dir, 'discriminator'),
                            global_step=global_step)
            self.checkpoint(saver=saver_generator,
                            model_dir=os.path.join(self.checkpoint_dir, 'generator'),
                            global_step=global_step)
            self.checkpoint(saver=saver_frameworks,
                            model_dir=os.path.join(self.checkpoint_dir, 'frameworks'),
                            global_step=global_step)
            print(self.print_separater)


        print("Training Completed.")



    def infer_network_g2d_for_train(self,
                                    data_provider,
                                    save_image_path,
                                    ei,
                                    saver_discriminator, saver_generator,saver_frameworks,
                                    global_step):
        def top_k_accuracy(logits,true_label,label_vec,k):
            top_k_indices = np.argsort(-logits,axis=1)[:,0:k]
            for ii in range(k):
                estm_label = label_vec[top_k_indices[:,ii]]
                diff = np.abs(estm_label - true_label)
                if ii==0:
                    full_diff=np.reshape(diff,[diff.shape[0],1])
                else:
                    full_diff=np.concatenate([full_diff,np.reshape(diff,[diff.shape[0],1])],axis=1)
            top_k_accuracy_list=list()
            for ii in range(len(self.accuracy_k)):
                this_k = self.accuracy_k[ii]
                this_k_diff = full_diff[:,0:this_k]
                if this_k==0:
                    this_k_diff=np.reshape(this_k_diff,[this_k_diff.shape[0],1])
                min_v = np.min(this_k_diff,axis=1)
                correct = [i for i, v in enumerate(min_v) if v == 0]
                accuracy = float(len(correct)) / float(len(true_label)) * 100
                top_k_accuracy_list.append(accuracy)
            return top_k_accuracy_list


        def TDR(parafix):


            label_vec = np.array(self.involved_label1_list)
            print_info_start=time.time()
            evalHandle = getattr(self, "eval_handle")

            iter_num = len(data_provider.validate_iterator.data_list) / (self.batch_size) + 1

            start_time = time.time()
            print(self.print_separater)
            full_category_logits = np.zeros([iter_num*self.batch_size,len(label_vec)])
            true_label = np.zeros([iter_num*self.batch_size])
            counter=0
            for bid in range(iter_num):
                iter_start_time = time.time()

                batch_packs,  batch_label1, batch_label0 = \
                    data_provider.validate_iterator.get_next_batch(sess=self.sess)


                label_disc = batch_label1

                generate_start=time.time()
                channels = self.input_output_img_filter_num
                source_batch = batch_packs[:, :, :, 0:channels]
                input_batch = batch_packs[:, :, :, channels * 2:]
                generated_target_imgs = self.generate_fake_samples(source=source_batch,
                                                                   input_target=input_batch)
                generate_elapsed = time.time()-generate_start

                discriminate_start = time.time()
                category_logits = self.evaluate_samples(source=source_batch,
                                                        target=generated_target_imgs,
                                                        input_style=input_batch)
                discriminate_elapsed = time.time()-discriminate_start

                get_logit_start = time.time()

                full_category_logits[bid*self.batch_size:(bid+1)*self.batch_size,:] = category_logits
                true_label[bid * self.batch_size:(bid + 1) * self.batch_size] = label_disc


                get_logits_elapsed = time.time()-get_logit_start
                counter=counter+self.batch_size

                if time.time() - print_info_start > self.print_info_seconds or bid == 0 or bid == iter_num-1:
                    print_info_start = time.time()
                    full_estm_indices = np.argmax(full_category_logits[0:counter,:], axis=1)
                    full_estm_label = label_vec[full_estm_indices]
                    full_diff = true_label[0:counter] - full_estm_label
                    full_correct = [i for i, v in enumerate(full_diff) if v == 0]
                    full_arcy_label = float(len(full_correct)) / float(counter) * 100

                    print("EvlEpc:%d,Itr:%d/%d,Elps:%.2fsecs/%.2fmins,Smp:%d/%d,Acry:%.3f" %
                          (ei, bid + 1, iter_num, time.time() - iter_start_time, float(time.time() - start_time) / 60,
                           bid * self.batch_size+1,
                           len(data_provider.validate_iterator.data_list),
                           full_arcy_label))

                    print("G:%.3f,D:%.3f,Lgt:%.3f" %
                          (generate_elapsed,
                           discriminate_elapsed,
                           get_logits_elapsed))

                    print(self.print_separater)


            full_category_logits = full_category_logits[0:len(data_provider.validate_iterator.data_list), :]

            label_estm_indices = np.argmax(full_category_logits,axis=1)
            label_estm = label_vec[label_estm_indices]

            true_label = true_label[0:len(data_provider.validate_iterator.data_list)]
            top_k_acry_list = top_k_accuracy(logits=full_category_logits,
                                             true_label=true_label, label_vec=label_vec, k=self.accuracy_k[len(self.accuracy_k)-1])

            return full_category_logits, label_estm, top_k_acry_list



        if (not self.found_new_record_on_the_previous_epoch):
            shutil.rmtree(self.previous_inf_dir)
            print("Removed:%s" % self.previous_inf_dir)

        full_category_logits, \
        label0_estm, label0_acry_list \
            = TDR(parafix='_validation.txt')



        print(self.print_separater)
        print("Crrt:Top@",end="")
        curt_acry_info = "Top@"
        for ii in self.accuracy_k:
            if not ii == self.accuracy_k[len(self.accuracy_k)-1]:
                curt_acry_info = curt_acry_info + ("%d_" % ii)
                print("%d/" % ii, end="")
            else:
                curt_acry_info = curt_acry_info + ("%d@" % ii)
                print("%d:" % ii, end="")
        for ii in label0_acry_list:
            if not ii == label0_acry_list[len(label0_acry_list)-1]:
                curt_acry_info=curt_acry_info+("%.3f_" % ii)
                print("%.3f/" % ii, end="")
            else:
                curt_acry_info = curt_acry_info + ("%.3f" % ii)
                print("%.3f;" % ii, end="")
        print("")

        print("Prvs:Top@", end="")
        prev_acry_info="Top@"
        for ii in self.accuracy_k:
            if not ii == self.accuracy_k[len(self.accuracy_k)-1]:
                prev_acry_info = prev_acry_info + ("%d_" % ii)
                print("%d/" % ii, end="")
            else:
                prev_acry_info = prev_acry_info + ("%d@" % ii)
                print("%d:" % ii, end="")
        for ii in self.previous_highest_arcy_label:
            if not ii == label0_acry_list[len(label0_acry_list)-1]:
                prev_acry_info=prev_acry_info+("%.3f_" % ii)
                print("%.3f/" % ii, end="")
            else:
                prev_acry_info = prev_acry_info + ("%.3f" % ii)
                print("%.3f;" % ii, end="")
        print("")

        print(self.print_separater)

        save_image_path = os.path.join(save_image_path, curt_acry_info)
        if os.path.exists(save_image_path):
            shutil.rmtree(save_image_path)
        os.makedirs(save_image_path)
        write_image_list_path = (os.path.join(save_image_path, curt_acry_info+".txt"))
        self.previous_inf_dir = save_image_path



        prev_acry = self.previous_highest_arcy_label
        found_new_record=False
        for ii in range(len(prev_acry)):
            if prev_acry[ii]<label0_acry_list[ii]:
                found_new_record=True
                break


        if found_new_record:
            if found_new_record:
                self.previous_highest_arcy_label = label0_acry_list

            self.found_new_record_on_the_previous_epoch = True

            current_accuracy_info0 = time.strftime('%Y-%m-%d@%H:%M:%S', time.localtime())
            current_accuracy_info1 = current_accuracy_info0 + (",Epc:%d,Rnd:%d:" %
                                                               (ei,len(self.previous_highest_accuracy_info_list)/3 + 1))
            current_accuracy_info2 = ('CurtHghstAcry:%s' % self.previous_highest_arcy_label)
            self.previous_highest_accuracy_info_list.append(current_accuracy_info1)
            self.previous_highest_accuracy_info_list.append(current_accuracy_info2)
            self.previous_highest_accuracy_info_list.append(self.print_separater)
            current_time = time.strftime('%Y-%m-%d @ %H:%M:%S', time.localtime())
            print("Time:%s,Checkpoint:SaveCheckpoint@step:%d" % (current_time, global_step.eval(session=self.sess)))

            self.checkpoint(saver=saver_discriminator,
                            model_dir=os.path.join(self.checkpoint_dir, 'discriminator'),
                            global_step=global_step)
            self.checkpoint(saver=saver_generator,
                            model_dir=os.path.join(self.checkpoint_dir, 'generator'),
                            global_step=global_step)
            self.checkpoint(saver=saver_frameworks,
                            model_dir=os.path.join(self.checkpoint_dir, 'frameworks'),
                            global_step=global_step)
            print(self.print_separater)
        else:
            self.found_new_record_on_the_previous_epoch = False


        return write_image_list_path


    def dense_to_one_hot(self,
                         input_label,
                         involved_label_list):

        # for abnormal label process:
        # for those labels in the test list not occurring in the training
        abnormal_marker_indices = list()
        for ii in range(len(input_label)):
            if not input_label[ii] in involved_label_list:
                abnormal_marker_indices.append(ii)
        data_indices = np.arange(len(input_label))
        data_indices_list = data_indices.tolist()
        for ii in abnormal_marker_indices:
            data_indices_list.remove(data_indices[ii])
        data_indices = np.array(data_indices_list)

        label_length = len(involved_label_list)
        input_label_matrix = np.tile(np.asarray(input_label), [len(involved_label_list), 1])
        fine_tune_martix = np.transpose(np.tile(involved_label_list, [self.batch_size, 1]))
        # print(involved_label_list)
        # print(input_label)


        diff = input_label_matrix - fine_tune_martix
        find_positions = np.argwhere(np.transpose(diff) == 0)
        input_label_indices = np.transpose(find_positions[:, 1:]).tolist()

        output_one_hot_label = np.zeros((len(input_label), label_length), dtype=np.float32)
        if data_indices.tolist():
            output_one_hot_label[data_indices, input_label_indices] = 1
        return output_one_hot_label



    def feed_dictionary_generation_for_d(self,
                                         critic_penalty_input,
                                         batch_packs,
                                         batch_label0,batch_label1,
                                         current_lr,
                                         learning_rate):

        channels = self.input_output_img_filter_num
        true_style = batch_packs[:, :, :,0:channels]
        content_prototypes = batch_packs[:, :, :, channels:channels+self.content_input_num*channels]
        style_references = batch_packs[:, :, :, 1+self.content_input_num*channels:]
        rnd_index_style = rnd.sample(range(self.style_input_number), 1)[0]
        rnd_index_content = rnd.sample(range(self.content_input_num),1)[0]
        output_dict = {}

        generator_handle = getattr(self, "generator_handle")
        discriminator_handle = getattr(self, "discriminator_handle")



        output_dict.update({discriminator_handle.current_critic_logit_penalty:critic_penalty_input})
        output_dict.update({discriminator_handle.true_style: true_style})
        output_dict.update({discriminator_handle.content_prototype: np.reshape(content_prototypes[:,:,:, rnd_index_content],
                                                                               [self.batch_size,
                                                                                self.img2img_width,
                                                                                self.img2img_width,1])})
        output_dict.update({discriminator_handle.style_reference: np.reshape(style_references[:,:,:, rnd_index_style],
                                                                             [self.batch_size,
                                                                              self.img2img_width,
                                                                              self.img2img_width,1])})
        output_dict.update({discriminator_handle.discriminator_true_label: batch_label1})

        output_dict.update({generator_handle.true_style: true_style})
        output_dict.update({generator_handle.true_label0: batch_label0})
        output_dict.update({generator_handle.true_label1: batch_label1})
        for ii in range(self.style_input_number):
            output_dict.update({generator_handle.style_reference_list[ii]:
                                    np.reshape(style_references[:, :, :, ii],
                                               [self.batch_size,
                                                self.img2img_width,
                                                self.img2img_width,1])})
        output_dict.update({generator_handle.content_prototype:content_prototypes})


        output_dict.update({learning_rate: current_lr})
        return output_dict

    def feed_dictionary_generation_for_g(self,
                                         batch_packs,
                                         batch_label0,batch_label1,
                                         current_lr,
                                         learning_rate):

        channels = self.input_output_img_filter_num
        true_style = batch_packs[:, :, :, 0:channels]
        content_prototypes = batch_packs[:, :, :, channels:channels + self.content_input_num * channels]
        style_references = batch_packs[:, :, :, 1 + self.content_input_num * channels:]
        rnd_index_style = rnd.sample(range(self.style_input_number), 1)[0]
        rnd_index_content = rnd.sample(range(self.content_input_num), 1)[0]
        output_dict = {}


        generator_handle = getattr(self, "generator_handle")
        discriminator_handle = getattr(self, "discriminator_handle")

        output_dict.update({generator_handle.true_style: true_style})
        output_dict.update({generator_handle.true_label0: batch_label0})
        output_dict.update({generator_handle.true_label1: batch_label1})
        for ii in range(self.style_input_number):
            output_dict.update({generator_handle.style_reference_list[ii]: np.reshape(style_references[:, :, :, ii],
                                                                                      [self.batch_size,
                                                                                       self.img2img_width,
                                                                                       self.img2img_width,1])})
        output_dict.update({generator_handle.content_prototype:content_prototypes})

        output_dict.update({discriminator_handle.discriminator_true_label:batch_label1})
        output_dict.update({discriminator_handle.true_style: true_style})
        output_dict.update({discriminator_handle.content_prototype:
                                np.reshape(content_prototypes[:, :, :, rnd_index_content],
                                           [self.batch_size,
                                            self.img2img_width,
                                            self.img2img_width, 1])})
        output_dict.update({discriminator_handle.style_reference:
                                np.reshape(style_references[:, :, :, rnd_index_style],
                                           [self.batch_size,
                                            self.img2img_width,
                                            self.img2img_width, 1])})

        output_dict.update({learning_rate: current_lr})

        return output_dict
