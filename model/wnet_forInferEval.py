# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

GRAYSCALE_AVG = 127.5

import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import shutil
from collections import namedtuple
import re

from utilities.utils import correct_ckpt_path
from utilities.utils import image_show

import time


from model.generators import WNet_Generator as wnet_generator
from model.generators import EmdNet_Generator as emdnet_generator
from model.generators import ResEmd_EmdNet_Generator as resemdnet_generator
from model.generators import AdobeNet_Generator as adobenet_generator
from model.generators import ResMixerNet_Generator as resmixernet_generator


label0_split_train_test_label = 3000


import math
import utilities.infer_implementations as inf_tools



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
                             ["generated_target_train","generated_target_infer"])

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
eps = 1e-9

class WNet(object):

    # constructor
    def __init__(self,
                 print_info_seconds=-1,
                 style_input_number=-1,

                 experiment_id='0',

                 content_data_dir='/tmp/',
                 file_list_txt_content=None,

                 style_data_dir='/tmp/',
                 file_list_txt_true_style=None,
                 file_list_txt_input_style=None,

                 img_width=64,
                 channels=1,



                 generator_devices='/device:CPU:0',

                 generator_residual_at_layer=3,
                 generator_residual_blocks=5,

                 ## for infer only
                 model_dir='./', save_path='./',

                 targeted_content_input_txt='./',
                 targeted_style_input_txt='./',
                 known_style_img_path='./',


                 ):

        self.print_separater = "#############################################################################"

        self.initializer = 'XavierInit'
        self.style_input_number=style_input_number

        self.experiment_id = experiment_id
        #self.evaluation_resule_save_dir=evaluation_resule_save_dir

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
                if 'NN' in experiment_id:
                    self.other_info = 'NN'
            else:
                self.generator_implementation = generator_dict['EmdNet']
        elif 'WNet' in experiment_id:
            self.generator_implementation = generator_dict['WNet']
            self.generator_residual_at_layer = generator_residual_at_layer
            self.generator_residual_blocks = generator_residual_blocks
            if 'DenseMixer' in experiment_id:
                self.other_info='DenseMixer'
            elif 'ResidualMixer' in experiment_id:
                self.other_info='ResidualMixer'
        elif 'Adobe' in experiment_id:
            self.generator_implementation = generator_dict['AdobeNet']
        elif 'ResMixer' in experiment_id:
            self.generator_implementation = generator_dict['ResMixerNet']
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


        self.print_info_seconds=print_info_seconds


        self.img2img_width = img_width
        self.source_img_width = img_width


        self.content_data_dir = content_data_dir
        self.style_data_dir = style_data_dir
        self.file_list_txt_content = file_list_txt_content
        self.file_list_txt_true_style = file_list_txt_true_style
        self.file_list_txt_input_style = file_list_txt_input_style
        self.input_output_img_filter_num = channels



        self.generator_devices = generator_devices



        # properties for inferring
        self.model_dir=model_dir
        self.save_path=save_path
        if os.path.exists(self.save_path) and not (self.save_path == './'):
            shutil.rmtree(self.save_path)
        if not self.save_path == './':
            os.makedirs(self.save_path)

        # for styleadd infer only
        self.targeted_content_input_txt=targeted_content_input_txt
        self.targeted_style_input_txt = targeted_style_input_txt
        self.known_style_img_path=known_style_img_path


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


    def character_generation(self):

        def _feed(content, style):
            return_dict = {}
            return_dict.update({content_prototype_placeholder:content})
            for ii in range(style.shape[0]):
                return_dict.update({style_reference_placeholder_list[ii]: np.expand_dims(style[ii,:,:,:],axis=0)})
            return return_dict


        charset_level1, character_label_level1 = \
            inf_tools.get_chars_set_from_level1_2(path='../ContentTxt/GB2312_Level_1.txt',
                                                  level=1)
        charset_level2, character_label_level2 = \
            inf_tools.get_chars_set_from_level1_2(path='../ContentTxt/GB2312_Level_2.txt',
                                                  level=2)

        print(self.print_separater)
        content_prototype, content_label1_vec, valid_mark, content_char_list, content_char_label0_list = \
            inf_tools.get_revelant_data(targeted_input_txt=self.targeted_content_input_txt,
                                        level1_charlist=charset_level1,
                                        level2_charlist=charset_level2,
                                        level1_labellist=character_label_level1,
                                        level2_labellist=character_label_level2,
                                        file_list_txt=self.file_list_txt_content,
                                        file_data_dir=self.content_data_dir,
                                        img_width=self.img2img_width,
                                        img_filters=self.input_output_img_filter_num,
                                        info='ContentPrototype')
        true_style_reference, true_style_label1_vec, valid_mark, _, true_style_char_label0_list = \
            inf_tools.get_revelant_data(targeted_input_txt=self.targeted_content_input_txt,
                                        level1_charlist=charset_level1,
                                        level2_charlist=charset_level2,
                                        level1_labellist=character_label_level1,
                                        level2_labellist=character_label_level2,
                                        file_list_txt=self.file_list_txt_true_style,
                                        file_data_dir=self.style_data_dir,
                                        img_width=self.img2img_width,
                                        img_filters=self.input_output_img_filter_num,
                                        info='True StyleReference')
        input_style_reference, input_style_label1_vec, valid_mark, style_char_list, input_style_char_label0_list = \
            inf_tools.get_revelant_data(targeted_input_txt=self.targeted_style_input_txt,
                                        level1_charlist=charset_level1,
                                        level2_charlist=charset_level2,
                                        level1_labellist=character_label_level1,
                                        level2_labellist=character_label_level2,
                                        file_list_txt=self.file_list_txt_input_style,
                                        file_data_dir=self.style_data_dir,
                                        img_width=self.img2img_width,
                                        img_filters=self.input_output_img_filter_num,
                                        info='Input StyleReference')



        if not valid_mark:
            print("Generation Terminated.")





        with tf.Graph().as_default():
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            # build style reference encoder
            with tf.device(self.generator_devices):



                style_reference_placeholder_list = list()
                for ii in range(self.style_input_number):
                    current_style_reference_placeholder = tf.placeholder(tf.float32, [1, self.source_img_width,
                                                                                      self.source_img_width,
                                                                                      1],
                                                                         name='placeholder_style_reference')
                    style_reference_placeholder_list.append(current_style_reference_placeholder)


                content_prototype_placeholder = tf.placeholder(tf.float32,
                                                               [1,
                                                                self.source_img_width,
                                                                self.source_img_width,
                                                                content_prototype.shape[3]],
                                                               name='placeholder_content_prototype')

                generated_infer,_,_,_,_,_,content_full_faeture_list,style_full_feature_list,decoded_feature_list = \
                    self.generator_implementation(content_prototype=content_prototype_placeholder,
                                                  style_reference=style_reference_placeholder_list,
                                                  is_training=False,
                                                  batch_size=1,
                                                  generator_device=self.generator_devices,
                                                  residual_at_layer=self.generator_residual_at_layer,
                                                  residual_block_num=self.generator_residual_blocks,
                                                  scope='generator',
                                                  initializer=self.initializer,
                                                  weight_decay=False, weight_decay_rate=0,
                                                  reuse=False,
                                                  adain_use=self.adain_use,
                                                  adain_preparation_model=self.adain_preparation_model,
                                                  debug_mode=False,
                                                  other_info=self.other_info)

                gen_vars_save = self.find_norm_avg_var([var for var in tf.trainable_variables() if 'generator' in var.name])



            # model load
            saver_generator = tf.train.Saver(max_to_keep=1, var_list=gen_vars_save)
            generator_restored = self.restore_model(saver=saver_generator,
                                                    model_dir=self.model_dir,
                                                    model_name='Generator')

            if not generator_restored:
                return
            print(self.print_separater)

        # generating characters
        content_prototype_num = content_prototype.shape[3]
        styles_number_to_be_generated = input_style_reference.shape[3]
        content_number_to_be_generated = content_prototype.shape[0]
        round_num_per_style = input_style_reference.shape[0] / self.style_input_number

        content_save_path = os.path.join(self.save_path, 'Contents')
        if not os.path.exists(content_save_path):
            os.makedirs(content_save_path)
        for content_prototype_counter in range(content_prototype_num):
            full_content = np.zeros(dtype=np.float32,
                                    shape=[content_number_to_be_generated,
                                           self.img2img_width, self.img2img_width,
                                           1])
            current_content_prototype = content_prototype[:, :, :, content_prototype_counter]
            current_content_prototype = np.expand_dims(current_content_prototype, axis=3)
            full_content[:,:,:,:] = current_content_prototype

            for content_counter in range(content_number_to_be_generated):
                current_content_char_label0 = content_char_label0_list[content_counter]
                current_label0_1 = current_content_char_label0[0:3]
                current_label0_2 = current_content_char_label0[3:6]
                current_label0_id = (int(current_label0_1) - 160 - 16) * 94 + (int(current_label0_2) - 160 - 1)
                if not current_label0_id < label0_split_train_test_label:
                    full_content[content_counter,:,:,:] = 1-full_content[content_counter,:,:,:]

            content_paper = inf_tools.matrix_paper_generation(images=full_content,
                                                              rows=content_number_to_be_generated,
                                                              columns=1)

            misc.imsave(os.path.join(content_save_path, 'Content%s.png' % content_label1_vec[content_prototype_counter]), content_paper)



        timer_start = time.time()
        for style_counter in range(styles_number_to_be_generated):
            local_timer_start = time.time()
            for round_counter in range(round_num_per_style):

                current_style_char = style_char_list[round_counter * self.style_input_number:(round_counter + 1) * self.style_input_number]
                current_style_char_label0 = input_style_char_label0_list[round_counter * self.style_input_number:(round_counter + 1) * self.style_input_number]
                dir_name_str = '_'
                for tmp in current_style_char_label0:
                    dir_name_str = dir_name_str + tmp
                    if not tmp == current_style_char_label0[-1]:
                        dir_name_str = dir_name_str + '_'


                current_round_save_dir = os.path.join(self.save_path,'Round%02d%s' % (round_counter+1,dir_name_str))


                full_generated = np.zeros(dtype=np.float32,
                                          shape=[content_prototype.shape[0],
                                                 self.img2img_width, self.img2img_width,
                                                 self.input_output_img_filter_num])
                full_true_style = np.zeros(dtype=np.float32,
                                           shape=[content_prototype.shape[0],
                                                  self.img2img_width, self.img2img_width,
                                                  self.input_output_img_filter_num])
                current_input_style = \
                    input_style_reference[round_counter*self.style_input_number:(round_counter+1)*self.style_input_number,:,:,style_counter]
                current_input_style = np.expand_dims(current_input_style,axis=3)

                for content_counter in range(content_number_to_be_generated):
                    current_content_char = np.expand_dims(np.squeeze(content_prototype[content_counter, :, :, :]),
                                                          axis=0)
                    if current_content_char.ndim==3:
                        current_content_char = np.expand_dims(current_content_char, axis=3)
                    current_generated_char = self.sess.run(generated_infer,
                                                           feed_dict=_feed(content=current_content_char,
                                                                           style=current_input_style))
                    current_true_char = true_style_reference[content_counter, :, :, style_counter]
                    current_true_char = np.expand_dims(np.expand_dims(current_true_char, axis=0), axis=3)


                    current_content_char = content_char_list[content_counter]
                    current_content_char_label0 = content_char_label0_list[content_counter]
                    current_label0_1 = current_content_char_label0[0:3]
                    current_label0_2 = current_content_char_label0[3:6]
                    current_label0_id = (int(current_label0_1) - 160 - 16) * 94 + (int(current_label0_2) - 160 - 1)
                    if not current_label0_id<label0_split_train_test_label:
                        # for test characters
                        current_generated_char = 1-current_generated_char
                        current_true_char=1-current_true_char

                    full_generated[content_counter, :, :, :] = current_generated_char
                    full_true_style[content_counter,:,:,:] = current_true_char

                    diff = np.mean(np.abs(full_generated-full_true_style))

                generated_paper = inf_tools.matrix_paper_generation(images=full_generated,
                                                                    rows=content_number_to_be_generated,
                                                                    columns=1)
                true_style_paper = inf_tools.matrix_paper_generation(images=full_true_style,
                                                                     rows=content_number_to_be_generated,
                                                                     columns=1)

                current_style_label = input_style_label1_vec[style_counter]
                if len(current_style_label)==1:
                    current_style_label = '0' + current_style_label
                if not os.path.exists(current_round_save_dir):
                    os.makedirs(current_round_save_dir)
                misc.imsave(os.path.join(current_round_save_dir, 'Diff@%.9f_Style%s_GeneratedStyle.png'
                                         % (diff,current_style_label)),
                            generated_paper)
                misc.imsave(os.path.join(current_round_save_dir, 'Diff@%.9f_Style%s_TrueStyle.png'
                                         % (diff, current_style_label)),
                            true_style_paper)
                # misc.imsave(os.path.join(current_round_save_dir, 'Style%s_GeneratedStyle.png'
                #                          % (current_style_label)),
                #             generated_paper)
                # misc.imsave(os.path.join(current_round_save_dir, 'Style%s_TrueStyle.png'
                #                          % (current_style_label)),
                #             true_style_paper)

            time_elapsed = time.time()-timer_start
            local_time_elapsed = time.time()-local_timer_start
            avg_elaped_per_style = time_elapsed / (style_counter+1)

            time_estimated_remain = avg_elaped_per_style * styles_number_to_be_generated - time_elapsed
            print("CurrentProcess: Style:%d/%d(%s), CurrentStyle/Avg:%.3f/%.3fsecs, TimerRemain:%.3fmins"
                  %(style_counter+1, styles_number_to_be_generated, input_style_label1_vec[style_counter],
                    local_time_elapsed, avg_elaped_per_style, time_estimated_remain/60))


        print(self.print_separater)
        print("Generated Completed")


