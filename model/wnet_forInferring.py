# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

GRAYSCALE_AVG = 127.5
TINIEST_LR = 0.00005

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




from model.generators import WNet_Generator as wnet_generator
from model.generators import EmdNet_Generator as emdnet_generator
from model.generators import ResEmd_EmdNet_Generator as resemdnet_generator
from model.generators import AdobeNet_Generator as adobenet_generator
from model.generators import ResMixerNet_Generator as resmixernet_generator




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
                 adain_use=-1,

                 experiment_id='0',
                 content_data_dir='/tmp/',
                 style_train_data_dir='/tmp/',
                 style_validation_data_dir='/tmp/',
                 file_list_txt_content=None,
                 file_list_txt_style_train=None,
                 file_list_txt_style_validation=None,
                 channels=-1,


                 img_width=64,



                 generator_devices='/device:CPU:0',

                 generator_residual_at_layer=3,
                 generator_residual_blocks=5,

                 ## for infer only
                 model_dir='./', save_path='./',

                 targeted_content_input_txt='./',
                 save_mode='-1',
                 known_style_img_path='./',


                 ):

        self.print_separater = "#################################################################"

        self.initializer = 'XavierInit'
        self.style_input_number=style_input_number

        self.experiment_id = experiment_id

        self.adain_mark = adain_use
        self.adain_use = ('1' in adain_use)
        if adain_use and 'Multi' in adain_use:
            self.adain_preparation_model = 'Multi'
        elif adain_use and 'Single' in adain_use:
            self.adain_preparation_model = 'Single'
        else:
            self.adain_preparation_model = None

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
        self.style_train_data_dir = style_train_data_dir
        self.style_validation_data_dir = style_validation_data_dir
        self.file_list_txt_content = file_list_txt_content
        self.file_list_txt_style_train = file_list_txt_style_train
        self.file_list_txt_style_validation = file_list_txt_style_validation
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
        self.save_mode=save_mode
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
            for ii in range(style.shape[3]):
                return_dict.update({style_reference_placeholder_list[ii]: np.expand_dims(style[:,:,:,ii],axis=3)})
            return return_dict


        charset_level1, character_label_level1 = \
            inf_tools.get_chars_set_from_level1_2(path='../ContentTxt/GB2312_Level_1.txt',
                                                  level=1)
        charset_level2, character_label_level2 = \
            inf_tools.get_chars_set_from_level1_2(path='../ContentTxt/GB2312_Level_2.txt',
                                                  level=2)

        # get data available
        # tmp = list()
        # tmp.append('/DataA/Harric/ChineseCharacterExp/CASIA_Dataset/Sources/PrintedSources/64_FoundContentPrototypeTtfOtfs/Simplified/')
        # content_prototype1, content_label1_vec1, valid_mark1 = \
        #     inf_tools.get_prototype_on_targeted_content_input_txt(
        #         targeted_content_input_txt=self.targeted_content_input_txt,
        #         level1_charlist=charset_level1,
        #         level2_charlist=charset_level2,
        #         level1_labellist=character_label_level1,
        #         level2_labellist=character_label_level2,
        #         content_file_list_txt=self.file_list_txt_content,
        #         content_file_data_dir=tmp,
        #         img_width=self.img2img_width,
        #         img_filters=self.input_output_img_filter_num)

        print(self.print_separater)
        content_prototype, content_label1_vec, valid_mark = \
            inf_tools.get_prototype_on_targeted_content_input_txt(targeted_content_input_txt=self.targeted_content_input_txt,
                                                                  level1_charlist=charset_level1,
                                                                  level2_charlist=charset_level2,
                                                                  level1_labellist=character_label_level1,
                                                                  level2_labellist=character_label_level2,
                                                                  content_file_list_txt=self.file_list_txt_content,
                                                                  content_file_data_dir=self.content_data_dir,
                                                                  img_width=self.img2img_width,
                                                                  img_filters=self.input_output_img_filter_num)




        if not valid_mark:
            print("Generation Terminated.")

        style_reference = inf_tools.get_style_references(img_path=self.known_style_img_path,
                                                         resave_path=self.save_path,
                                                         style_input_number=self.style_input_number)

        output_paper_shape = self.save_mode.split(':')
        output_paper_rows = int(output_paper_shape[0])
        output_paper_cols = int(output_paper_shape[1])
        if output_paper_rows * output_paper_cols < content_prototype.shape[0]:
            print('Incorrect Paper Size !@!~!@~!~@~!@~')
            return



        with tf.Graph().as_default():
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            # build style reference encoder
            with tf.device(self.generator_devices):



                style_reference_placeholder_list = list()
                for ii in range(style_reference.shape[3]):
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
        full_content = np.zeros(dtype=np.float32,
                                shape=[content_prototype.shape[0],
                                       self.img2img_width,self.img2img_width,
                                       content_prototype.shape[3]])
        full_generated = np.zeros(dtype=np.float32,
                                  shape=[content_prototype.shape[0],
                                         self.img2img_width,self.img2img_width,
                                         self.input_output_img_filter_num])
        current_counter=0
        for current_generating_content_counter in range(content_prototype.shape[0]):
            current_content_char = np.expand_dims(np.squeeze(content_prototype[current_generating_content_counter,:,:,:]),axis=0)
            if current_content_char.ndim==3:
                current_content_char = np.expand_dims(current_content_char,axis=3)
            current_generated_char, \
            style_full_features,content_full_features,decoded_full_features = \
                self.sess.run([generated_infer,
                               style_full_feature_list,content_full_faeture_list,decoded_feature_list],
                              feed_dict=_feed(content=current_content_char,
                                              style=style_reference))




            # feature_saving_path = '/Users/harric/Desktop/Features'
            #
            # style_path = os.path.join(feature_saving_path,'Style')
            # for ii in range(len(style_full_features)):
            #     current_style_path = 'StyleNo%d' % ii
            #     current_style_path = os.path.join(style_path, current_style_path)
            #     if os.path.exists(current_style_path):
            #         shutil.rmtree(current_style_path)
            #     os.makedirs(current_style_path)
            #     for jj in range(len(style_full_features[ii])):
            #         if style_full_features[ii][jj].shape[1] > 1:
            #             current_style_current_feature_level_path = 'LevelNo%d' % jj
            #             current_style_current_feature_level_path = os.path.join(current_style_path, current_style_current_feature_level_path)
            #             if os.path.exists(current_style_current_feature_level_path):
            #                 shutil.rmtree(current_style_current_feature_level_path)
            #             os.makedirs(current_style_current_feature_level_path)
            #
            #             for kk in range(style_full_features[ii][jj].shape[3]):
            #
            #                 inf_tools.numpy_img_save(img=image_revalue(style_full_features[ii][jj][:, :, :, kk],
            #                                                            tah_mark=True),
            #                                          path=os.path.join(current_style_current_feature_level_path,
            #                                                            'Style%dFeatureLevel%dChannel%d.png'
            #                                                            % (ii,jj,kk)))
            #     inf_tools.numpy_img_save(img=image_revalue(style_reference[:,:,:,ii], tah_mark=False),
            #                              path=os.path.join(current_style_path,
            #                                                'StyleRef%d.png' % ii))
            #
            # content_path = os.path.join(feature_saving_path,'Content')
            # for ii in range(len(content_full_features)):
            #     if content_full_features[ii].shape[1] > 1:
            #         current_content_path = 'ContentFeatureLevel%d' % ii
            #         current_content_path = os.path.join(content_path,current_content_path)
            #         if os.path.exists(current_content_path):
            #             shutil.rmtree(current_content_path)
            #         os.makedirs(current_content_path)
            #         for jj in range(content_full_features[ii].shape[3]):
            #             inf_tools.numpy_img_save(img=image_revalue(content_full_features[ii][:, :, :, jj],tah_mark=True),
            #                                      path=os.path.join(current_content_path,
            #                                                        'ContentFeatureLevel%dChannel%d.png'
            #                                                        % (ii, jj)))
            #
            # content_path = os.path.join(content_path,'ContentPrototype')
            # if os.path.exists(content_path):
            #     shutil.rmtree(content_path)
            # os.makedirs(content_path)
            # for ii in range(current_content_char.shape[3]):
            #     inf_tools.numpy_img_save(img=image_revalue(current_content_char[:, :, :, ii],tah_mark=False),
            #                              path=os.path.join(content_path,
            #                                                'ContentPro%d.png.png'
            #                                                % ii))
            #
            # decoded_path = os.path.join(feature_saving_path,'Decoder')
            # for ii in range(len(decoded_full_features)):
            #     if decoded_full_features[ii].shape[1] > 1:
            #         current_decoded_path = 'DecodedFeatureLevel%d' % ii
            #         current_decoded_path = os.path.join(decoded_path,current_decoded_path)
            #         if os.path.exists(current_decoded_path):
            #             shutil.rmtree(current_decoded_path)
            #         os.makedirs(current_decoded_path)
            #         for jj in range(decoded_full_features[ii].shape[3]):
            #             inf_tools.numpy_img_save(img=image_revalue(decoded_full_features[ii][:, :, :, jj],tah_mark=True),
            #                                      path=os.path.join(current_decoded_path,
            #                                                        'DecodedFeatureLevel%dChannel%d.png'
            #                                                        % (ii, jj)))

            full_generated[current_counter,:,:,:]=current_generated_char
            for ii in range(content_prototype.shape[3]):
                full_content[current_counter,:,:,ii]=current_content_char[:,:,:,ii]

            if current_counter % 5 == 0:
                print("Generated %d/%d samples already" % (current_counter+1,content_prototype.shape[0]))
            current_counter+=1
        print("In total %d chars have been generated" % full_generated.shape[0])
        print(self.print_separater)

        # saving chars
        generated_paper = inf_tools.matrix_paper_generation(images=full_generated,
                                                            rows=output_paper_rows,
                                                            columns=output_paper_cols)
        misc.imsave(os.path.join(self.save_path, 'Generated.png'), generated_paper)
        print("Generated Paper Saved @ %s" % os.path.join(self.save_path, 'Generated.png'))

        style_paper = inf_tools.matrix_paper_generation(images=np.transpose(style_reference,axes=(3,1,2,0)),
                                                        rows=output_paper_rows,
                                                        columns=output_paper_cols)
        misc.imsave(os.path.join(self.save_path, 'RealStyle.png'), style_paper)
        print("Style Paper Saved @ %s" % os.path.join(self.save_path, 'ActualStyles.png'))

        for ii in range(content_prototype.shape[3]):
            content_paper=inf_tools.matrix_paper_generation(images=np.expand_dims(full_content[:,:,:,ii],axis=3),
                                                            rows=output_paper_rows,
                                                            columns=output_paper_cols)
            if not os.path.exists(os.path.join(self.save_path,'Content')):
                os.makedirs(os.path.join(self.save_path,'Content'))
            misc.imsave(os.path.join(os.path.join(self.save_path,'Content'), 'Content%05d_%s.png'
                                     % (ii,content_label1_vec[ii])), content_paper)
        print("Content Papers Saved @ %s" % os.path.join(os.path.join(self.save_path,'Content')))

        print(self.print_separater)
        print("Generated Completed")


