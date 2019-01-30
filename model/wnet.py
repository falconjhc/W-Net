# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

GRAYSCALE_AVG = 127.5
TINIEST_LR = 0.00005

import sys
sys.path.append('..')

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



discriminator_dict = {"DisMdy6conv": discriminator_mdy_6_convs}
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
                 print_info_seconds=-1,
                 train_data_augment=-1,
                 train_data_augment_flip=-1,
                 init_training_epochs=-1,
                 final_training_epochs=-1,
                 style_input_number=-1,
                 adain_use=-1,

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


                 Pixel_Reconstruction_Penalty=100,
                 Lconst_content_Penalty=15,
                 Lconst_style_Penalty=15,
                 Discriminative_Penalty=1,
                 Discriminator_Categorical_Penalty=1,
                 Discriminator_Gradient_Penalty=1,

                 Feature_Penalty_True_Fake_Target=5,
                 Feature_Penalty_Style_Reference=5,
                 Feature_Penalty_Content_Prototype=5,
                 Batch_StyleFeature_Discrimination_Penalty=10,

                 generator_weight_decay_penalty = 0.001,
                 discriminator_weight_decay_penalty = 0.004,

                 resume_training=0,

                 generator_devices='/device:CPU:0',
                 discriminator_devices='/device:CPU:0',
                 feature_extractor_devices='/device:CPU:0',

                 generator_residual_at_layer=3,
                 generator_residual_blocks=5,
                 discriminator='DisMdy6conv',
                 true_fake_target_extractor_dir='/tmp/',
                 content_prototype_extractor_dir='/tmp/',
                 style_reference_extractor_dir='/tmp/',

                 ## for infer only
                 model_dir='./', save_path='./',

                 # for styleadd infer only
                 targeted_content_input_txt='./',
                 target_file_path='-1',
                 save_mode='-1',
                 content_input_number_actual=0,
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

        self.print_info_seconds=print_info_seconds
        self.discriminator_initialization_iters=25
        self.init_training_epochs=init_training_epochs
        self.final_training_epochs=final_training_epochs
        self.final_training_epochs=final_training_epochs
        self.model_save_epochs=15
        self.debug_mode = debug_mode
        self.experiment_dir = experiment_dir
        self.log_dir=log_dir

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
        self.train_data_augment_flip = (train_data_augment_flip==1)




        self.Discriminative_Penalty = Discriminative_Penalty + eps
        self.Discriminator_Gradient_Penalty = Discriminator_Gradient_Penalty + eps
        self.Pixel_Reconstruction_Penalty = Pixel_Reconstruction_Penalty + eps
        self.Feature_Penalty_True_Fake_Target = Feature_Penalty_True_Fake_Target + eps
        self.Feature_Penalty_Style_Reference = Feature_Penalty_Style_Reference + eps
        self.Feature_Penalty_Content_Prototype = Feature_Penalty_Content_Prototype  + eps
        self.Lconst_content_Penalty = Lconst_content_Penalty + eps
        self.Lconst_style_Penalty = Lconst_style_Penalty + eps
        self.Batch_StyleFeature_Discrimination_Penalty=Batch_StyleFeature_Discrimination_Penalty+eps
        self.lr = lr



        self.Discriminator_Categorical_Penalty = Discriminator_Categorical_Penalty + eps
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
        self.save_path=save_path
        if os.path.exists(self.save_path) and not (self.save_path == './'):
            shutil.rmtree(self.save_path)
        if not self.save_path == './':
            os.makedirs(self.save_path)

        # for styleadd infer only
        self.targeted_content_input_txt=targeted_content_input_txt
        self.target_file_path=target_file_path
        self.save_mode=save_mode
        self.content_input_number_actual=content_input_number_actual
        self.random_content = not self.content_input_number_actual == 0
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

        if "WNet" in self.experiment_id:
            if self.other_info=='DenseMixer':
                model_id = "Exp%s_GenEncDec%d-Des%d@Lyr%d_%s" % (self.experiment_id,
                                                                 encoder_decoder_layer_num,
                                                                 self.generator_residual_blocks,
                                                                 self.generator_residual_at_layer,
                                                                 self.discriminator)
            elif self.other_info=='ResidualMixer':
                model_id = "Exp%s_GenEncDec%d-Res%d@Lyr%d_%s" % (self.experiment_id,
                                                                 encoder_decoder_layer_num,
                                                                 self.generator_residual_blocks,
                                                                 self.generator_residual_at_layer,
                                                                 self.discriminator)
        elif "Emd" in self.experiment_id:
            model_id = "Exp%s_GenEncDec%d_%s" % (self.experiment_id,
                                                 encoder_decoder_layer_num,
                                                 self.discriminator)
        elif 'Adobe' in self.experiment_id:
            model_id = "Exp%s_GenEncDec%d_%s" % (self.experiment_id,
                                                 encoder_decoder_layer_num,
                                                 self.discriminator)
        elif 'ResMixer' in self.experiment_id:
            model_id = "Exp%s_GenEncDec%d_%s" % (self.experiment_id,
                                                 encoder_decoder_layer_num,
                                                 self.discriminator)

        if self.random_content:
            model_id = model_id + "_Random%dContent" % self.content_input_number_actual

        model_ckpt_dir = os.path.join(self.checkpoint_dir, model_id)
        model_log_dir = os.path.join(self.log_dir, model_id)
        model_save_path = os.path.join(self.inf_data_dir, model_id)
        return model_id, model_ckpt_dir, model_log_dir, model_save_path

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

    def generate_fake_samples(self,training_mark,current_iterator,
                              training_img_tensor_list):

        evalHandle = getattr(self, "eval_handle")
        if training_mark:
            fake_images, \
            true_style,content_prototypes,style_references, \
            label0_onehot, label1_onehot, \
            label0_dense,label1_dense, \
            output_training_img_list, \
                = self.sess.run([evalHandle.training_generated_images,
                                 current_iterator.output_tensor_list[0],
                                 current_iterator.output_tensor_list[1],
                                 current_iterator.output_tensor_list[2],
                                 current_iterator.output_tensor_list[3],
                                 current_iterator.output_tensor_list[4],
                                 current_iterator.output_tensor_list[5],
                                 current_iterator.output_tensor_list[6],
                                 training_img_tensor_list])
        else:
            fake_images, \
            true_style, content_prototypes, style_references, \
            label0_onehot, label1_onehot, \
            label0_dense, label1_dense \
                = self.sess.run([evalHandle.inferring_generated_images,
                                 current_iterator.output_tensor_list[0],
                                 current_iterator.output_tensor_list[1],
                                 current_iterator.output_tensor_list[2],
                                 current_iterator.output_tensor_list[3],
                                 current_iterator.output_tensor_list[4],
                                 current_iterator.output_tensor_list[5],
                                 current_iterator.output_tensor_list[6]])
            output_training_img_list=list()

        return fake_images, \
               true_style, content_prototypes, style_references, output_training_img_list, \
               label0_onehot, label1_onehot, label0_dense, label1_dense
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
        c_logits = self.sess.run(evalHandle.inferring_categorical_logits,
                                 feed_dict=feed_dict_for_d_tmp())

        return c_logits


    def validate_model(self,
                       train_mark,
                       summary_writer, global_step,
                       data_provider,
                       discriminator_handle,generator_handle,feature_extractor_handle):
        summary_handle = getattr(self,"summary_handle")

        if train_mark:
            merged_real_dis_extr_summaries = summary_handle.trn_real_dis_extr_summaries
            merged_fake_dis_extr_summaries = summary_handle.trn_fake_dis_extr_summaries
            check_image = summary_handle.check_train_image_summary
            check_image_input = summary_handle.check_train_image
            current_iterator = data_provider.train_iterator
            # batch_true_style, \
            # batch_prototype, batch_reference, \
            # batch_label0_onehot, batch_label1_onehot, \
            # batch_label0_dense, batch_label1_dense = \
            #     current_iterator.get_next_batch(sess=self.sess)

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

        else:
            merged_real_dis_extr_summaries = summary_handle.val_real_dis_extr_summaries
            merged_fake_dis_extr_summaries = summary_handle.val_fake_dis_extr_summaries
            check_image = summary_handle.check_validate_image_summary
            check_image_input = summary_handle.check_validate_image
            current_iterator = data_provider.validate_iterator
            # batch_true_style, \
            # batch_prototype, batch_reference, \
            # batch_label0_onehot, batch_label1_onehot, \
            # batch_label0_dense, batch_label1_dense = \
            #     current_iterator.get_next_batch(sess=self.sess)

            training_img_tensor_list=list()




        generated_batch, \
        true_style,content_prototypes,style_references, \
        training_img_list, \
        label0_onehot,label1_onehot, \
        label0_dense,label1_dense, \
            = self.generate_fake_samples(training_mark=train_mark,
                                         current_iterator=current_iterator,
                                         training_img_tensor_list=training_img_tensor_list)
        random_content_prototype_index = rnd.randint(a=0,b=self.content_input_number_actual-1)
        random_style_reference_index = rnd.randint(a=0,b=self.style_input_number-1)
        selected_content = np.expand_dims(content_prototypes[:, :, :, random_content_prototype_index],axis=3)
        selected_style = np.expand_dims(style_references[:, :, :, random_style_reference_index],axis=3)

        summary_fake_output = self.sess.run(merged_fake_dis_extr_summaries,
                                            feed_dict={feature_extractor_handle.infer_input_img: generated_batch,
                                                       feature_extractor_handle.true_label0: label0_onehot,
                                                       feature_extractor_handle.true_label1: label1_onehot,
                                                       discriminator_handle.infer_label1: label1_onehot,
                                                       discriminator_handle.infer_content_prototype: selected_content,
                                                       discriminator_handle.infer_style_reference: selected_style,
                                                       discriminator_handle.infer_true_fake: generated_batch})


        summary_real_output = self.sess.run(merged_real_dis_extr_summaries,
                                            feed_dict={feature_extractor_handle.infer_input_img:true_style,
                                                       feature_extractor_handle.true_label0: label0_onehot,
                                                       feature_extractor_handle.true_label1: label1_onehot,
                                                       discriminator_handle.infer_label1:label1_onehot,
                                                       discriminator_handle.infer_content_prototype:selected_content,
                                                       discriminator_handle.infer_style_reference:selected_style,
                                                       discriminator_handle.infer_true_fake:true_style})



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
                                                self.content_input_number_actual],
                                         dtype=np.float32)
        for ii in range(self.content_input_number_actual):
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

        current_display_content_prototype_indices = rnd.sample(range(self.content_input_number_actual),
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
        summary_writer.add_summary(summray_img, global_step.eval(session=self.sess))

        if self.debug_mode==1 or ((self.debug_mode==0) and global_step.eval(session=self.sess)>=2500):
            summary_writer.add_summary(summary_real_output, global_step.eval(session=self.sess))
            summary_writer.add_summary(summary_fake_output, global_step.eval(session=self.sess))








    def summary_finalization(self,
                             g_loss_summary,
                             d_loss_summary,
                             trn_real_dis_extr_summaries, val_real_dis_extr_summaries,
                             trn_fake_dis_extr_summaries, val_fake_dis_extr_summaries,
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
                                       trn_real_dis_extr_summaries=trn_real_dis_extr_summaries,
                                       val_real_dis_extr_summaries=val_real_dis_extr_summaries,
                                       trn_fake_dis_extr_summaries=trn_fake_dis_extr_summaries,
                                       val_fake_dis_extr_summaries=val_fake_dis_extr_summaries)
        setattr(self, "summary_handle", summary_handle)

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







    def feature_extractor_build(self, g_loss, g_merged_summary,data_provider):

        def calculate_high_level_feature_loss(feature1,feature2):
            for counter in range(len(feature1)):

                feature_diff = feature1[counter] - feature2[counter]
                if not feature_diff.shape.ndims==4:
                    feature_diff = tf.reshape(feature_diff,[int(feature_diff.shape[0]),int(feature_diff.shape[1]),1,1])
                squared_feature_diff = feature_diff**2
                mean_squared_feature_diff = tf.reduce_mean(squared_feature_diff,axis=[1,2,3])
                square_root_mean_squared_feature_diff = tf.sqrt(eps+mean_squared_feature_diff)
                this_feature_loss = tf.reduce_mean(square_root_mean_squared_feature_diff)


                # feature1_normed = (feature1[counter] + 1) / 2
                # feature2_normed = (feature2[counter] + 1) / 2
                feature1_normed = tf.nn.tanh(feature1[counter])+2
                feature2_normed = tf.nn.tanh(feature2[counter])+2
                vn_loss = tf.trace(tf.multiply(feature1_normed, tf.log(feature1_normed)) -
                                   tf.multiply(feature1_normed, tf.log(feature2_normed)) +
                                   - feature1_normed + feature2_normed + eps)
                vn_loss = tf.reduce_mean(vn_loss)


                if counter == 0:
                    final_loss_mse = this_feature_loss
                    final_loss_vn = vn_loss
                else:
                    final_loss_mse += this_feature_loss
                    final_loss_vn += vn_loss
            final_loss_mse = final_loss_mse / len(feature1)
            final_loss_vn = final_loss_vn / len(feature1)

            return final_loss_mse, final_loss_vn

        def build_feature_extractor(input_true_img,
                                    extractor_usage,output_high_level_features):
            generator_handle = getattr(self,'generator_handle')
            output_logit_list = list()


            with tf.variable_scope(tf.get_variable_scope()):
                with tf.device(self.feature_extractor_device):


                    real_features,network_info = \
                        feature_extractor_network(image=input_true_img,
                                                  batch_size=self.batch_size,
                                                  device=self.feature_extractor_device,
                                                  reuse=False,
                                                  keep_prob=1,
                                                  initializer=self.initializer,
                                                  network_usage=extractor_usage,
                                                  output_high_level_features=output_high_level_features)
                    fake_features, _ = \
                        feature_extractor_network(image=generator_handle.generated_target_train,
                                                  batch_size=self.batch_size,
                                                  device=self.feature_extractor_device,
                                                  reuse=True,
                                                  keep_prob=1,
                                                  initializer=self.initializer,
                                                  network_usage=extractor_usage,
                                                  output_high_level_features=output_high_level_features)



            feature_loss_mse, feature_loss_vn = calculate_high_level_feature_loss(feature1=real_features,
                                                                                  feature2=fake_features)

            return output_logit_list, feature_loss_mse, feature_loss_vn, network_info

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
        true_label0 = tf.placeholder(tf.float32, [self.batch_size, len(self.involved_label0_list)])
        true_label1 = tf.placeholder(tf.float32, [self.batch_size, len(self.involved_label1_list)])

        if self.extractor_true_fake_enabled:
            true_fake_infer_list, true_fake_feature_loss_mse, true_fake_feature_loss_vn, network_info = \
                build_feature_extractor(input_true_img=data_provider.train_iterator.output_tensor_list[0],
                                        extractor_usage='TrueFake_FeatureExtractor',
                                        output_high_level_features=[1,2,3,4,5])
            g_loss += true_fake_feature_loss_mse * self.Feature_Penalty_True_Fake_Target
            feature_true_fake_loss_l2_summary = tf.summary.scalar("Loss_Reconstruction/TrueFake_L2",
                                                                  tf.abs(true_fake_feature_loss_mse))
            g_merged_summary = tf.summary.merge([g_merged_summary, feature_true_fake_loss_l2_summary])

            g_loss += true_fake_feature_loss_vn * self.Feature_Penalty_True_Fake_Target
            feature_true_fake_loss_vn_summary = tf.summary.scalar("Loss_Reconstruction/TrueFake_VN",
                                                                  tf.abs(true_fake_feature_loss_vn))
            g_merged_summary = tf.summary.merge([g_merged_summary, feature_true_fake_loss_vn_summary])


            extr_vars_true_fake = [var for var in tf.trainable_variables() if 'TrueFake_FeatureExtractor' in var.name]
            extr_vars_true_fake = self.find_norm_avg_var(extr_vars_true_fake)
            extr_vars_true_fake = self.variable_dict(var_input=extr_vars_true_fake, delete_name_from_character='/')
            saver_extractor_true_fake = tf.train.Saver(max_to_keep=1, var_list=extr_vars_true_fake)


            print("TrueFakeExtractor @ %s with %s;" % (self.feature_extractor_device, network_info))



        else:
            true_fake_infer_list=list()
            true_fake_infer_list.append(-1)
            true_fake_infer_list.append(-1)
            saver_extractor_true_fake = None


        if self.extractor_content_prototype_enabled:
            content_prototype = data_provider.train_iterator.output_tensor_list[1]
            selected_index = tf.random_uniform(shape=[],minval=0,maxval=int(content_prototype.shape[3]),dtype=tf.int64)
            selected_content_prototype = tf.expand_dims(content_prototype[:,:,:,selected_index], axis=3)
            content_prototype_infer_list,content_prototype_feature_mse_loss, content_prototype_feature_vn_loss, network_info = \
                build_feature_extractor(input_true_img=selected_content_prototype,
                                        extractor_usage='ContentPrototype_FeatureExtractor',
                                        output_high_level_features=[3,4,5])
            g_loss += content_prototype_feature_mse_loss * self.Feature_Penalty_Content_Prototype
            feature_content_prototype_mse_loss_summary = tf.summary.scalar("Loss_Reconstruction/ContentPrototype_L2",
                                                                           tf.abs(content_prototype_feature_mse_loss))
            g_merged_summary = tf.summary.merge([g_merged_summary, feature_content_prototype_mse_loss_summary])

            g_loss += content_prototype_feature_vn_loss * self.Feature_Penalty_Content_Prototype
            feature_content_prototype_vn_loss_summary = tf.summary.scalar("Loss_Reconstruction/ContentPrototype_VN",
                                                                          tf.abs(content_prototype_feature_vn_loss))
            g_merged_summary = tf.summary.merge([g_merged_summary, feature_content_prototype_vn_loss_summary])



            extr_vars_content_prototype = [var for var in tf.trainable_variables() if 'ContentPrototype_FeatureExtractor' in var.name]
            extr_vars_content_prototype = self.find_norm_avg_var(extr_vars_content_prototype)
            extr_vars_content_prototype = self.variable_dict(var_input=extr_vars_content_prototype, delete_name_from_character='/')
            saver_extractor_content_prototype = tf.train.Saver(max_to_keep=1, var_list=extr_vars_content_prototype)
            print("ContentPrototypeExtractor @ %s with %s;" % (self.feature_extractor_device, network_info))


        else:
            content_prototype_infer_list=list()
            content_prototype_infer_list.append(-1)
            saver_extractor_content_prototype = None
            selected_content_prototype = -1

        if self.extractor_style_reference_enabled:
            style_reference = data_provider.train_iterator.output_tensor_list[2]
            selected_index = tf.random_uniform(shape=[], minval=0, maxval=int(style_reference.shape[3]), dtype=tf.int64)
            selected_style_reference = tf.expand_dims(style_reference[:, :, :, selected_index], axis=3)
            style_reference_infer_list, style_reference_feature_mse_loss, style_reference_feature_vn_loss, network_info = \
                build_feature_extractor(input_true_img = selected_style_reference,
                                        extractor_usage='StyleReference_FeatureExtractor',
                                        output_high_level_features=[3,4,5])
            g_loss += style_reference_feature_mse_loss * self.Feature_Penalty_Style_Reference
            feature_style_reference_mse_loss_summary = tf.summary.scalar("Loss_Reconstruction/StyleReference_L2",
                                                                         tf.abs(style_reference_feature_mse_loss))
            g_merged_summary = tf.summary.merge([g_merged_summary, feature_style_reference_mse_loss_summary])

            g_loss += style_reference_feature_vn_loss * self.Feature_Penalty_Style_Reference
            feature_style_reference_vn_loss_summary = tf.summary.scalar("Loss_Reconstruction/StyleReference_VN",
                                                                        tf.abs(style_reference_feature_vn_loss))
            g_merged_summary = tf.summary.merge([g_merged_summary, feature_style_reference_vn_loss_summary])


            extr_vars_style_reference = [var for var in tf.trainable_variables() if 'StyleReference_FeatureExtractor' in var.name]
            extr_vars_style_reference = self.find_norm_avg_var(extr_vars_style_reference)
            extr_vars_style_reference = self.variable_dict(var_input=extr_vars_style_reference, delete_name_from_character='/')
            saver_extractor_style_reference = tf.train.Saver(max_to_keep=1, var_list=extr_vars_style_reference)
            print("StyleReferenceExtractor @ %s with %s;" % (self.feature_extractor_device, network_info))

        else:
            style_reference_infer_list=list()
            style_reference_infer_list.append(-1)
            saver_extractor_style_reference = None
            selected_style_reference = -1


        feature_extractor_handle = FeatureExtractorHandle(infer_input_img=input_target_infer,
                                                          true_label0=true_label0,
                                                          true_label1=true_label1,
                                                          selected_content_prototype=selected_content_prototype,
                                                          selected_style_reference=selected_style_reference)
        setattr(self, "feature_extractor_handle", feature_extractor_handle)


        saver_list.append(saver_extractor_true_fake)
        saver_list.append(saver_extractor_content_prototype)
        saver_list.append(saver_extractor_style_reference)


        return g_loss, g_merged_summary,saver_list, \
               extr_trn_real_merged,extr_trn_fake_merged,extr_val_real_merged,extr_val_fake_merged



    def generator_build(self,data_provider):

        name_prefix = 'generator'

        # network architechture
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(self.generator_devices):


                content_prototype_train = data_provider.train_iterator.output_tensor_list[1]
                style_reference_train_list = list()
                for ii in range(self.style_input_number):
                    style_reference_train = tf.expand_dims(data_provider.train_iterator.output_tensor_list[2][:,:,:,ii], axis=3)
                    style_reference_train_list.append(style_reference_train)

                true_style_train = data_provider.train_iterator.output_tensor_list[0]

                # build the generator
                generated_target_train, encoded_content_prototype_train, encoded_style_reference_train, network_info, \
                style_shortcut_batch_diff, style_residual_batch_diff,_,_,_ = \
                    self.generator_implementation(content_prototype=content_prototype_train,
                                                  style_reference=style_reference_train_list,
                                                  is_training=True,
                                                  batch_size=self.batch_size,
                                                  generator_device=self.generator_devices,
                                                  residual_at_layer=self.generator_residual_at_layer,
                                                  residual_block_num=self.generator_residual_blocks,
                                                  scope=name_prefix,
                                                  reuse=False,
                                                  initializer=self.initializer,
                                                  weight_decay=self.weight_decay_generator,
                                                  weight_decay_rate=self.generator_weight_decay_penalty,
                                                  adain_use=self.adain_use,
                                                  adain_preparation_model=self.adain_preparation_model,
                                                  debug_mode=self.debug_mode,
                                                  other_info=self.other_info)


                # encoded of the generated target on the content prototype encoder
                encoded_content_prototype_generated_target = \
                    self.encoder_implementation(images=tf.tile(generated_target_train,
                                                               [1,1,1,self.content_input_number_actual]),
                                                is_training=True,
                                                encoder_device=self.generator_devices,
                                                residual_at_layer=self.generator_residual_at_layer,
                                                residual_connection_mode='Multi',
                                                scope=name_prefix+'/content_encoder',
                                                reuse=True,
                                                initializer=self.initializer,
                                                weight_decay=False,
                                                weight_decay_rate=self.generator_weight_decay_penalty,
                                                adain_use=self.adain_use)[0]

                # encoded of the generated target on the style reference encoder
                if not 'WNet' in self.experiment_id :
                    tmp_input = tf.tile(generated_target_train,
                                        [1, 1, 1, self.style_input_number])
                else:
                    tmp_input = generated_target_train
                encoded_style_reference_generated_target = \
                    self.encoder_implementation(images=tmp_input,
                                                is_training=True,
                                                encoder_device=self.generator_devices,
                                                residual_at_layer=self.generator_residual_at_layer,
                                                residual_connection_mode='Single',
                                                scope=name_prefix + '/style_encoder',
                                                reuse=True,
                                                initializer=self.initializer,
                                                weight_decay=False,
                                                weight_decay_rate=self.generator_weight_decay_penalty,
                                                adain_use=self.adain_use)[0]








                # for inferring
                content_prototype_infer = data_provider.validate_iterator.output_tensor_list[1]
                style_reference_infer_list = list()
                for ii in range(self.style_input_number):
                    style_reference_infer = tf.expand_dims(data_provider.validate_iterator.output_tensor_list[2][:, :, :, ii], axis=3)
                    style_reference_infer_list.append(style_reference_infer)


                generated_target_infer = \
                    self.generator_implementation(content_prototype=content_prototype_infer,
                                                  style_reference=style_reference_infer_list,
                                                  is_training=False,
                                                  batch_size=self.batch_size,
                                                  generator_device=self.generator_devices,
                                                  residual_at_layer=self.generator_residual_at_layer,
                                                  residual_block_num=self.generator_residual_blocks,
                                                  scope=name_prefix,
                                                  reuse=True,
                                                  initializer=self.initializer,
                                                  weight_decay=False,
                                                  weight_decay_rate=eps,
                                                  adain_use=self.adain_use,
                                                  adain_preparation_model=self.adain_preparation_model,
                                                  debug_mode=True,
                                                  other_info=self.other_info)[0]

                curt_generator_handle = GeneratorHandle(generated_target_train=generated_target_train,
                                                        generated_target_infer=generated_target_infer)
                setattr(self, "generator_handle", curt_generator_handle)



        # loss build
        g_loss=0
        g_merged_summary = []

        # batch discrimination loss on style encoder
        if self.Batch_StyleFeature_Discrimination_Penalty > eps * 10:
            if not style_shortcut_batch_diff==-1:
                style_shortcut_batch_discrimination_penalty = self.Batch_StyleFeature_Discrimination_Penalty * style_shortcut_batch_diff
                g_loss += style_shortcut_batch_discrimination_penalty
                style_batch_diff_shortcut_summary = tf.summary.scalar("Loss_Generator/StyleDiscrimination_ShortCut",
                                                                      style_shortcut_batch_discrimination_penalty / self.Batch_StyleFeature_Discrimination_Penalty)
                g_merged_summary = tf.summary.merge([g_merged_summary, style_batch_diff_shortcut_summary])

            if not style_residual_batch_diff == -1:
                style_residual_batch_discrimination_penalty = self.Batch_StyleFeature_Discrimination_Penalty * style_residual_batch_diff
                g_loss += style_residual_batch_discrimination_penalty
                style_batch_diff_residual_summary = tf.summary.scalar("Loss_Generator/StyleDiscrimination_Residual",
                                                                      style_residual_batch_discrimination_penalty / self.Batch_StyleFeature_Discrimination_Penalty)
                g_merged_summary = tf.summary.merge([g_merged_summary, style_batch_diff_residual_summary])


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
            const_loss_content = const_loss_content / self.content_input_number_actual
            g_loss += const_loss_content
            const_content_loss_summary = tf.summary.scalar("Loss_Generator/ConstContentPrototype",
                                                           tf.abs(const_loss_content) / self.Lconst_content_Penalty)
            g_merged_summary=tf.summary.merge([g_merged_summary, const_content_loss_summary])
        if self.Lconst_style_Penalty > eps * 10:
            current_const_loss_style = tf.square(encoded_style_reference_train[:,:,:,0:int(encoded_style_reference_generated_target.shape[3])]
                                                 - encoded_style_reference_generated_target)
            current_const_loss_style = tf.reduce_mean(current_const_loss_style) * self.Lconst_style_Penalty
            g_loss += current_const_loss_style
            const_style_loss_summary = tf.summary.scalar("Loss_Generator/ConstStyleReference",
                                                         tf.abs(current_const_loss_style) / self.Lconst_style_Penalty)
            g_merged_summary=tf.summary.merge([g_merged_summary, const_style_loss_summary])



        # l1 loss
        if self.Pixel_Reconstruction_Penalty > eps * 10:
            l1_loss = tf.abs(generated_target_train - true_style_train)
            l1_loss = tf.reduce_mean(l1_loss) * self.Pixel_Reconstruction_Penalty
            l1_loss_summary = tf.summary.scalar("Loss_Reconstruction/Pixel_L1",
                                                tf.abs(l1_loss) / self.Pixel_Reconstruction_Penalty)
            g_loss+=l1_loss
            g_merged_summary = tf.summary.merge([g_merged_summary, l1_loss_summary])


        gen_vars_train = [var for var in tf.trainable_variables() if 'generator' in var.name]
        gen_vars_save = self.find_norm_avg_var(gen_vars_train)
        saver_generator = tf.train.Saver(max_to_keep=self.model_save_epochs, var_list=gen_vars_save)


        print(
            "Generator @%s with %s;" % (self.generator_devices, network_info))
        return generated_target_infer, generated_target_train, \
               g_loss, g_merged_summary, \
               gen_vars_train, saver_generator


    def discriminator_build(self,
                            g_loss,
                            g_merged_summary,
                            data_provider):


        def _calculate_accuracy_and_entropy(logits, true_labels, summary_name_parafix):
            prdt_labels = tf.argmax(logits,axis=1)
            true_labels = tf.argmax(true_labels,axis=1)
            correct_prediction = tf.equal(prdt_labels,true_labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) * 100
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.nn.softmax(logits))
            entropy = tf.reduce_mean(entropy)
            acry_summary = tf.summary.scalar("Accuracy_Discriminator/AuxClassifier_"+summary_name_parafix, accuracy)
            enpy_summary = tf.summary.scalar("Entropy_Discriminator/AuxClassifier_"+summary_name_parafix, entropy)
            return acry_summary,enpy_summary

        generator_handle = getattr(self,'generator_handle')
        fake_style_train = generator_handle.generated_target_train

        name_prefix = 'discriminator'

        discriminator_category_logit_length = len(self.involved_label1_list)

        critic_logit_length = int(np.floor(math.log(discriminator_category_logit_length) / math.log(2)))
        critic_logit_length = np.power(2,critic_logit_length+1)
        critic_logit_length = np.max([critic_logit_length,512])
        critic_logit_length = np.min([critic_logit_length, 1024])

        current_critic_logit_penalty = tf.placeholder(tf.float32, [], name='current_critic_logit_penalty')

        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(self.discriminator_devices):
                true_label1_train = data_provider.train_iterator.output_tensor_list[4]
                true_style_train = data_provider.train_iterator.output_tensor_list[0]

                content_prototype_train_all = data_provider.train_iterator.output_tensor_list[1]
                style_reference_train_all = data_provider.train_iterator.output_tensor_list[2]
                content_train_random_index = tf.random_uniform(shape=[], minval=0,maxval=self.content_input_number_actual,dtype=tf.int64)
                style_train_random_index = tf.random_uniform(shape=[], minval=0, maxval=self.style_input_number,dtype=tf.int64)
                content_prototype_train = tf.expand_dims(content_prototype_train_all[:,:,:,content_train_random_index], axis=3)
                style_reference_train = tf.expand_dims(style_reference_train_all[:, :, :, style_train_random_index], axis=3)
                real_pack_train = tf.concat([content_prototype_train, true_style_train, style_reference_train], axis=3)
                fake_pack_train = tf.concat([content_prototype_train, fake_style_train, style_reference_train], axis=3)

                real_C_logits,real_Discriminator_logits,network_info = \
                    self.discriminator_implementation(image=real_pack_train,
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
                    self.discriminator_implementation(image=fake_pack_train,
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
                interpolated_pair = real_pack_train*epsilon + (1-epsilon)*fake_pack_train
                _,intepolated_Cr_logits,_ = self.discriminator_implementation(image=interpolated_pair,
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




                # discriminator infer
                infer_content_prototype = tf.placeholder(tf.float32, [self.batch_size,
                                                                      self.img2img_width,
                                                                      self.img2img_width,
                                                                      self.input_output_img_filter_num])
                infer_true_fake = tf.placeholder(tf.float32, [self.batch_size,
                                                              self.img2img_width,
                                                              self.img2img_width,
                                                              self.input_output_img_filter_num])
                infer_style_reference = tf.placeholder(tf.float32, [self.batch_size,
                                                                    self.img2img_width,
                                                                    self.img2img_width,
                                                                    self.input_output_img_filter_num])
                infer_label1 = tf.placeholder(tf.float32,[self.batch_size,len(self.involved_label1_list)])
                infer_pack = tf.concat([infer_content_prototype, infer_true_fake, infer_style_reference], axis=3)
                infer_categorical_logits,_,_ = \
                    self.discriminator_implementation(image=infer_pack,
                                                      parameter_update_device=self.discriminator_devices,
                                                      category_logit_num=discriminator_category_logit_length,
                                                      batch_size=self.batch_size,
                                                      critic_length=critic_logit_length,
                                                      reuse=True,
                                                      initializer=self.initializer,
                                                      weight_decay=False,
                                                      weight_decay_rate=eps,
                                                      scope=name_prefix)
                infer_categorical_logits = tf.nn.softmax(infer_categorical_logits)

                curt_discriminator_handle = DiscriminatorHandle(current_critic_logit_penalty=current_critic_logit_penalty,
                                                                infer_content_prototype=infer_content_prototype,
                                                                infer_style_reference=infer_style_reference,
                                                                infer_true_fake=infer_true_fake,
                                                                infer_label1=infer_label1)
                setattr(self, "discriminator_handle", curt_discriminator_handle)



        # loss build
        d_loss = 0
        d_merged_summary=[]


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
                                                                         labels=true_label1_train)
            fake_category_loss = tf.nn.softmax_cross_entropy_with_logits(logits=fake_C_logits,
                                                                         labels=true_label1_train)

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
            #d_loss_real_fake_summary = tf.summary.scalar("TrainingProgress_DiscriminatorRealFakeLoss",
            #                                             tf.reduce_mean(tf.abs(real_Discriminator_logits - fake_Discriminator_logits)) / self.Discriminative_Penalty)
            d_loss_real_fake_summary1 = tf.summary.scalar("TrainingProgress/LogitDifferenceMean",
                                                          tf.reduce_mean(tf.abs(real_Discriminator_logits - fake_Discriminator_logits)) / self.Discriminative_Penalty)
            d_loss_real_fake_summary2 = tf.summary.scalar("TrainingProgress/AdversarialDifference",
                                                          tf.abs(d_loss_real + d_loss_fake) / self.Discriminative_Penalty)
            if self.Discriminator_Gradient_Penalty > 10 * eps:
                d_gradient_loss = discriminator_slopes
                d_gradient_loss = tf.reduce_mean(d_gradient_loss) * self.Discriminator_Gradient_Penalty
                d_gradient_loss_summary = tf.summary.scalar("Loss_Discriminator/D_Gradient",
                                                            tf.abs(
                                                                d_gradient_loss) / self.Discriminator_Gradient_Penalty)
                d_loss += d_gradient_loss
                d_merged_summary = tf.summary.merge([d_merged_summary,
                                                     d_gradient_loss_summary,
                                                     d_loss_real_fake_summary1,
                                                     d_loss_real_fake_summary2])

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
        trn_real_acry, trn_real_enty = _calculate_accuracy_and_entropy(logits=infer_categorical_logits,
                                                                       true_labels=infer_label1,
                                                                       summary_name_parafix="TrainReal")
        trn_fake_acry, trn_fake_enty = _calculate_accuracy_and_entropy(logits=infer_categorical_logits,
                                                                       true_labels=infer_label1,
                                                                       summary_name_parafix="TrainFake")
        val_real_acry, val_real_enty = _calculate_accuracy_and_entropy(logits=infer_categorical_logits,
                                                                       true_labels=infer_label1,
                                                                       summary_name_parafix="TestReal")
        val_fake_acry, val_fake_enty = _calculate_accuracy_and_entropy(logits=infer_categorical_logits,
                                                                       true_labels=infer_label1,
                                                                       summary_name_parafix="TestFake")


        trn_real_summary = tf.summary.merge([trn_real_acry, trn_real_enty])
        trn_fake_summary = tf.summary.merge([trn_fake_acry, trn_fake_enty])

        tst_real_summary = tf.summary.merge([val_real_acry, val_real_enty])
        tst_fake_summary = tf.summary.merge([val_fake_acry, val_fake_enty])

        dis_vars_train = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        dis_vars_save = self.find_norm_avg_var(dis_vars_train)



        saver_discriminator = tf.train.Saver(max_to_keep=self.model_save_epochs, var_list=dis_vars_save)


        print("Discriminator @ %s with %s;" % (self.discriminator_devices,network_info))
        return g_merged_summary, d_merged_summary, \
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
        self.sess.run(tf.tables_initializer())



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
                                         info_print_interval=self.print_info_seconds / 10,
                                         input_width=self.source_img_width,
                                         input_filters=self.input_output_img_filter_num,
                                         augment_train_data=self.train_data_augment,
                                         augment_train_data_flip=self.train_data_augment_flip,
                                         style_input_num=self.style_input_number,
                                         content_data_dir=self.content_data_dir,
                                         style_train_data_dir=self.style_train_data_dir,
                                         style_validation_data_dir=self.style_validation_data_dir,
                                         file_list_txt_content=self.file_list_txt_content,
                                         file_list_txt_style_train=self.file_list_txt_style_train,
                                         file_list_txt_style_validation=self.file_list_txt_style_validation,
                                         debug_mode=self.debug_mode,
                                         content_input_number_actual=self.content_input_number_actual)

            self.involved_label0_list, self.involved_label1_list = data_provider.get_involved_label_list()
            self.content_input_num = data_provider.content_input_num
            self.display_style_reference_num = np.min([4, self.style_input_number])
            if self.content_input_number_actual==0:
                self.content_input_number_actual = self.content_input_num
            self.display_content_reference_num = np.min([4, self.content_input_number_actual])


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
            generated_batch_infer, generated_batch_train, \
            g_loss, g_merged_summary, \
            gen_vars_train, saver_generator \
                = self.generator_build(data_provider=data_provider)

            # for feature extractor
            g_loss, g_merged_summary, feature_extractor_saver_list, \
            extr_trn_real_merged, extr_trn_fake_merged, extr_val_real_merged, extr_val_fake_merged= \
                self.feature_extractor_build(g_loss=g_loss,
                                             g_merged_summary=g_merged_summary,
                                             data_provider=data_provider)

            # for discriminator building
            g_merged_summary, d_merged_summary, \
            g_loss, d_loss, \
            dis_trn_real_summary, dis_trn_fake_summary, dis_val_real_summary, dis_val_fake_summary, \
            dis_vars_train, saver_discriminator = \
                self.discriminator_build(g_loss=g_loss,
                                         g_merged_summary=g_merged_summary,
                                         data_provider=data_provider)
            evalHandle = EvalHandle(inferring_generated_images=generated_batch_infer,
                                    training_generated_images=generated_batch_train)
            setattr(self, "eval_handle", evalHandle)


            # # for optimizer creation
            optimizer_g, optimizer_d = \
                self.create_optimizer(learning_rate=learning_rate,
                                      global_step=global_step,
                                      gen_vars_train=gen_vars_train,
                                      generator_loss_train=g_loss,
                                      dis_vars_train=dis_vars_train,
                                      discriminator_loss_train=d_loss)

            trn_real_dis_extr_summary_merged = tf.summary.merge([dis_trn_real_summary, extr_trn_real_merged])
            trn_fake_dis_extr_summary_merged = tf.summary.merge([dis_trn_fake_summary, extr_trn_fake_merged])
            val_real_dis_extr_summary_merged = tf.summary.merge([dis_val_real_summary, extr_val_real_merged])
            val_fake_dis_extr_summary_merged = tf.summary.merge([dis_val_fake_summary, extr_val_fake_merged])

            # summaries
            self.summary_finalization(g_loss_summary=g_merged_summary,
                                      d_loss_summary=d_merged_summary,
                                      trn_real_dis_extr_summaries=trn_real_dis_extr_summary_merged,
                                      val_real_dis_extr_summaries=val_real_dis_extr_summary_merged,
                                      trn_fake_dis_extr_summaries=trn_fake_dis_extr_summary_merged,
                                      val_fake_dis_extr_summaries=val_fake_dis_extr_summary_merged,
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
            print("DataAugment/Flip:%d/%d, InputStyleNum:%d" % (self.train_data_augment, self.train_data_augment_flip, self.style_input_number))
            print(self.print_separater)
            print("Penalties:")
            print("Generator: PixelL1:%.3f,ConstCP/SR:%.3f/%.3f,Wgt:%.6f, BatchDist:%.5f;"
                  % (self.Pixel_Reconstruction_Penalty,
                     self.Lconst_content_Penalty,
                     self.Lconst_style_Penalty,
                     self.generator_weight_decay_penalty,
                     self.Batch_StyleFeature_Discrimination_Penalty))
            print("Discriminator: Cat:%.3f,Dis:%.3f,WST-Grdt:%.3f,Wgt:%.6f;" % (self.Discriminator_Categorical_Penalty,
                                                                                self.Discriminative_Penalty,
                                                                                self.Discriminator_Gradient_Penalty,
                                                                                self.discriminator_weight_decay_penalty))
            print("FeatureReconstructionOnExtractor: TrueFalse:%.3f, ContentPrototype:%.3f, StyleReference:%.3f;" %
                  (self.Feature_Penalty_True_Fake_Target,
                   self.Feature_Penalty_Content_Prototype,
                   self.Feature_Penalty_Style_Reference))

            print("InitLearningRate:%.3f" % self.lr)
            print("AdaIN_Mode:%s" % self.adain_mark)
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
        def W_GAN(current_epoch,discriminator_handle):

            if current_epoch <= self.final_training_epochs:
                self.g_iters = 5
            else:
                self.g_iters = 3

            if global_step.eval(session=self.sess) <= self.g_iters * 5 * self.discriminator_initialization_iters:
                g_iters = self.g_iters * 5
            else:
                g_iters = self.g_iters

            info=""

            batch_true_style_train,\
            batch_train_prototype, batch_train_reference, \
            batch_train_label0_onehot, batch_train_label1_onehot,\
            batch_train_label0_dense, batch_train_label1_dense, \
            true_style_threshold_train, content_threshold_train, style_threshold_train= \
                data_provider.train_iterator.get_next_batch(sess=self.sess)
            #
            # batch_true_style_val, \
            # batch_val_prototype, batch_val_reference, \
            # batch_val_label0_onehot, batch_val_label1_onehot, \
            # batch_val_label0_dense, batch_val_label1_dense,\
            # true_style_threshold_val, content_threshold_val, style_threshold_val = \
            #     data_provider.validate_iterator.get_next_batch(sess=self.sess)

            optimization_start = time.time()

            if dis_vars_train \
                    or global_step.eval(session=self.sess) == global_step_start:
                _ = self.sess.run(optimizer_d, feed_dict={learning_rate: current_lr_real,
                                                          discriminator_handle.current_critic_logit_penalty: current_critic_logit_penalty_value})

                info=info+"OptimizeOnD"

            # optimization for generator every (g_iters) iterations
            if ((global_step.eval(session=self.sess)) % g_iters == 0
                or global_step.eval(session=self.sess) == global_step_start + 1) and gen_vars_train:
                _ = self.sess.run(optimizer_g, feed_dict={learning_rate: current_lr_real})



                info = info + "&&G"
            optimization_elapsed = time.time() - optimization_start

            return optimization_elapsed,info



        summary_start = time.time()
        sample_start = time.time()
        print_info_start = time.time()
        record_start = time.time()
        discriminator_handle = getattr(self, "discriminator_handle")
        feature_extractor_handle = getattr(self, "feature_extractor_handle")
        generator_handle = getattr(self, "generator_handle")



        if self.resume_training==1:
            ei_start = epoch_step.eval(self.sess)
            current_lr = self.lr * np.power(learning_rate_decay_rate, ei_start)

        else:
            ei_start = 0
            current_lr = self.lr

        global_step_start = global_step.eval(session=self.sess)
        print("InitTrainingEpochs:%d, FinalTrainingEpochStartAt:%d" % (self.init_training_epochs,self.final_training_epochs))
        print("TrainingStart:Epoch:%d, GlobalStep:%d, LearnRate:%.5f" % (ei_start+1,global_step_start+1,current_lr))
        print("ContentLabel1Vec:")
        print(data_provider.content_label1_vec)


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
                print("decay learning rate from %.7f to %.7f" % (current_lr, update_lr))
                print(self.print_separater)
                current_lr = update_lr

            for bid in range(self.itrs_for_current_epoch):
                if time.time() - training_start_time <= 4800 or ei < self.init_training_epochs:
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
                    current_critic_logit_penalty_value = (float(global_step.eval(session=self.sess))/float(self.init_training_epochs*self.itrs_for_current_epoch))*self.Discriminative_Penalty + eps
                    current_lr_real = current_lr * 0.1
                    # current_lr_real = current_lr
                else:
                    current_critic_logit_penalty_value = self.Discriminative_Penalty
                    current_lr_real = current_lr
                current_critic_logit_penalty_value = current_critic_logit_penalty_value * 0.001
                # current_lr_real = max(current_lr_real, TINIEST_LR)


                optimization_consumed, \
                info = W_GAN(current_epoch=epoch_step.eval(session=self.sess),
                             discriminator_handle=discriminator_handle)

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

                    percentage_completed = float(global_step.eval(session=self.sess)) / float((self.epoch - ei_start) * self.itrs_for_current_epoch) * 100
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


                if ((time.time()-summary_start>summary_seconds) and (self.debug_mode==0) and global_step.eval(session=self.sess)>=2500) \
                        or self.debug_mode==1:
                    summary_start = time.time()

                    if dis_vars_train:
                        d_summary = self.sess.run(
                            summary_handle.d_merged,
                            feed_dict={discriminator_handle.current_critic_logit_penalty:
                                           current_critic_logit_penalty_value})


                        summary_writer.add_summary(d_summary, global_step.eval(session=self.sess))
                    if gen_vars_train:
                        g_summary = self.sess.run(summary_handle.g_merged)
                        summary_writer.add_summary(g_summary, global_step.eval(session=self.sess))

                    learning_rate_summary = self.sess.run(summary_handle.learning_rate,
                                                          feed_dict={learning_rate: current_lr_real})
                    summary_writer.add_summary(learning_rate_summary, global_step.eval(session=self.sess))
                    summary_writer.flush()


                if time.time()-sample_start>sample_seconds or global_step.eval(session=self.sess)==global_step_start+1 or bid==self.itrs_for_current_epoch-1:
                    sample_start = time.time()

                    # check for train set
                    self.validate_model(train_mark=True,
                                        summary_writer=summary_writer,
                                        global_step=global_step,
                                        discriminator_handle=discriminator_handle,
                                        feature_extractor_handle=feature_extractor_handle,
                                        generator_handle=generator_handle,
                                        data_provider=data_provider)

                    # check for validation set
                    self.validate_model(train_mark=False,
                                        summary_writer=summary_writer,
                                        global_step=global_step,
                                        discriminator_handle=discriminator_handle,
                                        feature_extractor_handle=feature_extractor_handle,
                                        generator_handle=generator_handle,
                                        data_provider=data_provider)

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

