# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
import sys
import os
sys.path.append('../../')

from model.wnet import WNet as WNET
eps = 1e-9


data_path_root = '/data0/Harric/ChineseCharacterExp/'
model_log_path_root = '/data1/Harric/ChineseCharacterExp/'

# exp_root_path = '/Users/harric/Downloads/WNet_Exp/'


# OPTIONS SPECIFICATION
# resume_training = 0: training from stratch
#                   1: training from a based model
input_args = [
              '--debug_mode','0',
              '--style_input_number','4', # how many style inputs
              '--init_training_epochs','1',
              '--final_training_epochs','1500',
              '--adain_use','1-Single',

              '--generator_device','/device:GPU:0',
              '--discriminator_device', '/device:GPU:0',
              '--feature_extractor_device','/device:GPU:0',


              '--train_data_augment','1', # translation? rotation?
              '--train_data_augment_flip','0',
              '--experiment_id','20181225-AdaIN-Single_StylePf50_ContentPf32',# experiment name prefix
              '--experiment_dir','tfModels_WNet/', # model saving location
              '--log_dir','tfLogs_WNet/',# log file saving location
              '--print_info_seconds','750',

              '--content_data_dir', # standard data location
    'CASIA_Dataset/PrintedData/',

              '--style_train_data_dir', # training data location
    'CASIA_Dataset/PrintedData/GB2312_L1/',

              '--style_validation_data_dir',# validation data location
    'CASIA_Dataset/PrintedData/GB2312_L1/',

              '--file_list_txt_content', # file list of the standard data
    '../../FileList/PrintedData/Char_0_3754_Writer_Selected32_Printed_Fonts_GB2312L1.txt',
    
              '--file_list_txt_style_train', # file list of the training data
    '../../FileList/PrintedData/Char_0_3754_Font_0_49_GB2312L1.txt',

              '--file_list_txt_style_validation', # file list of the validation data
    '../../FileList/PrintedData/Char_0_3754_Font_50_79_GB2312L1.txt',


              # generator && discriminator
              '--generator_residual_at_layer','3',
              '--generator_residual_blocks','5',
              '--discriminator','DisMdy6conv',

              '--batch_size','4',
              '--img_width','64',
              '--channels','1',

              # optimizer parameters
              '--init_lr','0.0002',
              '--epoch','5000',
              '--resume_training','0', # 0: training from scratch; 1: training from a pre-trained point

              '--optimization_method','adam',
              '--final_learning_rate_pctg','0.01',


              # penalties
              '--generator_weight_decay_penalty','0.0001',
              '--discriminator_weight_decay_penalty','0.0003',
              '--Pixel_Reconstruction_Penalty','750',
              '--Lconst_content_Penalty','3',
              '--Lconst_style_Penalty','5',
              '--Discriminative_Penalty', '50',
              '--Discriminator_Categorical_Penalty', '50',
              '--Generator_Categorical_Penalty', '0.2',
              '--Discriminator_Gradient_Penalty', '10',
              '--Batch_StyleFeature_Discrimination_Penalty','0',


        # feature extractor parametrers
              '--true_fake_target_extractor_dir',
    'TrainedModel_CNN_WithAugment/ContentStyleBoth/Exp20181010_FeatureExtractor_ContentStyle_PF50_vgg16net/variables/',
              '--content_prototype_extractor_dir',
    'TrainedModel_CNN_WithAugment/ContentOnly/Exp20181010_FeatureExtractor_Content_PF32_vgg16net/variables/',
              '--style_reference_extractor_dir',
    'TrainedModel_CNN_WithAugment/StyleOnly/Exp20181010_FeatureExtractor_Style_PF50_vgg16net/variables/',
              '--Feature_Penalty_True_Fake_Target', '800',
              '--Feature_Penalty_Style_Reference','15',
              '--Feature_Penalty_Content_Prototype','15']




parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--debug_mode', dest='debug_mode',type=int,required=True)
parser.add_argument('--resume_training', dest='resume_training', type=int,required=True)
parser.add_argument('--train_data_augment', dest='train_data_augment', type=int,required=True)
parser.add_argument('--train_data_augment_flip', dest='train_data_augment_flip', type=int,required=True)
parser.add_argument('--print_info_seconds', dest='print_info_seconds',type=int,required=True)
parser.add_argument('--style_input_number', dest='style_input_number', type=int,required=True)
parser.add_argument('--content_input_number_actual', dest='content_input_number_actual',type=int, default=0)
parser.add_argument('--adain_use', dest='adain_use',type=str, default=None)


# directories setting
parser.add_argument('--experiment_dir', dest='experiment_dir', required=True)
parser.add_argument('--log_dir', dest='log_dir', required=True)
parser.add_argument('--experiment_id', dest='experiment_id', type=str, required=True)
parser.add_argument('--training_from_model_dir', dest='training_from_model_dir', default=None)


# network settings
parser.add_argument('--generator_device', dest='generator_device',type=str,required=True)
parser.add_argument('--generator_residual_at_layer', dest='generator_residual_at_layer', type=int, required=True)
parser.add_argument('--generator_residual_blocks', dest='generator_residual_blocks', type=int, required=True)





parser.add_argument('--discriminator', dest='discriminator', type=str, required=True)
parser.add_argument('--discriminator_device', dest='discriminator_device',type=str,required=True)


parser.add_argument('--feature_extractor_device', dest='feature_extractor_device',type=str,required=True)
parser.add_argument('--true_fake_target_extractor_dir', dest='true_fake_target_extractor_dir', type=str, required=True)
parser.add_argument('--style_reference_extractor_dir', dest='style_reference_extractor_dir', type=str, required=True)
parser.add_argument('--content_prototype_extractor_dir', dest='content_prototype_extractor_dir', type=str, required=True)



# input data setting
parser.add_argument('--content_data_dir',dest='content_data_dir',type=str,required=True)
parser.add_argument('--style_train_data_dir',dest='style_train_data_dir',type=str,required=True)
parser.add_argument('--style_validation_data_dir',dest='style_validation_data_dir',type=str,required=True)

parser.add_argument('--file_list_txt_content',dest='file_list_txt_content',type=str,required=True)
parser.add_argument('--file_list_txt_style_train',dest='file_list_txt_style_train',type=str,required=True)
parser.add_argument('--file_list_txt_style_validation',dest='file_list_txt_style_validation',type=str,required=True)

parser.add_argument('--channels',dest='channels',type=int,required=True)
parser.add_argument('--img_width',dest='img_width',type=int,required=True)



# for losses setting
parser.add_argument('--Pixel_Reconstruction_Penalty', dest='Pixel_Reconstruction_Penalty', type=float, required=True)
parser.add_argument('--Lconst_content_Penalty', dest='Lconst_content_Penalty', type=float, required=True)
parser.add_argument('--Lconst_style_Penalty', dest='Lconst_style_Penalty', type=float, required=True)
parser.add_argument('--Discriminative_Penalty', dest='Discriminative_Penalty', type=float, required=True)
parser.add_argument('--Discriminator_Categorical_Penalty', dest='Discriminator_Categorical_Penalty', type=float, required=True)
parser.add_argument('--Generator_Categorical_Penalty', dest='Generator_Categorical_Penalty', type=float, required=True)
parser.add_argument('--Discriminator_Gradient_Penalty', dest='Discriminator_Gradient_Penalty', type=float, required=True)
parser.add_argument('--generator_weight_decay_penalty', dest='generator_weight_decay_penalty', type=float, required=True)
parser.add_argument('--discriminator_weight_decay_penalty', dest='discriminator_weight_decay_penalty', type=float, required=True)
parser.add_argument('--Batch_StyleFeature_Discrimination_Penalty', dest='Batch_StyleFeature_Discrimination_Penalty', type=float, required=True)


parser.add_argument('--Feature_Penalty_True_Fake_Target', dest='Feature_Penalty_True_Fake_Target', type=float, required=True)
parser.add_argument('--Feature_Penalty_Style_Reference', dest='Feature_Penalty_Style_Reference', type=float, required=True)
parser.add_argument('--Feature_Penalty_Content_Prototype', dest='Feature_Penalty_Content_Prototype', type=float, required=True)


# training param setting
parser.add_argument('--init_lr', dest='init_lr',type=float,required=True)
parser.add_argument('--batch_size', dest='batch_size', type=int,required=True)
parser.add_argument('--final_learning_rate_pctg', dest='final_learning_rate_pctg', type=float, default=0.2)
parser.add_argument('--optimization_method',type=str,required=True)


# training time
parser.add_argument('--epoch', dest='epoch', type=int, required=True)
parser.add_argument('--init_training_epochs', dest='init_training_epochs', type=int, required=True)
parser.add_argument('--final_training_epochs', dest='final_training_epochs', type=int, required=True)





def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    cpu_device=[x.name for x in local_device_protos if x.device_type == 'CPU']
    gpu_device=[x.name for x in local_device_protos if x.device_type == 'GPU']



    for ii in range(len(gpu_device)):
        gpu_device[ii]=str(gpu_device[ii])
    for ii in range(len(cpu_device)):
        cpu_device[ii]=str(cpu_device[ii])

    print("Available CPU:%s with number:%d" % (cpu_device, len(cpu_device)))
    print("Available GPU:%s with number:%d" % (gpu_device, len(gpu_device)))
    gpu_device.sort()
    cpu_device.sort()


    return cpu_device, gpu_device,len(cpu_device),len(gpu_device)



def main(_):
    print("#####################################################")
    avalialbe_cpu, available_gpu, available_cpu_num, available_gpu_num = get_available_gpus()
    selected_device_list = list()
    if available_gpu_num == 0:
        print("No available GPU found!!! The calculation will be performed with CPU only.")
        generator_device = avalialbe_cpu[0]
        discriminator_device = avalialbe_cpu[0]
    else:
        generator_device = args.generator_device
        discriminator_device = args.discriminator_device
    selected_device_list.append(generator_device)
    selected_device_list.append(discriminator_device)

    if args.Feature_Penalty_True_Fake_Target > 10*eps or \
            args.Feature_Penalty_True_Fake_Target > 10*eps or \
            args.Feature_Penalty_True_Fake_Target > 10*eps:
        if not available_gpu_num==0:
            feature_extractor_device = args.feature_extractor_device
        else:
            feature_extractor_device = avalialbe_cpu[0]
        selected_device_list.append(feature_extractor_device)
    else:
        feature_extractor_device = 'None'


    if not available_gpu_num == 0:
        selected_device_list=list(set(selected_device_list))
        selected_device_list.sort()
        if not available_gpu == selected_device_list:
            print("GPU Selection Error!")
            print("#####################################################")
            print("Available GPU:")
            for ii in available_gpu:
                print(ii)
            print("#####################################################")
            print("Selected GPU:")
            for ii in selected_device_list:
                print(ii)
            print("#####################################################")
            return

    print("#####################################################")
    print("GeneratorDevice:%s" % generator_device)
    print("DiscriminatorDevice:%s" % discriminator_device)
    print("FeatureExtractorDevice:%s" % feature_extractor_device)
    print("#####################################################")

    content_data_dir = args.content_data_dir.split(',')
    for ii in range(len(content_data_dir)):
        content_data_dir[ii] = os.path.join(data_path_root, content_data_dir[ii])
    style_train_data_dir = args.style_train_data_dir.split(',')
    for ii in range(len(style_train_data_dir)):
        style_train_data_dir[ii] = os.path.join(data_path_root, style_train_data_dir[ii])
    style_validation_data_dir = args.style_validation_data_dir.split(',')
    for ii in range(len(style_validation_data_dir)):
        style_validation_data_dir[ii] = os.path.join(data_path_root, style_validation_data_dir[ii])

    model = WNET(debug_mode=args.debug_mode,
                 print_info_seconds=args.print_info_seconds,
                 experiment_dir=os.path.join(model_log_path_root, args.experiment_dir),
                 experiment_id=args.experiment_id,
                 log_dir=os.path.join(model_log_path_root, args.log_dir),
                 training_from_model=args.training_from_model_dir,
                 train_data_augment=args.train_data_augment,
                 train_data_augment_flip=args.train_data_augment_flip,
                 style_input_number=args.style_input_number,
                 content_input_number_actual=args.content_input_number_actual,
                 adain_use=args.adain_use,

                 content_data_dir=content_data_dir,
                 style_train_data_dir=style_train_data_dir,
                 style_validation_data_dir=style_validation_data_dir,
                 file_list_txt_content=args.file_list_txt_content.split(','),
                 file_list_txt_style_train=args.file_list_txt_style_train.split(','),
                 file_list_txt_style_validation=args.file_list_txt_style_validation.split(','),
                 channels=args.channels, epoch=args.epoch,
                 init_training_epochs=args.init_training_epochs,
                 final_training_epochs=args.final_training_epochs,

                 optimization_method=args.optimization_method,

                 batch_size=args.batch_size, img_width=args.img_width,
                 lr=args.init_lr, final_learning_rate_pctg=args.final_learning_rate_pctg,

                 Pixel_Reconstruction_Penalty=args.Pixel_Reconstruction_Penalty,
                 Lconst_content_Penalty=args.Lconst_content_Penalty,
                 Lconst_style_Penalty=args.Lconst_style_Penalty,
                 Discriminative_Penalty=args.Discriminative_Penalty,
                 Discriminator_Categorical_Penalty=args.Discriminator_Categorical_Penalty,
                 Generator_Categorical_Penalty=args.Generator_Categorical_Penalty,
                 Discriminator_Gradient_Penalty=args.Discriminator_Gradient_Penalty,
                 generator_weight_decay_penalty=args.generator_weight_decay_penalty,
                 discriminator_weight_decay_penalty=args.discriminator_weight_decay_penalty,
                 Batch_StyleFeature_Discrimination_Penalty=args.Batch_StyleFeature_Discrimination_Penalty,

                 Feature_Penalty_True_Fake_Target=args.Feature_Penalty_True_Fake_Target,
                 Feature_Penalty_Style_Reference=args.Feature_Penalty_Style_Reference,
                 Feature_Penalty_Content_Prototype=args.Feature_Penalty_Content_Prototype,

                 resume_training=args.resume_training,

                 generator_devices=generator_device,
                 discriminator_devices=discriminator_device,
                 feature_extractor_devices=feature_extractor_device,

                 generator_residual_at_layer=args.generator_residual_at_layer,
                 generator_residual_blocks=args.generator_residual_blocks,
                 discriminator=args.discriminator,
                 true_fake_target_extractor_dir=os.path.join(model_log_path_root, args.true_fake_target_extractor_dir),
                 content_prototype_extractor_dir=os.path.join(model_log_path_root, args.content_prototype_extractor_dir),
                 style_reference_extractor_dir = os.path.join(model_log_path_root, args.style_reference_extractor_dir)
                 )


    model.train_procedures()



#input_args = []
args = parser.parse_args(input_args)


if __name__ == '__main__':
    tf.app.run()
