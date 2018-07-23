# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
import sys
import os
sys.path.append('..')

from model.img2img import Img2Img
eps = 1e-9


exp_root_path = '/DataA/Harric/MSMC_Exp/'
# exp_root_path = '/Users/harric/Downloads/WNet_Exp/'


# OPTIONS SPECIFICATION
# resume_training = 0: training from stratch
#                   1: training from a based model
# training_mode = StyleInit:
#               = StyleReTrain:
#               = FineTuneClassifier: FOR DISCRIMINATOR FINE TUNE ON THE CATEGORY LOSS
input_args = [
              '--debug_mode','0',
              '--style_input_number','4', # how many style inputs
              '--init_training_epochs','5',
              '--final_training_epochs','250',

              '--generator_device','/device:GPU:0',
              '--discriminator_device', '/device:GPU:0',
              '--feature_extractor_device','/device:GPU:0',


              '--train_data_augment','1', # translation? rotation?
              '--experiment_id','20180723_StyleHw50_ContentHw32',# experiment name prefix
              '--experiment_dir','../../Exp_MSMC', # model saving location
              '--log_dir','tfLogs_MSMC/',# log file saving location
              '--print_info_seconds','900',

              '--content_data_dir', # standard data location
    'CASIA_64/HandWritingData/CASIA-HWDB1.1/,'
    'CASIA_64/HandWritingData/CASIA-HWDB2.1/',

              '--style_train_data_dir', # training data location
    'CASIA_64/HandWritingData/CASIA-HWDB1.1/,'
    'CASIA_64/HandWritingData/CASIA-HWDB2.1/',

              '--style_validation_data_dir',# validation data location
    'CASIA_64/HandWritingData/CASIA-HWDB2.1/',

              '--file_list_txt_content', # file list of the standard data
    '../FileList/HandWritingData/Char_0_3754_Writer_1001_1032_Isolated.txt,'
    '../FileList/HandWritingData/Char_0_3754_Writer_1001_1032_Cursive.txt',

              '--file_list_txt_style_train', # file list of the training data
    '../FileList/HandWritingData/Char_0_3754_Writer_1101_1150_Isolated.txt,'
    '../FileList/HandWritingData/Char_0_3754_Writer_1101_1150_Cursive.txt',

              '--file_list_txt_style_validation', # file list of the validation data
    '../FileList/HandWritingData/Char_0_3754_Writer_1296_1300_Cursive.txt',

              # pre-trained feature extractor to build the feature loss for the generator
              '--feature_extractor','extr_vgg16net',
              '--feature_extractor_model_dir',
    'TrainedModel_ExtraNet_WithWeightDecay/Exp20180514_Hw50_vgg16net/variables/',


              # generator && discriminator
              '--generator_residual_at_layer','3',
              '--generator_residual_blocks','5',
              '--discriminator','DisMdy6conv',

              '--batch_size','32',
              '--img_width','64',
              '--channels','1',

              # optimizer parameters
              '--init_lr','0.0005',
              '--epoch','5000',
              '--resume_training','0', # 0: training from scratch; 1: training from a pre-trained point

              '--optimization_method','adam',
              '--final_learning_rate_pctg','0.01',


              # penalties
              '--generator_weight_decay_penalty','0.0001',
              '--discriminator_weight_decay_penalty','0.0003',
              '--L1_Penalty','50',
              '--Feature_Penalty','80',
              '--Lconst_content_Penalty','3',
              '--Lconst_style_Penalty','5',
              '--Discriminative_Penalty', '3',
              '--Discriminator_Categorical_Penalty', '1',
              '--Generator_Categorical_Penalty', '0.2',
              '--Discriminator_Gradient_Penalty', '10']



parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--debug_mode', dest='debug_mode',type=int,required=True)
parser.add_argument('--resume_training', dest='resume_training', type=int,required=True)
parser.add_argument('--train_data_augment', dest='train_data_augment', type=int,required=True)
parser.add_argument('--print_info_seconds', dest='print_info_seconds',type=int,required=True)
parser.add_argument('--style_input_number', dest='style_input_number', type=int,required=True)


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


parser.add_argument('--feature_extractor', dest='feature_extractor', type=str, required=True)
parser.add_argument('--feature_extractor_device', dest='feature_extractor_device',type=str,required=True)
parser.add_argument('--feature_extractor_model_dir', dest='feature_extractor_model_dir', type=str, required=True)



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
parser.add_argument('--L1_Penalty', dest='L1_Penalty', type=int, required=True)
parser.add_argument('--Feature_Penalty', dest='Feature_Penalty', type=int, required=True)
parser.add_argument('--Lconst_content_Penalty', dest='Lconst_content_Penalty', type=int, required=True)
parser.add_argument('--Lconst_style_Penalty', dest='Lconst_style_Penalty', type=int, required=True)
parser.add_argument('--Discriminative_Penalty', dest='Discriminative_Penalty', type=int, required=True)
parser.add_argument('--Discriminator_Categorical_Penalty', dest='Discriminator_Categorical_Penalty', type=int, required=True)
parser.add_argument('--Generator_Categorical_Penalty', dest='Generator_Categorical_Penalty', type=float, required=True)
parser.add_argument('--Discriminator_Gradient_Penalty', dest='Discriminator_Gradient_Penalty', type=int, required=True)
parser.add_argument('--generator_weight_decay_penalty', dest='generator_weight_decay_penalty', type=float, required=True)
parser.add_argument('--discriminator_weight_decay_penalty', dest='discriminator_weight_decay_penalty', type=float, required=True)


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


    if args.Feature_Penalty > 10*eps \
        and not args.feature_extractor_device =='None' \
        and not args.feature_extractor == 'None' \
        and not args.feature_extractor_model_dir == 'None':

        if not available_gpu_num==0:
            feature_extractor_device = args.feature_extractor_device
        else:
            feature_extractor_device = avalialbe_cpu[0]
        selected_device_list.append(feature_extractor_device)
    else:
        feature_extractor_device = 'None'
        args.feature_extractor = 'None'
        args.feature_extractor_model_dir = 'None'
        args.Feature_Penalty = 0


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
        content_data_dir[ii] = os.path.join(exp_root_path, content_data_dir[ii])
    style_train_data_dir = args.style_train_data_dir.split(',')
    for ii in range(len(style_train_data_dir)):
        style_train_data_dir[ii] = os.path.join(exp_root_path, style_train_data_dir[ii])
    style_validation_data_dir = args.style_validation_data_dir.split(',')
    for ii in range(len(style_validation_data_dir)):
        style_validation_data_dir[ii] = os.path.join(exp_root_path, style_validation_data_dir[ii])

    model = Img2Img(debug_mode=args.debug_mode,
                    print_info_seconds=args.print_info_seconds,
                    experiment_dir=args.experiment_dir, experiment_id=args.experiment_id,
                    log_dir=os.path.join(exp_root_path, args.log_dir),
                    training_from_model=args.training_from_model_dir,
                    train_data_augment=args.train_data_augment,
                    style_input_number=args.style_input_number,

                    content_data_dir=content_data_dir,
                    style_train_data_dir=style_train_data_dir,
                    style_validation_data_dir=style_validation_data_dir,
                    file_list_txt_content=args.file_list_txt_content.split(','),
                    file_list_txt_style_train=args.file_list_txt_style_train.split(','),
                    file_list_txt_style_validation=args.file_list_txt_style_validation.split(','),
                    channels=args.channels,
                    epoch=args.epoch,
                    init_training_epochs=args.init_training_epochs,
                    final_training_epochs=args.final_training_epochs,

                    optimization_method=args.optimization_method,

                    batch_size=args.batch_size, img_width=args.img_width,
                    lr=args.init_lr, final_learning_rate_pctg=args.final_learning_rate_pctg,

                    L1_Penalty=args.L1_Penalty,
                    Feature_Penalty=args.Feature_Penalty,
                    Lconst_content_Penalty=args.Lconst_content_Penalty,
                    Lconst_style_Penalty=args.Lconst_style_Penalty,
                    Discriminative_Penalty=args.Discriminative_Penalty,
                    Discriminator_Categorical_Penalty=args.Discriminator_Categorical_Penalty,
                    Generator_Categorical_Penalty=args.Generator_Categorical_Penalty,
                    Discriminator_Gradient_Penalty=args.Discriminator_Gradient_Penalty,
                    generator_weight_decay_penalty=args.generator_weight_decay_penalty,
                    discriminator_weight_decay_penalty=args.discriminator_weight_decay_penalty,

                    resume_training=args.resume_training,

                    generator_devices=generator_device,
                    discriminator_devices=discriminator_device,
                    feature_extractor_devices=feature_extractor_device,

                    generator_residual_at_layer=args.generator_residual_at_layer,
                    generator_residual_blocks=args.generator_residual_blocks,
                    discriminator=args.discriminator,
                    feature_extractor=args.feature_extractor,
                    feature_extractor_model_dir=os.path.join(exp_root_path, args.feature_extractor_model_dir),
                    )


    model.train_procedures()



#input_args = []
args = parser.parse_args(input_args)


if __name__ == '__main__':
    tf.app.run()
