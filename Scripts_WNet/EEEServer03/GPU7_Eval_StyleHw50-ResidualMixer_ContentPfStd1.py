# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
import sys
import os
sys.path.append('../../')

from model.wnet_forEvaluation import WNet as WNET
eps = 1e-9



data_path_root = '/home/harric/ChineseCharacterExp/'
model_path_root = '/Data_HDD/Harric/ChineseCharacterExp/'

content_known_unknown='Known'
style_known_unknown='Known'
style_input_number=4
evaluation_resule_save_dir = '/Data_HDD/Harric/ChineseCharacterExp/EvalResult/EvaluationResult_201901/'

# W-Net
evaluating_generator_dir=\
    'tfModels2019_WNet/checkpoint/Exp20190129-WNet-ResidualMixer-NonAdaIN_StyleHw50_ContentPfStd1_GenEncDec6-Res5@Lyr3_DisMdy6conv/generator/'
# evaluating_generator_dir=\
#     'tfModels2019_WNet/checkpoint/Exp20190213-WNet-DenseMixer-NonAdaIN_StyleHw50_ContentPfStd1_GenEncDec6-Des5@Lyr3_DisMdy6conv/generator/'



## content dir
content_data_dir=list()
content_data_dir.append('CASIA_Dataset/StandardChars/GB2312_L1/')

# style dir
style_train_data_dir=list()
style_train_data_dir.append('CASIA_Dataset/HandWritingData_240Binarized/CASIA-HWDB1.1/')
style_train_data_dir.append('CASIA_Dataset/HandWritingData_240Binarized/CASIA-HWDB2.1/')

# fixed style dir
fixed_style_data_dir=list()
fixed_style_data_dir.append('CASIA_Dataset/HandWritingData_240Binarized/CASIA-HWDB1.1/')
fixed_style_data_dir.append('CASIA_Dataset/HandWritingData_240Binarized/CASIA-HWDB2.1/')



## known content
known_content_file_list=list()
known_content_file_list.append('../../TrainTestFileList/StandardChars/Char_0_3754_GB2312L1_Train.txt')

## unknown content
unknown_content_file_list=list()
unknown_content_file_list.append('../../TrainTestFileList/StandardChars/Char_0_3754_GB2312L1_Test.txt')



## known style
known_style_file_list=list()
if content_known_unknown == 'Known':
    known_style_file_list.append('../../TrainTestFileList/HandWritingData/Char_0_3754_Writer_1101_1150_Isolated_Train.txt')
    known_style_file_list.append('../../TrainTestFileList/HandWritingData/Char_0_3754_Writer_1101_1150_Cursive_Train.txt')
else:
    known_style_file_list.append('../../TrainTestFileList/HandWritingData/Char_0_3754_Writer_1101_1150_Isolated_Test.txt')
    known_style_file_list.append('../../TrainTestFileList/HandWritingData/Char_0_3754_Writer_1101_1150_Cursive_Test.txt')
known_fixed_style_file_list=list()
known_fixed_style_file_list.append('../../EvaluationDataFileLists/HandWritingData/StyleChars_Paintings_Writer_1101_1150_Isolated.txt')
known_fixed_style_file_list.append('../../EvaluationDataFileLists/HandWritingData/StyleChars_Paintings_Writer_1101_1150_Cursive.txt')


## unknown style
unknown_style_file_list=list()
if content_known_unknown == 'Known':
    unknown_style_file_list.append('../../TrainTestFileList/HandWritingData/Char_0_3754_Writer_1151_1200_Isolated_Train.txt')
    unknown_style_file_list.append('../../TrainTestFileList/HandWritingData/Char_0_3754_Writer_1151_1200_Cursive_Train.txt')
else:
    unknown_style_file_list.append('../../TrainTestFileList/HandWritingData/Char_0_3754_Writer_1151_1200_Isolated_Test.txt')
    unknown_style_file_list.append('../../TrainTestFileList/HandWritingData/Char_0_3754_Writer_1151_1200_Cursive_Test.txt')
unknown_fixed_style_file_list=list()
unknown_fixed_style_file_list.append('../../EvaluationDataFileLists/HandWritingData/StyleChars_Paintings_Writer_1151_1200_Isolated.txt')
unknown_fixed_style_file_list.append('../../EvaluationDataFileLists/HandWritingData/StyleChars_Paintings_Writer_1151_1200_Cursive.txt')


input_args = [

    '--debug_mode','0',
    '--generator_device','/device:GPU:0',
    '--feature_extractor_device','/device:GPU:0',


    '--fixed_char_list_txt',
    '../../ContentTxt/StyleChars_Paintings_20.txt',


    # generator
    '--generator_residual_at_layer','3',
    '--generator_residual_blocks','5',

    '--batch_size','8',

    # feature extractor parametrers
    '--true_fake_target_extractor_dir',
    'tfModels_FeatureExtractor/checkpoint/Exp20190119_FeatureExtractor_ContentStyle_HW300Pf144_vgg16net/variables/',
    '--content_prototype_extractor_dir',
    'tfModels_FeatureExtractor/checkpoint/Exp20190119_FeatureExtractor_Content_HW300Pf144_vgg16net/variables/',
    '--style_reference_extractor_dir',
    'tfModels_FeatureExtractor/checkpoint/Exp20190119_FeatureExtractor_Style_HW300Pf144_vgg16net/variables/',

]


parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--debug_mode', dest='debug_mode',type=int,required=True)




# network settings
parser.add_argument('--generator_device', dest='generator_device',type=str,required=True)
parser.add_argument('--generator_residual_at_layer', dest='generator_residual_at_layer', type=int, required=True)
parser.add_argument('--generator_residual_blocks', dest='generator_residual_blocks', type=int, required=True)





parser.add_argument('--feature_extractor_device', dest='feature_extractor_device',type=str,required=True)
parser.add_argument('--true_fake_target_extractor_dir', dest='true_fake_target_extractor_dir', type=str, required=True)
parser.add_argument('--style_reference_extractor_dir', dest='style_reference_extractor_dir', type=str, required=True)
parser.add_argument('--content_prototype_extractor_dir', dest='content_prototype_extractor_dir', type=str, required=True)


# input data setting
parser.add_argument('--fixed_char_list_txt',dest='fixed_char_list_txt',type=str,required=True)



parser.add_argument('--channels',dest='channels',type=int,default=1)
parser.add_argument('--img_width',dest='img_width',type=int,default=64)







# training param setting
parser.add_argument('--batch_size', dest='batch_size', type=int,required=True)







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
        feature_extractor_device = avalialbe_cpu[0]
    else:
        generator_device = args.generator_device
        feature_extractor_device = args.feature_extractor_device
    selected_device_list.append(generator_device)

    experiment_id_list = evaluating_generator_dir.split('/')
    for experiment_id in experiment_id_list:
        if 'Exp' in experiment_id:
            break



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
    print("FeatureExtractorDevice:%s" % feature_extractor_device)
    print("#####################################################")

    # content data process
    if content_known_unknown == 'Known':
        content_file_list = known_content_file_list
        experiment_id = experiment_id + '-ContentKnown'
    elif content_known_unknown == 'UnKnown':
        content_file_list = unknown_content_file_list
        experiment_id = experiment_id + '-ContentUnKnown'
    for ii in range(len(content_data_dir)):
        content_data_dir[ii] = os.path.join(data_path_root, content_data_dir[ii])

    # style data process and fixed style data process
    if style_known_unknown == 'Known':
        style_file_list = known_style_file_list
        fixed_style_file_list = known_fixed_style_file_list
        experiment_id = experiment_id + '-StyleKnown'
    elif style_known_unknown == 'UnKnown':
        style_file_list = unknown_style_file_list
        fixed_style_file_list = unknown_fixed_style_file_list
        experiment_id = experiment_id + '-StyleUnKnown'
    for ii in range(len(style_train_data_dir)):
        style_train_data_dir[ii] = os.path.join(data_path_root, style_train_data_dir[ii])
    for ii in range(len(fixed_style_data_dir)):
        fixed_style_data_dir[ii] = os.path.join(data_path_root, fixed_style_data_dir[ii])


    if 'Style4' in experiment_id:
        style_input_number=4
    elif 'Style1' in experiment_id:
        style_input_number=1
    else:
        global style_input_number
        experiment_id = experiment_id+'-Style%d' % style_input_number


    model = WNET(debug_mode=args.debug_mode,
                 experiment_id=experiment_id,
                 evaluation_resule_save_dir=evaluation_resule_save_dir,
                 style_input_number=style_input_number,
                 content_data_dir=content_data_dir,
                 style_train_data_dir=style_train_data_dir,
                 fixed_style_reference_dir=fixed_style_data_dir,
                 fixed_file_list_txt_style_reference=fixed_style_file_list,
                 file_list_txt_content=content_file_list,
                 file_list_txt_style_train=style_file_list,
                 fixed_char_list_txt=args.fixed_char_list_txt,
                 channels=args.channels,
                 batch_size=args.batch_size, img_width=args.img_width,
                 generator_devices=generator_device,
                 feature_extractor_devices=feature_extractor_device,
                 generator_residual_at_layer=args.generator_residual_at_layer,
                 generator_residual_blocks=args.generator_residual_blocks,
                 true_fake_target_extractor_dir=os.path.join(model_path_root, args.true_fake_target_extractor_dir),
                 content_prototype_extractor_dir=os.path.join(model_path_root, args.content_prototype_extractor_dir),
                 style_reference_extractor_dir = os.path.join(model_path_root, args.style_reference_extractor_dir),
                 evaluating_generator_dir=os.path.join(model_path_root, evaluating_generator_dir))


    model.evaluate_process()



#input_args = []
args = parser.parse_args(input_args)


if __name__ == '__main__':
    tf.app.run()
