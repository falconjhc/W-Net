# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
import sys
sys.path.append('..')
import os
import time

from model.wnet_forInferEval import WNet as WNET

# model_root = '/Data_HDD/Harric/ChineseCharacterExp/'
# data_root = '/home/harric/ChineseCharacterExp/'
model_root = '/Users/harric/ChineseCharacterExp/'
data_root = '/Users/harric/ChineseCharacterExp/'


# W-Net
# evaluating_generator_dir=\
#     'tfModels2019_WNet/checkpoint/Exp20190129-WNet-DenseMixer-AdaIN-Multi_StyleHw50_ContentPf32+Hw32_GenEncDec6-Des5@Lyr3_DisMdy6conv/generator/'
# evaluating_generator_dir=\
#     'tfModels2019_WNet/checkpoint/Exp20190129-WNet-DenseMixer-AdaIN-Single_StyleHw50_ContentPf32+Hw32_GenEncDec6-Des5@Lyr3_DisMdy6conv/generator/'
# evaluating_generator_dir=\
#     'tfModels2019_WNet/checkpoint/Exp20190129-WNet-DenseMixer-NonAdaIN_StyleHw50_ContentPf32+Hw32_GenEncDec6-Des5@Lyr3_DisMdy6conv/generator/'
# evaluating_generator_dir=\
#     'tfModels2019_WNet/checkpoint/Exp20190129-WNet-ResidualMixer-AdaIN-Multi_StyleHw50_ContentPf32+Hw32_GenEncDec6-Res5@Lyr3_DisMdy6conv/generator/'
# evaluating_generator_dir=\
#     'tfModels2019_WNet/checkpoint/Exp20190129-WNet-ResidualMixer-AdaIN-Single_StyleHw50_ContentPf32+Hw32_GenEncDec6-Res5@Lyr3_DisMdy6conv/generator/'
# evaluating_generator_dir=\
#     'tfModels2019_WNet/checkpoint/Exp20190129-WNet-ResidualMixer-NonAdaIN_StyleHw50_ContentPf32+Hw32_GenEncDec6-Res5@Lyr3_DisMdy6conv/generator/'

# Adobe-Net
# evaluating_generator_dir=\
#     'tfModels2019_AdobeNet/checkpoint/Exp20190129-AdobeNet-Style1_StyleHw50_ContentPf32+Hw32_GenEncDec6_DisMdy6conv/generator/'

# Emd-Net
# evaluating_generator_dir=\
#     'tfModels2019_EmdNet/checkpoint/Exp20190129-ResEmdNet-NN-Style1_StyleHw50_ContentPf32+Hw32_GenEncDec6_DisMdy6conv/generator/'
# evaluating_generator_dir=\
#     'tfModels2019_EmdNet/checkpoint/Exp20190129-ResEmdNet-Style1_StyleHw50_ContentPf32+Hw32_GenEncDec6_DisMdy6conv/generator/'
# evaluating_generator_dir=\
#     'tfModels2019_EmdNet/checkpoint/Exp20190129-EmdNet-Style1-AdaIN_StyleHw50_ContentPf32+Hw32_GenEncDec6_DisMdy6conv/generator/'
evaluating_generator_dir=\
    'tfModels2019_EmdNet/checkpoint/Exp20190129-EmdNet-Style1-NonAdaIN_StyleHw50_ContentPf32+Hw32_GenEncDec6_DisMdy6conv/generator/'


# ResMixer-Net
# evaluating_generator_dir=\
#     'tfModels2019_ResMixerNet/checkpoint/Exp20190129-ResMixer-5-DenseMixer-Style1_StyleHw50_ContentPf32+Hw32_GenEncDec6_DisMdy6conv/generator/'
# evaluating_generator_dir=\
#     'tfModels2019_ResMixerNet/checkpoint/Exp20190129-ResMixer-5-SimpleMixer-Style1_StyleHw50_ContentPf32+Hw32_GenEncDec6_DisMdy6conv/generator/'




style_input_number=4
save_path = '/Data_HDD/Harric/ChineseCharacterExp/GeneratedResult/Generated_201901/TrainStyle/'

# content prototype setting
content_data_dir=list()
content_data_dir.append('CASIA_Dataset/HandWritingData_240Binarized/CASIA-HWDB1.1/')
content_data_dir.append('CASIA_Dataset/HandWritingData_240Binarized/CASIA-HWDB2.1/')
content_data_dir.append('CASIA_Dataset/PrintedData/')
file_list_txt_content=list()
file_list_txt_content.append('../FileList/HandWritingData/Char_0_3754_Writer_1001_1032_Isolated.txt')
file_list_txt_content.append('../FileList/HandWritingData/Char_0_3754_Writer_1001_1032_Cursive.txt')
file_list_txt_content.append('../FileList/PrintedData/Char_0_3754_Writer_Selected32_Printed_Fonts_GB2312L1.txt')


# content to be generated setting
targeted_content_input_txt='../ContentTxt/ContentChars_BlancaPython_32.txt'

# style data setting
style_data_dir=list()
style_data_dir.append('CASIA_Dataset/HandWritingData_240Binarized/CASIA-HWDB2.1/')
style_data_dir.append('CASIA_Dataset/HandWritingData_240Binarized/CASIA-HWDB1.1/')

# true style setting
file_list_txt_true_style=list()
file_list_txt_true_style.append('../EvaluationDataFileLists/HandWritingData/ContentChar_BlancaPython_Writer_1101_1150_Cursive.txt')
file_list_txt_true_style.append('../EvaluationDataFileLists/HandWritingData/ContentChar_BlancaPython_Writer_1101_1150_Isolated.txt')

# input style setting
targeted_style_input_txt='../ContentTxt/StyleChars_Paintings_20.txt'
file_list_txt_input_style=list()
file_list_txt_input_style.append('../EvaluationDataFileLists/HandWritingData/StyleChars_Paintings_Writer_1101_1150_Cursive.txt')
file_list_txt_input_style.append('../EvaluationDataFileLists/HandWritingData/StyleChars_Paintings_Writer_1101_1150_Isolated.txt')









print_separater = "#################################################################"




input_args = ['--generator_residual_at_layer','3',
              '--generator_residual_blocks','5',
              '--generator_device','/device:GPU:0']

parser = argparse.ArgumentParser(description='Eval')



# network settings
parser.add_argument('--generator_residual_at_layer', dest='generator_residual_at_layer', type=int, required=True)
parser.add_argument('--generator_residual_blocks', dest='generator_residual_blocks', type=int, required=True)
parser.add_argument('--generator_device', dest='generator_device',type=str,required=True, help='Devices for generator')






def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    cpu_device=[x.name for x in local_device_protos if x.device_type == 'CPU']
    gpu_device=[x.name for x in local_device_protos if x.device_type == 'GPU']
    print("Available CPU:%s with number:%d" % (cpu_device, len(cpu_device)))
    print("Available GPU:%s with number:%d" % (gpu_device, len(gpu_device)))
    return cpu_device, gpu_device,len(cpu_device),len(gpu_device)



def main(_):
    print(print_separater)
    avalialbe_cpu, available_gpu, available_cpu_num, available_gpu_num = get_available_gpus()

    if available_gpu_num==0:
        args.generator_device = '/device:CPU:0'

    for ii in range(len(content_data_dir)):
        content_data_dir[ii] = os.path.join(data_root, content_data_dir[ii])
    for ii in range(len(style_data_dir)):
        style_data_dir[ii] = os.path.join(data_root, style_data_dir[ii])

    experiment_id_list = evaluating_generator_dir.split('/')
    for experiment_id in experiment_id_list:
        if 'Exp' in experiment_id:
            break

    if 'Style4' in experiment_id:
        style_input_number=4
    elif 'Style1' in experiment_id:
        style_input_number=1
    else:
        global style_input_number
        experiment_id = experiment_id+'-Style%d' % style_input_number

    global save_path
    save_path = os.path.join(save_path, experiment_id)

    model = WNET(style_input_number=style_input_number,
                 experiment_id=experiment_id,

                 targeted_content_input_txt=targeted_content_input_txt,
                 targeted_style_input_txt=targeted_style_input_txt,
                 save_path=save_path,

                 content_data_dir=content_data_dir,
                 file_list_txt_content=file_list_txt_content,
                 style_data_dir=style_data_dir,
                 file_list_txt_true_style=file_list_txt_true_style,
                 file_list_txt_input_style=file_list_txt_input_style,


                 generator_residual_at_layer=args.generator_residual_at_layer,
                 generator_residual_blocks=args.generator_residual_blocks,
                 generator_devices=args.generator_device,

                 model_dir=os.path.join(model_root,evaluating_generator_dir))

    model.character_generation()


#input_args = []
args = parser.parse_args(input_args)


if __name__ == '__main__':
    tf.app.run()
