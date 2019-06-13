# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('..')
import os
import time

from model.wnet_forInferEval import WNet as WNET

model_root = '/Data_HDD/Harric/ChineseCharacterExp/'
data_root = '/DataA/Harric/ChineseCharacterExp/'
save_path = '/Data_HDD/Harric/ChineseCharacterExp/GeneratedResult/Generated_201904/TestStyle/'

flags = tf.flags
flags.DEFINE_integer("style_input_number","-1","StyleReferenceNum")
flags.DEFINE_string("evaluating_generator_dir","-1","ModelDirLoad")

FLAGS = flags.FLAGS
style_input_number=FLAGS.style_input_number
evaluating_generator_dir=FLAGS.evaluating_generator_dir


# content prototype setting
content_data_dir=list()
content_data_dir.append('CASIA_Dataset/PrintedData_64Fonts/Simplified/GB2312_L1/')
file_list_txt_content=list()
file_list_txt_content.append('../FileList/PrintedData/Char_0_3754_64PrintedFonts_GB2312L1_Simplified.txt')


# content to be generated setting
targeted_content_input_txt='../ContentTxt/ContentChars_BlancaPython_32.txt'

# style data setting
style_data_dir=list()
style_data_dir.append('CASIA_Dataset/HandWritingData_240Binarized/CASIA-HWDB2.1/')
style_data_dir.append('CASIA_Dataset/HandWritingData_240Binarized/CASIA-HWDB1.1/')

# true style setting
file_list_txt_true_style=list()
file_list_txt_true_style.append('../EvaluationDataFileLists/HandWritingData/ContentChar_BlancaPython_Writer_1151_1200_Cursive.txt')
file_list_txt_true_style.append('../EvaluationDataFileLists/HandWritingData/ContentChar_BlancaPython_Writer_1151_1200_Isolated.txt')

# input style setting
targeted_style_input_txt='../ContentTxt/StyleChars_Paintings_20.txt'
file_list_txt_input_style=list()
file_list_txt_input_style.append('../EvaluationDataFileLists/HandWritingData/StyleChars_Paintings_Writer_1151_1200_Cursive.txt')
file_list_txt_input_style.append('../EvaluationDataFileLists/HandWritingData/StyleChars_Paintings_Writer_1151_1200_Isolated.txt')










print_separater = "#################################################################"

input_args = ['--generator_residual_at_layer','3',
              '--generator_residual_blocks','1',
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


    
    global evaluating_generator_dir
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
