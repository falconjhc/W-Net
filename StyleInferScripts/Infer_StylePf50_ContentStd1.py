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


from model.wnet import WNet as WNET

#exp_root_path = '/Users/harric/ChineseCharacterExp/'
exp_root_path = '/DataA/Harric/ChineseCharacterExp/'

print_separater = "#################################################################"




input_args = [
              '--targeted_content_input_txt',
    '../ContentTxt/过秦论_繁体_220.txt',
    		  '--save_mode','10:22',

    		  '--known_style_img_path',
    '../StyleExamples/Brush4.jpeg',            # input a image with multiple written chars
    #'../FontFiles/TTTGB-Medium.ttf', # input a ttf / otf file to generate printed chars
    # '../StyleExamples/PrintedSamples',  # input a image directory with multiple single chars

              '--content_data_dir', # standard data location
    '../FontFiles/HeiTi_Chinese.ttf',
    #'../FontFiles/HeiTi_Korean.ttf',
    #'../FontFiles/HeiTi_Jap1.otf',


  ####################################################################
  ####################################################################
  #################### DO NOT TOUCH BELOW ############################
  ####################################################################
  ####################################################################

              '--save_path',
    '../../GeneratedChars/'+ time.strftime('%Y-%m-%d@%H:%M:%S', time.localtime())+'/',
              
    		      '--debug_mode','0',
              '--style_input_number','4',

              '--file_list_txt_content',  'N/A',
              
              '--channels','1',
              '--img_width', '64',

              '--generator_residual_at_layer','3',
              '--generator_residual_blocks','5',

              '--generator_device','/device:GPU:0',

              '--model_dir',
    'TrainedModels_WNet/Exp20181115_StylePf50_ContentPfStd1_GenEncDec6-Res5@Lyr3_DisMdy6conv/generator/',

              ]

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--debug_mode', dest='debug_mode',type=int,required=True)
parser.add_argument('--style_input_number', dest='style_input_number', type=int,required=True)

# directories setting
parser.add_argument('--targeted_content_input_txt', dest='targeted_content_input_txt', type=str,required=True)
parser.add_argument('--save_path', dest='save_path', type=str,required=True)
parser.add_argument('--save_mode', dest='save_mode', type=str,required=True)



# network settings
parser.add_argument('--model_dir', dest='model_dir', required=True, type=str)
parser.add_argument('--known_style_img_path', dest='known_style_img_path', required=True, type=str)
parser.add_argument('--generator_residual_at_layer', dest='generator_residual_at_layer', type=int, required=True)
parser.add_argument('--generator_residual_blocks', dest='generator_residual_blocks', type=int, required=True)


parser.add_argument('--generator_device', dest='generator_device',type=str,required=True,
                    help='Devices for generator')

# input data setting
parser.add_argument('--content_data_dir',dest='content_data_dir',type=str,required=True)
parser.add_argument('--file_list_txt_content',dest='file_list_txt_content',type=str,required=True)


parser.add_argument('--channels',dest='channels',type=int,required=True)
parser.add_argument('--img_width',dest='img_width',type=int,required=True)

# training param setting


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

    content_data_dir = args.content_data_dir.split(',')
    for ii in range(len(content_data_dir)):
        if not (os.path.splitext(content_data_dir[ii])[-1] == '.ttc' \
                or os.path.splitext(content_data_dir[ii])[-1] == '.ttf' \
                or os.path.splitext(content_data_dir[ii])[-1] == '.otf' ):
            content_data_dir[ii] = os.path.join(exp_root_path, content_data_dir[ii])

    model = WNET(debug_mode=args.debug_mode,
                 style_input_number=args.style_input_number,

                 targeted_content_input_txt=args.targeted_content_input_txt,
                 save_path=args.save_path,
                 save_mode=args.save_mode,

                 content_data_dir=content_data_dir,
                 file_list_txt_content=args.file_list_txt_content.split(','),

                 channels=args.channels,
                 img_width=args.img_width,

                 generator_residual_at_layer=args.generator_residual_at_layer,
                 generator_residual_blocks=args.generator_residual_blocks,
                 generator_devices=args.generator_device,

                 model_dir=os.path.join(exp_root_path,args.model_dir),
                 known_style_img_path=args.known_style_img_path)

    model.character_generation()


#input_args = []
args = parser.parse_args(input_args)


if __name__ == '__main__':
    tf.app.run()
