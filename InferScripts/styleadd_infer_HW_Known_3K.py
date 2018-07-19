# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
import sys
sys.path.append('..')

from model.img2img import Img2Img

# OPTIONS SPECIFICATION
# resume_training = 0: training from stratch
#                   1: training from a based model


input_args = ['--debug_mode','1',
              '--infer_mode','0',

              '--targeted_chars_txt',
    '../FontAndChars/滾滾長江東逝水_繁体.txt',
              '--infer_dir',
    '/home/harric/Desktop/InferResults/W-Net/20180528_Sample32/滾滾長江東逝水_繁体/HW_From30Bases/KnownFonts/',
              '--save_mode','8:8',


              '--style_add_target_model_dir',
    '/DataA/Harric/CASIA_64/HandWritingData/SingleChars/CASIA-HWDB1.1/',

              '--style_add_target_file_list',
    '../FileList/HandWritingData/SingleChars/char_0_3754_writer_1001_1300_gb2312l1_isolated.txt',

              '--target_label1_selection','1101:1:1130',
              '--style_input_number','32',

              '--channels','1',
              '--img_width', '64',

              '--generator_residual_at_layer','3',
              '--generator_residual_blocks','5',
              '--discriminator','DisMdy6conv',

              '--generator_device','/device:GPU:0',
              '--discriminator_device','/device:GPU:0',
    
              '--model_dir',
    '../../TrainedModel/Exp20180418_MultiInput_Hw30_StyleAdd_ExtrcVgg16_GenEncDec6-Res5@Lyr3_DisMdy6conv/',

                '--softmax_temperature','1',
              '--batch_size', '8']

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--debug_mode', dest='debug_mode',type=int,required=True)
parser.add_argument('--softmax_temperature', dest='softmax_temperature', type=float, help='training_mode',required=True)


# directories setting
parser.add_argument('--source_font', dest='source_font', type=str,default='../FontAndChars/standard_font.ttf')
parser.add_argument('--targeted_chars_txt', dest='targeted_chars_txt', type=str,required=True)
parser.add_argument('--infer_dir', dest='infer_dir', type=str,required=True)
parser.add_argument('--save_mode', dest='save_mode', type=str,required=True)
parser.add_argument('--infer_mode', dest='infer_mode', type=int,required=True)



# network settings
parser.add_argument('--discriminator', dest='discriminator', type=str, required=True)
parser.add_argument('--model_dir', dest='model_dir', required=True, type=str)

parser.add_argument('--generator_residual_at_layer', dest='generator_residual_at_layer', type=int, required=True)
parser.add_argument('--generator_residual_blocks', dest='generator_residual_blocks', type=int, required=True)


parser.add_argument('--generator_device', dest='generator_device',type=str,required=True,
                    help='Devices for generator')
parser.add_argument('--discriminator_device', dest='discriminator_device',type=str,required=True,
                    help='Devices mode selection for discriminator')

# input data setting
parser.add_argument('--style_add_target_model_dir',dest='style_add_target_model_dir',type=str,required=True)
parser.add_argument('--style_add_target_file_list',dest='style_add_target_file_list',type=str,required=True)
parser.add_argument('--target_label1_selection',dest='target_label1_selection',type=str,required=True)
parser.add_argument('--style_input_number',dest='style_input_number',type=int,required=True)


parser.add_argument('--channels',dest='channels',type=int,required=True)
parser.add_argument('--img_width',dest='img_width',type=int,required=True)

# training param setting
parser.add_argument('--batch_size', dest='batch_size', type=int, help='number of examples in batch',required=True)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    cpu_device=[x.name for x in local_device_protos if x.device_type == 'CPU']
    gpu_device=[x.name for x in local_device_protos if x.device_type == 'GPU']
    print("Available CPU:%s with number:%d" % (cpu_device, len(cpu_device)))
    print("Available GPU:%s with number:%d" % (gpu_device, len(gpu_device)))
    return cpu_device, gpu_device,len(cpu_device),len(gpu_device)



def main(_):
    avalialbe_cpu, available_gpu, available_cpu_num, available_gpu_num = get_available_gpus()

    if available_gpu_num==0:
        args.discriminator_device = '/device:CPU:0'
        args.generator_device = '/device:CPU:0'

    if ',' in args.target_label1_selection:
        target_label1_selection = args.target_label1_selection.split(',')
    elif ':' in args.target_label1_selection:
        tmp = args.target_label1_selection.split(':')
        target_label1_selection = range(int(tmp[0]), int(tmp[2])+1, int(tmp[1]))


    model = Img2Img(debug_mode=args.debug_mode,

                    targeted_chars_txt=args.targeted_chars_txt,
                    infer_dir=args.infer_dir,
                    save_mode=args.save_mode,
                    source_font=args.source_font,

                    style_add_target_model_dir=args.style_add_target_model_dir.split(','),
                    style_add_target_file_list=args.style_add_target_file_list.split(','),
                    target_label1_selection=target_label1_selection,
                    style_input_number=args.style_input_number,

                    channels=args.channels,
                    img_width=args.img_width,

                    discriminator=args.discriminator,
                    generator_residual_at_layer=args.generator_residual_at_layer,
                    generator_residual_blocks=args.generator_residual_blocks,

                    generator_devices=args.generator_device,
                    discriminator_devices=args.discriminator_device,

                    model_dir=args.model_dir,
                    softmax_temperature=args.softmax_temperature,

                    batch_size=args.batch_size)


    if args.infer_dir == 0:
        model.styleadd_infer_procedures_mode0()
    else:
        model.styleadd_infer_procedures_mode1()




#input_args = []
args = parser.parse_args(input_args)


if __name__ == '__main__':
    tf.app.run()
