# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import sys
sys.path.append('..')
import os
import tensorflow as tf
import argparse
from model.feature_extractor_training import train_procedures

exp_root_path = '/DataA/Harric/MSMC_Exp/'
# exp_root_path = '/Users/harric/Downloads/WNet_Exp/'

input_args = [
            '--data_dir_train_path',
            'CASIA_64/PrintedData/GB2312_L1/',

              '--data_dir_validation_path',
            'CASIA_64/PrintedData/GB2312_L1/',

              '--file_list_txt_train',
            '../FileList/PrintedData/Char_0_3754_Font_0_49_GB2312L1.txt',

              '--file_list_txt_validation',
            '../FileList/PrintedData/Char_0_3754_Font_0_49_GB2312L1.txt',

              '--experiment_dir',
            '../../Exp_FeatureExtractor/',

              '--log_dir',
              'tfLogs_FeatureExtractor/',

              '--image_filters','1',
              '--experiment_id','20180802_FeatureExtractor_StyleContent_PF50',
              '--train_resume_mode','1',

              '--batch_size','64',
              '--image_size','64',
              '--epoch_num', '1000',
              '--network', 'vgg16net',
              '--init_lr','0.0001',
              '--label0_loss','1',
              '--label1_loss','1',
              '--center_loss_penalty_rate','0',

              '--debug_mode','0',
              '--cheat_mode','0']


parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--data_dir_train_path', dest='data_dir_train_path',type=str,required=True)
parser.add_argument('--data_dir_validation_path', dest='data_dir_validation_path',type=str,required=True)



parser.add_argument('--experiment_dir', dest='experiment_dir',type=str,required=True)
parser.add_argument('--log_dir', dest='log_dir',type=str,required=True)
parser.add_argument('--experiment_id', dest='experiment_id',type=str,required=True)



parser.add_argument('--file_list_txt_train',dest='file_list_txt_train',type=str,required=True)
parser.add_argument('--file_list_txt_validation',dest='file_list_txt_validation',type=str,required=True)


parser.add_argument('--init_lr', dest='init_lr',type=float,required=True)




parser.add_argument('--batch_size', dest='batch_size',type=int,required=True)
parser.add_argument('--image_size', dest='image_size',type=int,required=True)

parser.add_argument('--epoch_num', dest='epoch_num',type=int,required=True)
parser.add_argument('--network', dest='network',type=str,required=True)

parser.add_argument('--label0_loss', dest='label0_loss',type=float,required=True)
parser.add_argument('--label1_loss', dest='label1_loss',type=float,required=True)
parser.add_argument('--center_loss_penalty_rate', dest='center_loss_penalty_rate',type=float,required=True)



parser.add_argument('--debug_mode', dest='debug_mode',type=int,required=True)
parser.add_argument('--cheat_mode', dest='cheat_mode',type=int,required=True)


parser.add_argument('--image_filters', dest='image_filters',type=int,required=True)
parser.add_argument('--train_resume_mode', dest='train_resume_mode',type=int,required=True)











def main(_):

    train_procedures(args_input=args)






args = parser.parse_args(input_args)
args.data_dir_train_path=args.data_dir_train_path.split(',')
args.data_dir_validation_path=args.data_dir_validation_path.split(',')
args.file_list_txt_train=args.file_list_txt_train.split(',')
args.file_list_txt_validation=args.file_list_txt_validation.split(',')
for ii in range(len(args.data_dir_train_path)):
    args.data_dir_train_path[ii] = os.path.join(exp_root_path,args.data_dir_train_path[ii])
for ii in range(len(args.data_dir_validation_path)):
    args.data_dir_validation_path[ii] = os.path.join(exp_root_path,args.data_dir_validation_path[ii])
args.log_dir = os.path.join(exp_root_path,args.log_dir)
if __name__ == '__main__':
    tf.app.run()
