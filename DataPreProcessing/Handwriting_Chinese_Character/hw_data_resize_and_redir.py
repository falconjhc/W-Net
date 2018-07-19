# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import



import argparse
import os
import sys
import shutil
import time


import copy as cp
import numpy as np
from scipy import misc

import matplotlib.pyplot as plt
import pylab


from PIL import Image
eps = 1e-3


reload(sys)
sys.setdefaultencoding("utf-8")

input_args = ['--input_dir','/DataB/Harric/Chinese_Character_Generation/HandWritingData/SingleChars/CASIA-HWDB2.0',
              '--output_dir','/DataB/Harric/Chinese_Character_Generation/HandWritingData/SingleChars/CASIA-HWDB2.0_New',
              '--num_writer_each_group','5']


print_separater="##########################################################################"

parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--input_dir', dest='input_dir', required=True, type=str, help='path of the input dir')
parser.add_argument('--output_dir', dest='output_dir', required=True, type=str,help='path of the output dir')
parser.add_argument('--num_writer_each_group', dest='num_writer_each_group', type=int,required=True, help='number of writers in each sub dir')







target_img_size = 256

def image_show(img):
    img_out = cp.deepcopy(img)
    img_out = np.squeeze(img_out)
    img_shapes=img_out.shape
    if len(img_shapes)==2:
        curt_channel_img = img_out
        min_v = np.min(curt_channel_img)
        curt_channel_img = curt_channel_img - min_v
        max_v = np.max(curt_channel_img)
        curt_channel_img = curt_channel_img/ np.float32(max_v)
        img_out = curt_channel_img*255
    elif img_shapes[2] == 3:
        channel_num = img_shapes[2]
        for ii in range(channel_num):
            curt_channel_img = img[:,:,ii]
            min_v = np.min(curt_channel_img)
            curt_channel_img = curt_channel_img - min_v
            max_v = np.max(curt_channel_img)
            curt_channel_img = curt_channel_img / np.float32(max_v)
            img_out[:,:,ii] = curt_channel_img*255
    else:
        print("Channel Number is INCORRECT:%d" % img_shapes[2])



    plt.imshow(np.float32(img_out)/255)
    pylab.show()


def find_file_and_label_list_from_given_path(input_path, find_extension):
    data_list = []
    label0_list = []
    label1_list = []

    for root, dirs, files in os.walk(input_path):
        files.sort()
        for name in files:

            file_path = (os.path.join(root, name))
            file_extension = os.path.splitext(file_path)[1]
            file_extension = file_extension[1:]
            file_name = file_path.split('/')[len(file_path.split('/')) - 1]

            if file_extension == find_extension:
                data_list.append(os.path.join(root, name))

                label1 = file_name.split("_")[1]
                label1 = label1[0:label1.index('.')]
                label1 = int(label1)

                label0 = file_name.split("_")[0]
                label0 = int(label0)

                label0_list.append(label0)
                label1_list.append(label1)

    label0_vec = np.unique(label0_list)
    label1_vec = np.unique(label1_list)
    label0_vec.sort()
    label1_vec.sort()

    return data_list, label0_list, label1_list, label1_vec, label0_vec



# # def create_sub_dirs(writer_id_vec,each_group_writer_num, output_dir):
# #
# #     idx_list = list()
# #
# #     if os.path.exists(args.output_dir):
# #         shutil.rmtree(args.output_dir)
# #     os.makedirs(args.output_dir)
# #
# #     first_writer = writer_id_vec[0]
# #     first_group = first_writer % 5
# #     final_writer = writer_id_vec[len(writer_id_vec)-1]
# #     for ii in range(len(writer_id_vec)):
# #
# #
# #
# #         if ii % each_group_writer_num == 0:
# #             if ii == 0:
# #                 writer_start = 1
# #             else:
# #                 writer_start = writer_end + 1
# #             writer_end = writer_start + each_group_writer_num-1
# #             if writer_end > final_writer:
# #                 writer_end = final_writer
# #
# #             curt_sub_dir_name = ("writer%05d_To_%05d" % (writer_start,writer_end))
# #             curt_sub_dir_name_with_path = os.path.join(output_dir,curt_sub_dir_name)
# #
# #             os.makedirs(curt_sub_dir_name_with_path)
# #
# #             idx_list.append([writer_start,writer_end])
#
#     return idx_list









def resize_and_resave_to_new_dir(input_dir,output_dir,data_list,writer_list,num_writer_each_group):
    idx=0
    log_interval = len(data_list)/100
    full_start_time=time.time()
    for data_old_path in data_list:
        old_img = Image.open(data_old_path)
        data_new_path = data_old_path.replace(input_dir,output_dir)
        data_new_dir = str(os.path.split(os.path.realpath(data_new_path))[0])
        new_img = old_img.resize((target_img_size, target_img_size), Image.ANTIALIAS)

        writer = writer_list[idx]

        if writer >=0 and writer <=num_writer_each_group:
            writer_start = writer_list[0]
            writer_end = (writer_start / num_writer_each_group + 1) * num_writer_each_group
        elif writer % num_writer_each_group==0:
            writer_start = writer / num_writer_each_group * num_writer_each_group - num_writer_each_group + 1
            writer_end = writer_start + num_writer_each_group - 1
        else:
            writer_start = writer / num_writer_each_group * num_writer_each_group + 1
            writer_end = writer_start + num_writer_each_group - 1





        curt_sub_dir_name = ("writer%05d_To_%05d" % (writer_start, writer_end))
        curt_output_dir=os.path.join(output_dir,curt_sub_dir_name)

        if not os.path.exists(curt_output_dir):
            os.makedirs(curt_output_dir)
            print("SubDir created: %s" % curt_output_dir)

        file_name  = str(os.path.split(os.path.realpath(data_old_path))[1])
        curt_output_dir_with_img_file_name = os.path.join(curt_output_dir,file_name)
        new_img.save(curt_output_dir_with_img_file_name)
        # misc.imsave(curt_output_dir_with_img_file_name,new_img)

        if idx % log_interval==0:
            curt_time=time.time()
            from_start_time_elapsed = curt_time - full_start_time
            percentage_completed = float(idx) / float(len(data_list)) * 100
            percentage_to_be_fulfilled = 100 - percentage_completed
            hrs_estimated_remaining = (float(from_start_time_elapsed) / (
                    percentage_completed + eps)) * percentage_to_be_fulfilled / 3600
            print("Idx:%d/%d, CompletePctg:%.2f, TimeRemainingEstimated:%.2fhrs" % (idx,len(data_list),percentage_completed,hrs_estimated_remaining))

        idx+=1

def resize_and_resave_to_new_dir_do_not_create_new_subdirs(input_dir,output_dir,data_list):
    idx=0
    log_interval = len(data_list)/100
    full_start_time=time.time()
    for data_old_path in data_list:
        data_new_path = data_old_path.replace(input_dir,output_dir)
        data_new_dir_path = str(os.path.split(os.path.realpath(data_new_path))[0])
        if not os.path.exists(data_new_dir_path):
            os.makedirs(data_new_dir_path)

        old_img = Image.open(data_old_path)
        new_img = old_img.resize((target_img_size, target_img_size), Image.ANTIALIAS)
        new_img.save(data_new_path)

        if idx % log_interval==0:
            curt_time=time.time()
            from_start_time_elapsed = curt_time - full_start_time
            percentage_completed = float(idx) / float(len(data_list)) * 100
            percentage_to_be_fulfilled = 100 - percentage_completed
            hrs_estimated_remaining = (float(from_start_time_elapsed) / (
                    percentage_completed + eps)) * percentage_to_be_fulfilled / 3600
            print("Idx:%d/%d, CompletePctg:%.2f, TimeRemainingEstimated:%.2fhrs" % (idx,len(data_list),percentage_completed,hrs_estimated_remaining))

        idx+=1









args = parser.parse_args(input_args)
if __name__=="__main__":
    print(print_separater)
    print("ReadFrom:%s" % args.input_dir)
    print("SaveTo:%s" % args.output_dir)
    data_list, \
    label0_list, \
    label1_list, \
    label1_vec, \
    label0_vec = find_file_and_label_list_from_given_path(input_path=args.input_dir,
                                                          find_extension='png')
    print("Total sample number:%d, total writer number:%d" % (len(data_list), len(label1_vec)))
    print(print_separater)

    # idx_list = create_sub_dirs(writer_id_vec=label1_vec,
    #                            each_group_writer_num=args.num_writer_each_group,
    #                            output_dir=args.output_dir)
    # print("Total dir number:%d" % len(idx_list))
    # print(print_separater)
    # #raw_input("Press enter to continue")
    # print(print_separater)



    data_list_new = resize_and_resave_to_new_dir(input_dir=args.input_dir,
                                                 output_dir=args.output_dir,
                                                 data_list=data_list,
                                                 writer_list=label1_list,
                                                 num_writer_each_group=args.num_writer_each_group)

    # resize_and_resave_to_new_dir_do_not_create_new_subdirs(input_dir=args.input_dir,
    #                                                        output_dir=args.output_dir,
    #                                                        data_list=data_list)


    print(print_separater)
    print("All completed")
    print(print_separater)

    a=1



    a=1
    print(a)