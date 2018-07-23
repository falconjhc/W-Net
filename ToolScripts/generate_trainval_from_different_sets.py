# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import


import argparse
import os
import numpy as np
import random as rd


input_args = [
              # '--data_type','SINGLE', # 'SINGLE' or STANDARD'
              # '--data_dir_path','/DataA/Harric/MSMC_Exp/CASIA_64/HandWritingData/CASIA-HWDB2.1/',
              # '--file_write_path', '../FileList/HandWritingData/Char_0_3754_Writer_1001_1032_Cursive.txt',

              '--data_type','SINGLE', # 'SINGLE' or STANDARD'
              '--data_dir_path','/DataA/Harric/MSMC_Exp/CASIA_64/PrintedData/',
              '--file_write_path', '../FileList/PrintedData/Char_0_3755_Writer_Selected32_Printed_Fonts_GB2312L1L2.txt',



              # '--data_type','STANDARD', # 'SINGLE' OR 'PAIR or STANDARD'
              # '--data_dir_path','/DataA/Harric/WNet_Exp/CASIA_64/StandardChars/GB2312_L1/',
              # '--file_write_path', '../FileList/StandardChars/char_0_3754',

              '--label0','0:1:3754',
              '--label1','14,16,17,18,26,27,29,30,32,35,37,38,41,42,46,47,49,52,55,57,59,63,65,67,68,70,71,72,74,76,78,79']



parser = argparse.ArgumentParser(description='Generate Train / Validation Groups')
parser.add_argument('--data_type', dest='data_type',type=str,required=True)
parser.add_argument('--data_dir_path', dest='data_dir_path',type=str,required=True)
parser.add_argument('--label0', dest='label0',type=str,required=True)
parser.add_argument('--label1', dest='label1',type=str,required=True)


parser.add_argument('--file_write_path', dest='file_write_path',type=str,required=True)





# find all the data with label0 / label1 with the give path
# searching for all the dirs and subdirs
def find_file_and_label_list_from_given_path(input_path, data_type,selected_label0,selected_label1):


    find_extension = 'png'

    if not selected_label0 == 'ALL':
        selected_label0 = selected_label0.split(':')
        selected_label0 = range(int(selected_label0[0]), int(selected_label0[2]) + 1, int(selected_label0[1]))
    if not selected_label1 == 'ALL':
        if ':' in selected_label1:
            selected_label1 = selected_label1.split(':')
            selected_label1 = range(int(selected_label1[0]), int(selected_label1[2]) + 1, int(selected_label1[1]))
        elif ',' in selected_label1:
            selected_label1_output = list()
            selected_label1 = selected_label1.split(',')
            for ii in selected_label1:
                selected_label1_output.append(int(ii))
            selected_label1 = selected_label1_output



    data_list = []
    label0_list = []
    label1_list = []

    for root, dirs, files in os.walk(input_path):

        if files:



            files.sort()
            for name in files:

                file_path = (os.path.join(root, name))
                file_extension = os.path.splitext(file_path)[1]
                file_extension = file_extension[1:]
                file_name = file_path.split('/')[len(file_path.split('/')) - 1]

                if file_extension == find_extension:

                    label1 = file_name.split("_")[2]
                    label1 = label1[0:label1.index('.')]
                    label1 = int(label1)

                    if label1 == 1031 or label1 == 1032:
                        a=1

                    label0 = file_name.split("_")[1]
                    character_id_1 = int(label0[0:3])
                    character_id_2 = int(label0[3:6])
                    character_id = (character_id_1 - 160 - 16) * 94 + (character_id_2 - 160 - 1)
                    label0 = int(label0)

                    record_char0=False
                    record_char1=False
                    if selected_label0 == 'ALL':
                        record_char0=True
                    elif character_id in selected_label0:
                        record_char0=True

                    if selected_label1 == 'ALL':
                        record_char1=True
                    elif label1 in selected_label1:
                        record_char1=True
                    record_char = record_char0 and record_char1





                    if record_char:
                        label0_list.append(label0)
                        label1_list.append(label1)
                        data_list.append(os.path.join(root, name))



    label0_vec = np.unique(label0_list)
    label1_vec = np.unique(label1_list)
    label0_vec.sort()
    label1_vec.sort()

    return data_list, label0_list, label1_list, label1_vec, label0_vec




def find_file_and_label_list_from_given_path_for_standard_chars(input_path,selected_label0):
    find_extension = 'jpg'

    if not selected_label0 == 'ALL':
        selected_label0 = selected_label0.split(':')
        selected_label0 = range(int(selected_label0[0]), int(selected_label0[2]) + 1, int(selected_label0[1]))



    data_list = []
    label0_list = []
    label1_list = []

    for root, dirs, files in os.walk(input_path):

        if files:
            files.sort()
            for name in files:

                file_path = (os.path.join(root, name))
                file_extension = os.path.splitext(file_path)[1]
                file_extension = file_extension[1:]
                file_name = file_path.split('/')[len(file_path.split('/')) - 1]

                if file_extension == find_extension:


                    label0 = file_name.split("_")[1]
                    label0 = label0[0:label0.index('.')]
                    character_id_1 = int(label0[0:3])
                    character_id_2 = int(label0[3:6])
                    character_id = (character_id_1 - 160 - 16) * 94 + (character_id_2 - 160 - 1)
                    label0 = int(label0)

                    record_char0=False
                    if selected_label0 == 'ALL':
                        record_char0=True
                    elif character_id in selected_label0:
                        record_char0=True

                    if record_char0:
                        label0_list.append(label0)
                        label1_list.append(-1)
                        data_list.append(os.path.join(root, name))



    label0_vec = np.unique(label0_list)
    label0_vec.sort()

    label1_vec = np.unique(label1_list)
    label1_vec.sort()



    return data_list, label0_list, label0_vec, label1_list,label1_vec




def delete_prefix_for_file_path(data_list,prefix):
    prefix_length=len(prefix)
    new_list=list()
    for path in data_list:
        new_path = path[prefix_length:]
        new_list.append(new_path)
    return new_list


def write_to_file(path,data_list,label0_list,label1_list,mark):
    # if os.path.exists(path):
    #     print("File exists: %s, failed to write" % path)
    #     return

    file_handle = open(path,'w')
    full_line_num = len(data_list)
    for ii in range(full_line_num):
        if mark:
            write_info = str(1) +'@'+ str(label0_list[ii]) + '@' + str(label1_list[ii]) + '@' + data_list[ii]
        else:
            write_info = str(-1) + '@'+ str(label0_list[ii]) + '@' + str(label1_list[ii]) + '@' + data_list[ii]
        file_handle.write(write_info)
        file_handle.write('\n')
    file_handle.close()
    print("Write to File: %s" % path)




def check_selected_writer(output_writer,selected_writer):
    selected_writer = selected_writer.split(':')
    selected_writer = range(int(selected_writer[0]), int(selected_writer[2]) + 1, int(selected_writer[1]))

    not_exist_writer=list()
    for w in selected_writer:
        if w not in output_writer:
            not_exist_writer.append(w)


    return not_exist_writer

def check_selected_character(output_character,selected_character):
    selected_character = selected_character.split(':')
    selected_character = range(int(selected_character[0]), int(selected_character[2]) + 1, int(selected_character[1]))

    character_idx=list()
    for c in output_character:
        c_str = str(c)
        character_1 = int(c_str[0:3])
        character_2 = int(c_str[3:6])
        character_id = (character_1 - 160 - 16) * 94 + (character_2 - 160 - 1)
        character_idx.append(character_id)

    not_exist_character=list()
    not_exist_character_idx = list()
    for idx in selected_character:
        if idx not in character_idx:
            character_id1 = idx / 94 + 160 + 16
            character_id2 = idx % 94
            character_id2="%03d" % character_id2
            character_id = str(character_id1) + character_id2
            not_exist_character.append(character_id)
            not_exist_character_idx.append(idx)


    return not_exist_character,not_exist_character_idx






args = parser.parse_args(input_args)
def main():
    if not args.data_type =='STANDARD':
        data_list, \
        label0_list, \
        label1_list, \
        label1_vec, \
        label0_vec = find_file_and_label_list_from_given_path(input_path=args.data_dir_path,
                                                              data_type=args.data_type,
                                                              selected_label0=args.label0,
                                                              selected_label1=args.label1)
    else:
        data_list, \
        label0_list, \
        label0_vec,\
        label1_list,\
        label1_vec =\
            find_file_and_label_list_from_given_path_for_standard_chars(input_path=args.data_dir_path,
                                                                        selected_label0=args.label0)
    print("Writer:%d, Character:%d, DataNum:%d" % (len(label1_vec), len(label0_vec),len(data_list)))


    data_list = delete_prefix_for_file_path(data_list=data_list,
                                            prefix=args.data_dir_path)
    write_to_file(path=args.file_write_path,
                  data_list=data_list,
                  label0_list=label0_list,
                  label1_list=label1_list,
                  mark=True)



if __name__ == "__main__":
    main()



print("Accomplished All")