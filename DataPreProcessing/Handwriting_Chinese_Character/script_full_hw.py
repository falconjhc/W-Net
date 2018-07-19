# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import sys
import numpy as np
import os
import shutil

import cPickle as pickle
import random


import matplotlib.pyplot as plt
import pylab
import matplotlib.image as mpimg


from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import collections

reload(sys)
sys.setdefaultencoding("utf-8")


input_args = [
              '--dst_font_dir','/data/Harric/CASIA_256/HandWritingData/SingleChars/CASIA-HWDB2.1/',
              '--src_font','../standard_font.ttf',
              '--pair_jpg_256','/data/Harric/CASIA_256/HandWritingData/PairChars/CASIA-HWDB2.1/',
              '--pair_jpg_128','/data/Harric/CASIA_128/HandWritingData/PairChars/CASIA-HWDB2.1/',
              '--pair_jpg_64','/data/Harric/CASIA_64/HandWritingData/PairChars/CASIA-HWDB2.1/']

print_separater="##########################################################################"



parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--src_font', dest='src_font', required=True, help='path of the source font')
parser.add_argument('--dst_font_dir', dest='dst_font_dir', required=True, help='path of the target font dir')
parser.add_argument('--char_size', dest='char_size', type=int, default=150, help='character size')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', dest='x_offset', type=int, default=20, help='x offset')
parser.add_argument('--y_offset', dest='y_offset', type=int, default=20, help='y_offset')
parser.add_argument('--pair_jpg_256', dest='pair_jpg_256', required=True,help='directory to save paired examples')
parser.add_argument('--pair_jpg_128', dest='pair_jpg_128', required=True,help='directory to save paired examples')
parser.add_argument('--pair_jpg_64', dest='pair_jpg_64', required=True,help='directory to save paired examples')



args = parser.parse_args(input_args)


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)

    img_matrix = np.asarray(img)[:, :, 0]
    zero_indices = np.where(img_matrix == 0)
    exceed = 'NONE'


    up_p = np.min(zero_indices[0])
    down_p = np.max(zero_indices[0])
    left_p = np.min(zero_indices[1])
    right_p = np.max(zero_indices[1])

    up_down = down_p - up_p
    right_left = right_p - left_p
    if up_down > right_left:
        character_size = up_down
        if not character_size % 2 == 0:
            character_size = character_size + 1
            down_p = down_p + 1
        right_left_avg = (right_p + left_p) / 2
        right_p = right_left_avg + int(character_size / 2)
        left_p = right_left_avg - int(character_size / 2)
        if left_p < 0:
            exceed = 'LEFT'
            exceed_pixels = np.abs(left_p)
            left_p = 0
        if right_p > 255:
            exceed = 'RIGHT'
            exceed_pixels = right_p - 255
            right_p = 255

    else:
        character_size = right_left
        if not character_size % 2 == 0:
            character_size = character_size + 1
            right_p = right_p + 1

        up_down_avg = (up_p + down_p) / 2
        down_p = up_down_avg + int(character_size / 2)
        up_p = up_down_avg - int(character_size / 2)
        if up_p < 0:
            exceed = 'UP'
            exceed_pixels = np.abs(up_p)
            up_p = 0
        if down_p > 255:
            exceed = 'DOWN'
            exceed_pixels = down_p - 255
            down_p = 255

    img_matrix_cut = img_matrix[up_p:down_p, left_p:right_p]
    if not exceed == 'NONE':
        if exceed == 'LEFT':
            added_pixels = np.ones([img_matrix_cut.shape[0], exceed_pixels]) * 255
            img_matrix_cut = np.concatenate([added_pixels, img_matrix_cut], axis=1)
        elif exceed == 'RIGHT':
            added_pixels = np.ones([img_matrix_cut.shape[0], exceed_pixels]) * 255
            img_matrix_cut = np.concatenate([img_matrix_cut, added_pixels], axis=1)
        elif exceed == 'UP':
            added_pixels = np.ones([exceed_pixels, img_matrix_cut.shape[1]]) * 255
            img_matrix_cut = np.concatenate([added_pixels, img_matrix_cut], axis=0)
        elif exceed == 'DOWN':
            added_pixels = np.ones([exceed_pixels, img_matrix_cut.shape[1]]) * 255
            img_matrix_cut = np.concatenate([img_matrix_cut, added_pixels], axis=0)


    img_matrix_cut = np.tile(np.reshape(img_matrix_cut,
                                        [img_matrix_cut.shape[0], img_matrix_cut.shape[1], 1]),
                             [1, 1, 3])
    img_cut = Image.fromarray(np.uint8(img_matrix_cut))
    img_resize = img_cut.resize((150,150),Image.ANTIALIAS)
    img_output_256 = Image.new("RGB", (256, 256), (255, 255, 255))
    img_output_256.paste(img_resize,(52,52))

    img_output_128 = img_output_256.resize((128,128), Image.ANTIALIAS)
    img_output_64 = img_output_256.resize((64, 64), Image.ANTIALIAS)






    return img_output_256,img_output_128,img_output_64



def font2img(src=-1, dst=-1,
             charDrawing=-1,
             char_size=-1, canvas_size=-1,
             x_offset=-1, y_offset=-1,
             sample_dir_256=-1,
             sample_dir_128=-1,
             sample_dir_64=-1,
             writer_id=-1,character_id='',
             index=-1):


    src_img_256,src_img_128,src_img_64  = draw_single_char(charDrawing, src, canvas_size, x_offset, y_offset)
    if not dst==-1: # char drawing is corresponded with one hw
        dst_img_256 = Image.open(dst)
        dst_img_128 = dst_img_256.resize((128,128),Image.ANTIALIAS)
        dst_img_64 = dst_img_256.resize((64, 64), Image.ANTIALIAS)
    else: # char drawing is not corresponded with one hw --> 使用src来充数！
        dst_img_256 = src_img_256
        dst_img_128 = src_img_128
        dst_img_64 = src_img_64




    if not os.path.exists(sample_dir_256):
       os.makedirs(sample_dir_256)
    if not os.path.exists(sample_dir_128):
       os.makedirs(sample_dir_128)
    if not os.path.exists(sample_dir_64):
       os.makedirs(sample_dir_64)


    current_pair_dir = ("Font_Pair_No_%d" % writer_id)

    current_pair_dir_256 = os.path.join(sample_dir_256,current_pair_dir)
    if not os.path.exists(current_pair_dir_256):
        os.makedirs(current_pair_dir_256)
    current_pair_dir_128 = os.path.join(sample_dir_128, current_pair_dir)
    if not os.path.exists(current_pair_dir_128):
        os.makedirs(current_pair_dir_128)
    current_pair_dir_64 = os.path.join(sample_dir_64, current_pair_dir)
    if not os.path.exists(current_pair_dir_64):
        os.makedirs(current_pair_dir_64)

    e_256 = Image.new("RGB", (512,256), (255, 255, 255))
    e_256.paste(dst_img_256, (0,0))
    e_256.paste(src_img_256, (256, 0))
    e_256.save(os.path.join(current_pair_dir_256, "%09d_%s_%05d.jpg" % (index, character_id,writer_id)))

    e_128 = Image.new("RGB", (256, 128), (255, 255, 255))
    e_128.paste(dst_img_128, (0, 0))
    e_128.paste(src_img_128, (128, 0))
    e_128.save(os.path.join(current_pair_dir_128, "%09d_%s_%05d.jpg" % (index, character_id, writer_id)))

    e_64 = Image.new("RGB", (128, 64), (255, 255, 255))
    e_64.paste(dst_img_64, (0, 0))
    e_64.paste(src_img_64, (64, 0))
    e_64.save(os.path.join(current_pair_dir_64, "%09d_%s_%05d.jpg" % (index, character_id, writer_id)))



def get_chars_set_from_level1_2(path,level):
    """
    Expect a text file that each line is a char
    """
    chars = list()
    character_id_1_counter=0
    character_id_2_counter=0
    character_list=list()
    with open(path) as f:
        for line in f:

            line = u"%s" % line
            char_counter=0
            for char in line:

                current_char = line[char_counter]
                chars.append(current_char)




                if level==1:
                    character_id_1 = str(character_id_1_counter + 16 + 160)
                    character_id_2 = str(character_id_2_counter + 1 + 160)
                    character_id = character_id_1 + character_id_2

                    character_id_2_counter += 1
                    if character_id_2_counter == 94:
                        character_id_2_counter = 0
                        character_id_1_counter += 1
                elif level==2:
                    character_id_1 = str(character_id_1_counter + 56 + 160)
                    character_id_2 = str(character_id_2_counter + 1 + 160)
                    character_id = character_id_1 + character_id_2

                    character_id_2_counter += 1
                    if character_id_2_counter == 94:
                        character_id_2_counter = 0
                        character_id_1_counter += 1

                character_list.append(character_id)
                char_counter+=1
    return chars,character_list









def Image_From_Png_To_Jpg(src_font,
                          current_dst_dir,
                          current_pair_dir_256,current_pair_dir_128,current_pair_dir_64,
                          current_dir):


    dst_file_list_level_1 = []
    dst_file_list_level_2 = []
    writer_id_list = []
    level_1_character_list = []
    lever_1_writer_list = []
    level_2_character_list = []
    lever_2_writer_list = []
    for root, dirs, files in os.walk(current_dst_dir):
        files.sort()
        for name in files:

            if not ((name.find("DS") == -1) and (name.find("Th") == -1)):
                continue

            file_path = (os.path.join(root, name))
            file_extension = os.path.splitext(file_path)[1]

            split_1 = file_path.split('_', file_path.count('_'))

            character_id = split_1[len(split_1) - 2][
                           split_1[len(split_1) - 2].rfind('/') + 1:split_1[len(split_1) - 2].rfind('/') + 7]
            character_id_1 = int(character_id[0:3])
            character_id_2 = int(character_id[3:])
            writer_id = int(split_1[len(split_1) - 1][0:split_1[len(split_1) - 1].rfind('.')])

            if not writer_id in writer_id_list:
                writer_id_list.append(writer_id)

            valid_extension = (file_extension == '.png')
            valid_character = (character_id_1 >= 176 and character_id_1 <= 247) \
                              and \
                              (character_id_2 >= 161 and character_id_2 <= 254)



            if valid_extension and valid_character:


                if character_id_1 - 160 <= 55 and character_id_1 - 160 >= 16:
                    dst_file_list_level_1.append(os.path.join(root, name))
                    level_1_character_list.append((character_id_1 - 160 - 16) * 94 + (character_id_2 - 160 - 1))
                    lever_1_writer_list.append(writer_id)

                elif character_id_1 - 160 <= 87:
                    dst_file_list_level_2.append(os.path.join(root, name))
                    level_2_character_list.append((character_id_1 - 160 - 56) * 94 + (character_id_2 - 160 - 1))
                    lever_2_writer_list.append(writer_id)
    print("Paired File Mapping Finding Completed for Dir: %s." % current_dir)



    # generate pair files for tarin (GB2312 Level 1)
    file_counter = 0
    for dst_file in dst_file_list_level_1:
        split_1 = dst_file.split('_', dst_file.count('_'))

        character_id = split_1[len(split_1) - 2][
                       split_1[len(split_1) - 2].rfind('/') + 1:split_1[len(split_1) - 2].rfind('/') + 7]
        character_id_1 = int(character_id[0:3]) - 160 - 16
        character_id_2 = int(character_id[3:]) - 160 - 1
        writer_id = int(split_1[len(split_1) - 1][0:split_1[len(split_1) - 1].rfind('.')])

        index = character_id_1 * 94 + character_id_2
        character_id = str(character_id_1+160+16)+str(character_id_2+160+1)

        file_name = dst_file.split('/')[len(dst_file.split('/')) - 1]
        c_counter = int(file_name.split('_')[0])


        font2img(src=src_font, dst=dst_file,
                 charDrawing=charset_level1[index],
                 char_size=args.char_size, canvas_size=args.canvas_size,
                 x_offset=args.x_offset, y_offset=args.y_offset,
                 sample_dir_256=os.path.join(current_pair_dir_256, 'GB2312_L1'),
                 sample_dir_128 = os.path.join(current_pair_dir_128, 'GB2312_L1'),
                 sample_dir_64 = os.path.join(current_pair_dir_64, 'GB2312_L1'),
                 writer_id=writer_id, index=c_counter,
                 character_id=character_id)
        file_counter += 1
    print("GB2312 LevelNo.1 Paired Image Written for Dir: %s."  % current_dir)

    # generate pair files for validation and real essay
    writer_counter = 0
    hw_level2_counter=0
    for writer_id_traveller in writer_id_list:

        # current_val_dir = os.path.join(current_tmp_pair_dir, 'GB2312_L2')
        # # if not os.path.exists(current_val_dir):
        # #     os.mkdir(current_val_dir)
        #
        # current_pair_dir = ("Font_Pair_No_%d" % writer_id_traveller)
        # current_pair_dir = os.path.join(current_val_dir, current_pair_dir)
        # if not os.path.exists(current_pair_dir):
        #     os.mkdir(current_pair_dir)

        # generate pair files for validation (GB2312 Level 2)
        char_counter = 0
        file_counter = 0
        for c in charset_level2:

            found_writer_ids = []
            file_existed = False
            if char_counter in level_2_character_list:

                found_indices = [check for check, found in enumerate(level_2_character_list) if found == char_counter]
                for found_index in found_indices:
                    found_writer_ids.append(lever_2_writer_list[found_index])
                if writer_id_traveller in found_writer_ids:
                    file_existed = True
                    selected_index = [check for check, idx in enumerate(found_writer_ids) if idx == writer_id_traveller]
                else:
                    file_existed = False
                    selected_index = -1

            if file_existed == True:
                hw_level2_counter=hw_level2_counter+len(selected_index)

                for ii in selected_index:

                    selected_index_on_all_the_existed_files = found_indices[ii]
                    selected_existed_file_name_with_path = dst_file_list_level_2[selected_index_on_all_the_existed_files]

                    font2img(src=src_font, dst=selected_existed_file_name_with_path,
                             charDrawing=c,
                             char_size=args.char_size, canvas_size=args.canvas_size,
                             x_offset=args.x_offset, y_offset=args.y_offset,
                             sample_dir_256=os.path.join(current_pair_dir_256, 'GB2312_L2'),
                             sample_dir_128=os.path.join(current_pair_dir_128, 'GB2312_L2'),
                             sample_dir_64=os.path.join(current_pair_dir_64, 'GB2312_L2'),
                             writer_id=writer_id_traveller, index=file_counter,
                             character_id=character_label_level2[char_counter])
                    file_counter+=1
            else:
                font2img(src=src_font,
                         charDrawing=c,
                         char_size=args.char_size, canvas_size=args.canvas_size,
                         x_offset=args.x_offset, y_offset=args.y_offset,
                         sample_dir_256=os.path.join(current_pair_dir_256, 'GB2312_L2'),
                         sample_dir_128=os.path.join(current_pair_dir_128, 'GB2312_L2'),
                         sample_dir_64=os.path.join(current_pair_dir_64, 'GB2312_L2'),
                         writer_id=writer_id_traveller, index=file_counter,
                         character_id=character_label_level2[char_counter])
                file_counter += 1

            char_counter += 1

        writer_counter += 1
    print("GB2312 LevelNo.2 Paired Image Written for Dir: %s." % current_dir)

    return len(dst_file_list_level_1),hw_level2_counter




def list_all_dirs_with_chars(path):
    output_dir_list = list()
    for root, dirs, files in os.walk(path):
        if dirs == []:
            output_dir_list.append(root)
    return output_dir_list






if __name__ == "__main__":

    if os.path.exists(args.pair_jpg_256):
        shutil.rmtree(args.pair_jpg_256)
    if os.path.exists(args.pair_jpg_128):
        shutil.rmtree(args.pair_jpg_128)
    if os.path.exists(args.pair_jpg_64):
        shutil.rmtree(args.pair_jpg_64)


    os.makedirs(args.pair_jpg_256)
    os.makedirs(args.pair_jpg_128)
    os.makedirs(args.pair_jpg_64)

    src_font = ImageFont.truetype(args.src_font, size=args.char_size)



    ## get specific char sets
    charset_level1, character_label_level1 = get_chars_set_from_level1_2(path='../charset/GB2312_Level_1.txt', level=1)
    charset_level2, character_label_level2 = get_chars_set_from_level1_2(path='../charset/GB2312_Level_2.txt', level=2)







    # tgt_dir_list = os.listdir(args.dst_font_dir)
    tgt_dir_list = list_all_dirs_with_chars(args.dst_font_dir)
    tgt_dir_list.sort()

    level1_full=0
    level2_full=0
    level_full=0
    for current_dst_dir in tgt_dir_list:



        print(print_separater)
        group_name = current_dst_dir.split('/')
        group_name=group_name[len(group_name)-1]
        current_dst_dir_with_root=current_dst_dir



        level1_len,level2_len = \
            Image_From_Png_To_Jpg(src_font=src_font,
                                  current_dst_dir=current_dst_dir_with_root,
                                  current_pair_dir_256=args.pair_jpg_256,
                                  current_pair_dir_128=args.pair_jpg_128,
                                  current_pair_dir_64=args.pair_jpg_64,

                                  current_dir=current_dst_dir)

        level1_full = level1_full + level1_len
        level2_full = level2_full + level2_len
        level_full = level_full + level1_len + level2_len
        print("Level1:%d/%d, Level2:%d/%d, Full:%d/%d" % (level1_len,
                                                          level1_full,
                                                          level2_len,
                                                          level2_full,
                                                          level1_len+level2_len,
                                                          level_full))





