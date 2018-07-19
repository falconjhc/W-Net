# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

GRAYSCALE_AVG = 127.5



from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt
import pylab

from utilities.utils import image_show
import numpy as np
import os
import random as rnd
import scipy.misc as misc

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


def get_chars_set_from_searching(path,level1_charlist,level2_charlist,
                                 level1_labellist,level2_labellist):

    chars = list()
    character_list=list()
    with open(path) as f:

        for line in f:


            line = u"%s" % line
            char_counter=0
            for char in line:



                current_char = line[char_counter]
                char_counter += 1


                if not current_char =='\n':
                    level1_found = current_char in level1_charlist
                    level2_found = current_char in level2_charlist


                    if level1_found==1:
                        idx = level1_charlist.index(current_char)
                        character_id = level1_labellist[idx]
                    elif level2_found==2:
                        idx = level2_charlist.index(current_char)
                        character_id = level2_labellist[idx]
                    else:
                        character_id = 0

                    character_list.append(int(character_id))
                    chars.append(current_char)
                else:
                    a=1

    return chars,character_list



def find_source_char_img(charset, fontpath,
                         img_width,img_filters,
                         batch_size):

    standard_font = ImageFont.truetype(fontpath, size=150)
    full_chars = np.zeros([len(charset),img_width,img_width,img_filters])
    ii=0
    for c in charset:
        curt_char = draw_single_char(ch=c,
                                     font=standard_font,
                                     canvas_size=img_width,filters=img_filters)
        curt_char = np.subtract(np.divide(curt_char,
                                          np.ones(curt_char.shape) * GRAYSCALE_AVG),
                                np.ones(curt_char.shape))


        full_chars[ii,:,:,:]=curt_char
        ii+=1


    if not batch_size == 1:
        iter_num = len(charset) / batch_size + 1
    else:
        iter_num = len(charset) / batch_size
    full_batch_num = iter_num * batch_size
    added_needed = full_batch_num - full_chars.shape[0]
    full_chars = np.concatenate([full_chars,full_chars[0:added_needed,:,:,:]])

    return full_chars

def draw_single_char(ch, font, canvas_size, x_offset=20, y_offset=20,filters=-1):
    img_read = Image.new("RGB", (256, 256), (255, 255, 255))
    draw = ImageDraw.Draw(img_read)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    img_read = np.array(img_read)

    img_matrix = np.asarray(img_read)[:, :, 0]
    zero_indices = np.where(img_matrix == 0)
    exceed = 'NONE'

    if np.min(img_matrix) == np.max(img_matrix) or (0 not in img_matrix):

        img_output = np.zeros(shape=[256, 256, 3])
        img_output = Image.fromarray(np.uint8(img_output))
    else:
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
        img_resize = img_cut.resize((150, 150), Image.ANTIALIAS)
        img_output = Image.new("RGB", (256, 256), (255, 255, 255))
        img_output.paste(img_resize, (52, 52))


    if not canvas_size == 256:
        img_output = img_output.resize((canvas_size,canvas_size), Image.ANTIALIAS)

    img = np.array(img_output)

    if filters == 1:
        img = img[:, :, 0]
        img = np.reshape(img, [img.shape[0], img.shape[1], 1])




    return img


def find_true_targets(data_dir,txt_path,selected_label1,selected_label0_list,
                      charset,font,img_width,filter_num,batch_size):
    standard_font = ImageFont.truetype(font, size=150)
    full_data_list = list()
    full_label0_list = list()
    prefix_list=list()
    for ii in range(len(data_dir)):

        file_handle = open(txt_path[ii], 'r')
        lines = file_handle.readlines()

        for line in lines:
            curt_line = line.split('@')

            label0 = int(curt_line[1])
            label1 = int(curt_line[2])

            if not label1 == selected_label1:
                continue

            curt_data = curt_line[3].split('\n')[0]
            if curt_data[0] == '/':
                curt_data = curt_data[1:]
            curt_data_path = os.path.join(data_dir[ii], curt_data)
            full_data_list.append(curt_data_path)
            full_label0_list.append(label0)
            prefix_list.append(txt_path[ii])


        file_handle.close()


    counter=0
    full_chars = np.zeros([len(charset), img_width, img_width, filter_num])
    for ii in selected_label0_list:

        if ii in full_label0_list:



            found_indices = [check for check, found in enumerate(full_label0_list) if found == ii]
            if len(found_indices)>1:
                index_selected = rnd.sample(found_indices,1)[0]
            else:
                index_selected = found_indices[0]
            curt_data = full_data_list[index_selected]
            char_img = Image.open(curt_data)

            # index = full_label0_list.index(ii)
            # handwritingmark =  'HandWriting' in prefix_list[index]
            # if handwritingmark:
            #     char_img_read = char_img.resize((150, 150), Image.ANTIALIAS)
            #     char_img = Image.new("RGB", (256, 256), (255, 255, 255))
            #     char_img.paste(char_img_read, (52, 52))
            #     if not img_width == 256:
            #         char_img = char_img.resize((img_width,img_width),Image.ANTIALIAS)

            char_img = np.array(char_img)
            if filter_num == 1:
                char_img = char_img[:, :, 0]
                char_img = np.reshape(char_img, [char_img.shape[0], char_img.shape[1], 1])
                char_img = np.subtract(np.divide(char_img,
                                                 np.ones(char_img.shape) * GRAYSCALE_AVG),
                                       np.ones(char_img.shape))



        else:
            char_img = draw_single_char(ch=charset[counter],
                                        font=standard_font,
                                        canvas_size=img_width,
                                        filters=filter_num)
            char_img = np.subtract(np.divide(char_img,
                                             np.ones(char_img.shape) * GRAYSCALE_AVG),
                                   np.ones(char_img.shape))

            #image_show(char_img)
        full_chars[counter,:,:,:] = char_img


        counter+=1

    if not batch_size == 1:
        iter_num = len(charset) / batch_size + 1
    else:
        iter_num = len(charset) / batch_size
    full_batch_num = iter_num * batch_size
    added_needed = full_batch_num - full_chars.shape[0]
    full_chars = np.concatenate([full_chars, full_chars[0:added_needed, :, :, :]])


    return full_chars


def find_transfer_targets(data_dir,txt_path,selected_label1,style_input_number,
                          img_width,filter_num,batch_size):
    data_list = list()
    for ii in range(len(data_dir)):

        file_handle = open(txt_path[ii], 'r')
        lines = file_handle.readlines()


        for line in lines:
            curt_line = line.split('@')

            label1 = int(curt_line[2])

            if label1 == selected_label1:

                curt_data = curt_line[3].split('\n')[0]
                if curt_data[0] == '/':
                    curt_data = curt_data[1:]
                curt_data_path = os.path.join(data_dir[ii], curt_data)
                data_list.append(curt_data_path)
        file_handle.close()

    indices = range(len(data_list))
    selected_indices = rnd.sample(indices,style_input_number)
    data_list = [data_list[i] for i in selected_indices]

    full_chars = np.zeros([style_input_number, img_width, img_width, filter_num])
    counter=0
    for ii in data_list:
        char_img = misc.imread(ii)
        if filter_num == 1:
            char_img = char_img[:, :, 0]
            char_img = np.reshape(char_img, [char_img.shape[0], char_img.shape[1], 1])
            char_img = np.subtract(np.divide(char_img,
                                             np.ones(char_img.shape) * GRAYSCALE_AVG),
                                   np.ones(char_img.shape))
        full_chars[counter, :, :, :] = char_img
        counter+=1

    iter_num = style_input_number / batch_size + 1
    full_batch_num = iter_num * batch_size
    added_needed = full_batch_num - full_chars.shape[0]

    if added_needed < full_chars.shape[0]:
        full_chars = np.concatenate([full_chars, full_chars[0:added_needed, :, :, :]])
    else:
        for ii in range(added_needed):
            full_char_length = full_chars.shape[0]
            selected = rnd.sample(range(full_char_length),1)
            full_chars = np.concatenate([full_chars, full_chars[selected, :, :, :]])

    return full_chars




def matrix_paper_generation(images, rows, columns):
    char_width=images.shape[1]

    chars_per_row = columns
    chars_per_column = rows

    output_paper = Image.new("RGB", (char_width * chars_per_row,
                                     char_width * chars_per_column),
                             (255, 255, 255))

    column_counter = 1
    row_counter = 1
    for ii in range(images.shape[0]):
        curt_img = np.squeeze(images[ii, :, :, :])
        curt_img = curt_img - np.min(curt_img)
        curt_img = curt_img / np.max(curt_img)
        curt_img = curt_img * 255
        curt_img = np.tile(np.reshape(curt_img,
                                      [curt_img.shape[0], curt_img.shape[1], 1]),
                           [1, 1, 3])
        curt_pasted = Image.fromarray(np.uint8(curt_img))
        output_paper.paste(curt_pasted, ((column_counter - 1) * char_width,
                                         (row_counter - 1) * char_width))

        column_counter += 1

        if column_counter > chars_per_row:
            column_counter = 1
            row_counter += 1

    return output_paper



def one_row_or_column_generation(images,option):
    img_num = images.shape[0]
    char_width = images.shape[1]
    if option=='ROW':
        output_paper = Image.new("RGB", (char_width * img_num,
                                         char_width),
                                 (255, 255, 255))
        for ii in range(images.shape[0]):
            curt_img = np.squeeze(images[ii, :, :, :])
            curt_img = curt_img - np.min(curt_img)
            curt_img = curt_img / np.max(curt_img)
            curt_img = curt_img * 255
            curt_img = np.tile(np.reshape(curt_img,
                                          [curt_img.shape[0], curt_img.shape[1], 1]),
                               [1, 1, 3])
            curt_pasted = Image.fromarray(np.uint8(curt_img))
            output_paper.paste(curt_pasted,(ii*char_width,0))
    elif option=='COLUMN':
        output_paper = Image.new("RGB", (char_width ,
                                         char_width * img_num),
                                 (255, 255, 255))
        for ii in range(images.shape[0]):
            curt_img = np.squeeze(images[ii, :, :, :])
            curt_img = curt_img - np.min(curt_img)
            curt_img = curt_img / np.max(curt_img)
            curt_img = curt_img * 255
            curt_img = np.tile(np.reshape(curt_img,
                                          [curt_img.shape[0], curt_img.shape[1], 1]),
                               [1, 1, 3])
            curt_pasted = Image.fromarray(np.uint8(curt_img))
            output_paper.paste(curt_pasted, (0,ii * char_width))

    return output_paper
