# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import cv2


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

def get_prototype_on_targeted_content_input_txt(targeted_content_input_txt,
                                                level1_charlist,level2_charlist,
                                                level1_labellist,level2_labellist,
                                                content_file_list_txt,content_file_data_dir,
                                                img_width,img_filters):
    # get label0 for the targeted content input txt
    targeted_chars_list = list()
    targeted_character_label0_list = list()

    with open(targeted_content_input_txt) as f:
        for line in f:

            line = u"%s" % line
            char_counter = 0
            for char in line:

                current_char = line[char_counter]
                char_counter += 1

                if not current_char == '\n':
                    level1_found = current_char in level1_charlist
                    level2_found = current_char in level2_charlist

                    if level1_found == 1:
                        idx = level1_charlist.index(current_char)
                        character_id = level1_labellist[idx]
                    elif level2_found == 2:
                        idx = level2_charlist.index(current_char)
                        character_id = level2_labellist[idx]
                    else:
                        print("Fails! Didnt find %s" % unicode(char))
                        character_id = 0
                        return False

                    targeted_character_label0_list.append(str(character_id))
                    targeted_chars_list.append(current_char)
    actual_char_list=line


    # read all content data
    content_label0_list = list()
    content_label1_list = list()
    content_data_list = list()

    for ii in range(len(content_file_list_txt)):

        file_handle = open(content_file_list_txt[ii], 'r')
        lines = file_handle.readlines()

        for line in lines:
            curt_line = line.split('@')
            content_label1_list.append(curt_line[2])
            content_label0_list.append(curt_line[1])
            curt_data = curt_line[3].split('\n')[0]
            if curt_data[0] == '/':
                curt_data = curt_data[1:]
            curt_data_path = os.path.join(content_file_data_dir[ii], curt_data)
            content_data_list.append(curt_data_path)
        file_handle.close()

    # find corresponding content data
    content_label1_vec = np.unique(content_label1_list)
    content_label1_vec.sort()
    corresponding_content_prototype = np.zeros(shape=[len(targeted_character_label0_list),img_width,img_width,img_filters*len(content_label1_vec)],
                                               dtype=np.float32)
    label1_counter=0
    for content_label1 in content_label1_vec:
        current_label1_indices = [ii for ii in range(len(content_label1_list)) if content_label1_list[ii]==content_label1]
        current_label0_on_current_label1=list()
        current_data_on_current_label1=list()
        for ii in current_label1_indices:
            current_label0_on_current_label1.append(content_label0_list[ii])
            current_data_on_current_label1.append(content_data_list[ii])
        target_counter=0
        for ii in targeted_character_label0_list:
            if ii not in current_label0_on_current_label1:
                print("Fails! Didnt find %s" % unicode(actual_char_list[target_counter]))
                return False
            else:
                index_found = current_label0_on_current_label1.index(ii)
                char_img = misc.imread(current_data_on_current_label1[index_found])
                # print("%d %d" % (label1_counter,target_counter))
                if char_img.ndim==3:
                    char_img = np.expand_dims(char_img[:,:,0],axis=2)
                elif char_img.ndim==2:
                    char_img = np.expand_dims(char_img, axis=2)
                char_img = char_img / GRAYSCALE_AVG - 1
                corresponding_content_prototype[target_counter,:,:,label1_counter*img_filters:(label1_counter+1)*img_filters]\
                    =char_img
            target_counter+=1

        label1_counter+=1


    return corresponding_content_prototype, content_label1_vec


def get_style_references(img_path, style_input_number):
    def extract_peek(array_vals):
        minimun_val = 10
        minimun_range = 60
        start_i = None
        end_i = None
        peek_ranges = []
        for i, val in enumerate(array_vals):
            if val > minimun_val and start_i is None:
                start_i = i
            elif val > minimun_val and start_i is not None:
                pass
            elif val < minimun_val and start_i is not None:
                if i - start_i >= minimun_range:
                    end_i = i
                    # print(end_i - start_i)
                    peek_ranges.append((start_i, end_i))
                    start_i = None
                    end_i = None
            elif val < minimun_val and start_i is None:
                pass
            else:
                raise ValueError("cannot parse this case...")
        return peek_ranges

    def cutImage(img, peek_ranges):
        full_counter=0
        style_counter=0
        for i, peek_range in enumerate(peek_ranges):
            for vertical_range in vertical_peek_ranges2d[i]:
                x = vertical_range[0]
                y = peek_range[0]

                img_matrix = img[y:peek_range[1], x:vertical_range[1]]

                avg = np.mean(cv2.resize(img_matrix, (150, 150)))
                if avg < 250 and 0 in img_matrix:
                    print("FullCounter%d StyleCounter%d Avg:%f" % (full_counter, style_counter,avg))

                    img_matrix = np.floor(np.mean(img_matrix,axis=2))

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
                                                        [img_matrix_cut.shape[0], img_matrix_cut.shape[1], 1]), [1, 1, 3])
                    img_cut = Image.fromarray(np.uint8(img_matrix_cut))
                    img_resize = img_cut.resize((150, 150), Image.ANTIALIAS)
                    img_output_256 = Image.new("RGB", (256, 256), (255, 255, 255))
                    img_output_256.paste(img_resize, (52, 52))
                    img_output_64 = img_output_256.resize((64, 64), Image.ANTIALIAS)
                    img_output = np.array(img_output_64)
                    if img_output.ndim == 3:
                        img_output = np.expand_dims(img_output[:, :, 0], axis=2)
                    elif img_output.ndim == 2:
                        img_output = np.expand_dims(img_output, axis=2)
                    img_output = img_output / GRAYSCALE_AVG - 1

                    if style_counter == 0:
                        style_reference = img_output
                    else:
                        style_reference = np.concatenate([style_reference, img_output], axis=2)
                    style_counter+=1
                full_counter += 1

        return style_reference


    img = cv2.imread(img_path)
    img_new = img
    img_new[np.where(img<150)]=0
    img_new[np.where(img>=150)]=255
    img = img_new

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                               cv2.THRESH_BINARY_INV, 11, 2)
    horizontal_sum = np.sum(adaptive_threshold, axis=1)
    peek_ranges = extract_peek(horizontal_sum)
    line_seg_adaptive_threshold = np.copy(adaptive_threshold)

    for i, peek_range in enumerate(peek_ranges):
        x = 0
        y = peek_range[0]
        w = line_seg_adaptive_threshold.shape[1]
        h = peek_range[1] - y
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(line_seg_adaptive_threshold, pt1, pt2, 255)

    vertical_peek_ranges2d = []
    for peek_range in peek_ranges:
        start_y = peek_range[0]
        end_y = peek_range[1]
        line_img = adaptive_threshold[start_y:end_y, :]
        vertical_sum = np.sum(line_img, axis=0)
        vertical_peek_ranges = extract_peek(vertical_sum)
        vertical_peek_ranges2d.append(vertical_peek_ranges)

    style_reference = cutImage(img, peek_ranges)



    if (not style_input_number == 0) and style_input_number<style_reference.shape[2]:
        rnd_indices=rnd.sample(range(style_reference.shape[2]),style_input_number)
        rnd_counter=0
        for ii in rnd_indices:
            current_style_ref=np.expand_dims(style_reference[:,:,ii],axis=2)
            if rnd_counter == 0:
                new_style_reference=current_style_ref
            else:
                new_style_reference=np.concatenate([new_style_reference,current_style_ref],axis=2)
            rnd_counter+=1
        style_reference=new_style_reference

    style_reference = np.expand_dims(style_reference, axis=0)


    return style_reference




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
