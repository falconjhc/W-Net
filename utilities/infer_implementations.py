# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import cv2
import utilities.charcut as cc

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

GRAYSCALE_AVG = 127.5

print_separater = "#################################################################"

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

import copy as cp


def get_chars(path):
    chars = list()
    with open(path) as f:
        for line in f:

            line = u"%s" % line
            char_counter = 0
            for char in line:

                current_char = line[char_counter]
                chars.append(current_char)



                char_counter += 1
    return chars

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
    def read_content_from_dir():

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
                            return -1, -1, False

                        targeted_character_label0_list.append(str(character_id))
                        targeted_chars_list.append(current_char)
        actual_char_list = line
        print("In total %d targeted chars are found in the standard GB2312 set" % len(targeted_chars_list))



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
        corresponding_content_prototype = np.zeros(
            shape=[len(targeted_character_label0_list), img_width, img_width, img_filters * len(content_label1_vec)],
            dtype=np.float32)
        label1_counter = 0
        for content_label1 in content_label1_vec:
            current_label1_indices = [ii for ii in range(len(content_label1_list)) if
                                      content_label1_list[ii] == content_label1]
            current_label0_on_current_label1 = list()
            current_data_on_current_label1 = list()
            for ii in current_label1_indices:
                current_label0_on_current_label1.append(content_label0_list[ii])
                current_data_on_current_label1.append(content_data_list[ii])
            target_counter = 0
            for ii in targeted_character_label0_list:
                if ii not in current_label0_on_current_label1:
                    print("Fails! Didnt find %s" % unicode(actual_char_list[target_counter]))
                    return -1, -1, False
                else:
                    index_found = current_label0_on_current_label1.index(ii)
                    char_img = misc.imread(current_data_on_current_label1[index_found])
                    # print("%d %d" % (label1_counter,target_counter))
                    if char_img.ndim == 3:
                        char_img = np.expand_dims(char_img[:, :, 0], axis=2)
                    elif char_img.ndim == 2:
                        char_img = np.expand_dims(char_img, axis=2)
                    char_img = char_img / GRAYSCALE_AVG - 1
                    corresponding_content_prototype[target_counter, :, :,
                    label1_counter * img_filters:(label1_counter + 1) * img_filters] \
                        = char_img
                target_counter += 1

            label1_counter += 1

        print("In total %d targeted chars are corresponded with content prototypes" % len(targeted_chars_list))
        print(print_separater)
        return corresponding_content_prototype, content_label1_vec


    def draw_single_char(ch, font):
        canvas_size = 256
        x_offset = 20
        y_offset = 20

        img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)

        img_matrix = np.asarray(img)[:, :, 0]
        zero_indices = np.where(img_matrix == 0)
        exceed = 'NONE'

        if np.min(img_matrix) == np.max(img_matrix) or (0 not in img_matrix):

            img_output = np.zeros(shape=[256,256,3])
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

            if not exceed=='NONE':
                if exceed=='LEFT':
                    added_pixels = np.ones([img_matrix_cut.shape[0],exceed_pixels]) * 255
                    img_matrix_cut = np.concatenate([added_pixels, img_matrix_cut], axis=1)
                elif exceed=='RIGHT':
                    added_pixels = np.ones([img_matrix_cut.shape[0], exceed_pixels]) * 255
                    img_matrix_cut = np.concatenate([img_matrix_cut,added_pixels], axis=1)
                elif exceed=='UP':
                    added_pixels = np.ones([exceed_pixels, img_matrix_cut.shape[1]]) * 255
                    img_matrix_cut = np.concatenate([added_pixels, img_matrix_cut], axis=0)
                elif exceed=='DOWN':
                    added_pixels = np.ones([exceed_pixels, img_matrix_cut.shape[1]]) * 255
                    img_matrix_cut = np.concatenate([img_matrix_cut, added_pixels], axis=0)


            img_matrix_cut = np.tile(np.reshape(img_matrix_cut,
                                                [img_matrix_cut.shape[0], img_matrix_cut.shape[1], 1]),
                                     [1, 1, 3])
            img_cut = Image.fromarray(np.uint8(img_matrix_cut))
            img_resize = img_cut.resize((150, 150), Image.ANTIALIAS)
            img_output = Image.new("RGB", (256, 256), (255, 255, 255))
            img_output.paste(img_resize, (52, 52))
            img_output.resize((64,64), Image.ANTIALIAS)

        return img_output.resize((64,64), Image.ANTIALIAS)

    def generate_content_from_ttf():
        targeted_chars_list = list()
        targeted_character_label0_list = list()

        with open(targeted_content_input_txt) as f:
            for line in f:

                line = u"%s" % line
                char_counter = 0
                for char in line:

                    current_char = line[char_counter]
                    char_counter += 1

                    targeted_character_label0_list.append(str(current_char))
                    targeted_chars_list.append(current_char)
        print("In total %d targeted chars are found." % len(targeted_chars_list))



        content_label1_vec = list()
        corresponding_content_prototype = np.zeros(
            shape=[len(targeted_character_label0_list),
                   img_width, img_width,
                   img_filters * len(content_file_data_dir)],
            dtype=np.float32)


        for label1_counter in range(len(content_file_data_dir)):
            current_font_misc = ImageFont.truetype(content_file_data_dir[label1_counter], size=150)
            current_file_path = content_file_data_dir[label1_counter]
            current_file_name = os.path.splitext(current_file_path)[0]
            current_file_name = current_file_name.split('/')[len(current_file_name.split('/'))-1]
            content_label1_vec.append(current_file_name)

            for target_counter in range(len(targeted_chars_list)):
                char_misc = draw_single_char(ch=targeted_chars_list[target_counter], font=current_font_misc)
                char_img = np.asarray(char_misc)[:, :, 0]
                char_img = char_img / GRAYSCALE_AVG - 1
                char_img = np.expand_dims(char_img, axis=2)

                if char_img.ndim == 3:
                    char_img = np.expand_dims(char_img[:, :, 0], axis=2)
                elif char_img.ndim == 2:
                    char_img = np.expand_dims(char_img, axis=2)
                corresponding_content_prototype[target_counter, :, :,
                label1_counter * img_filters:(label1_counter + 1) * img_filters] \
                    = char_img

        return corresponding_content_prototype,content_label1_vec





    dir_content = True
    for check_dir in content_file_data_dir:
        if not os.path.isdir(check_dir):
            dir_content=False
            break

    ttf_content = True
    for check_ttf in content_file_data_dir:
        is_file = not os.path.isdir(check_ttf)
        is_ttf = os.path.splitext(check_ttf)[-1]=='.ttf'
        if not (is_file and is_ttf):
            ttf_content=False
            break

    ttc_content = True
    for check_ttc in content_file_data_dir:
        is_file = not os.path.isdir(check_ttc)
        is_ttc = os.path.splitext(check_ttc)[-1]=='.ttc'
        if not (is_file and is_ttc):
            ttc_content=False
            break

    otf_content = True
    for check_otf in content_file_data_dir:
        is_file = not os.path.isdir(check_otf)
        is_otf = os.path.splitext(check_otf)[-1]=='.otf'
        if not (is_file and is_otf):
            otf_content=False
            break

    if dir_content:
        corresponding_content_prototype, content_label1_vec = read_content_from_dir()

    if ttf_content or ttc_content or otf_content:
        corresponding_content_prototype, content_label1_vec = generate_content_from_ttf()

    return corresponding_content_prototype, content_label1_vec, True


def get_style_references(img_path, resave_path, style_input_number):
    if os.path.isdir(img_path):
        style_reference = \
            collect_chars_from_directory(img_path, resave_path)
    else:
        file_extension=os.path.splitext(img_path)[1]
        if file_extension=='.ttf':
            style_reference = \
                generated_from_ttf_otf_files(img_path, resave_path)
        else:
            style_reference = \
                crop_from_full_handwriting_essay_paper(img_path, resave_path)

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
    print("Selected %d style references for generation" % style_reference.shape[3])
    print(print_separater)
    return style_reference

def collect_chars_from_directory(img_path, resave_path):
    counter=0
    for root, dirs, files in os.walk(img_path):
        files.sort()
        for name in files:
            if not ((name.find("DS") == -1) and (name.find("Th") == -1)):
                continue
            file_path = (os.path.join(root, name))
            file_extension = os.path.splitext(file_path)[1]
            if file_extension=='.png':
                char_read=misc.imread(os.path.join(root,name))
                char_read=char_read[:,:,0]
                char_read = char_read / GRAYSCALE_AVG - 1
                char_read = np.expand_dims(char_read, axis=2)
                if counter == 0:
                    style_reference = char_read
                else:
                    style_reference = np.concatenate([style_reference, char_read], axis=2)
                counter+=1

    style_num = style_reference.shape[2]
    row_col_num = np.int64(np.ceil(np.sqrt(style_num)))
    resave_paper = matrix_paper_generation(images=np.expand_dims(np.transpose(style_reference,[2,0,1]),axis=3),
                                           rows=row_col_num,columns=row_col_num)
    misc.imsave(os.path.join(resave_path, 'InputStyleImg.png'), resave_paper)


    return style_reference

def generated_from_ttf_otf_files(img_path, resave_path):

    def draw_single_char(ch, font):
        canvas_size = 256
        x_offset = 20
        y_offset = 20

        img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)

        img_matrix = np.asarray(img)[:, :, 0]
        zero_indices = np.where(img_matrix == 0)
        exceed = 'NONE'

        if np.min(img_matrix) == np.max(img_matrix) or (0 not in img_matrix):

            img_output = np.zeros(shape=[256,256,3])
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

            if not exceed=='NONE':
                if exceed=='LEFT':
                    added_pixels = np.ones([img_matrix_cut.shape[0],exceed_pixels]) * 255
                    img_matrix_cut = np.concatenate([added_pixels, img_matrix_cut], axis=1)
                elif exceed=='RIGHT':
                    added_pixels = np.ones([img_matrix_cut.shape[0], exceed_pixels]) * 255
                    img_matrix_cut = np.concatenate([img_matrix_cut,added_pixels], axis=1)
                elif exceed=='UP':
                    added_pixels = np.ones([exceed_pixels, img_matrix_cut.shape[1]]) * 255
                    img_matrix_cut = np.concatenate([added_pixels, img_matrix_cut], axis=0)
                elif exceed=='DOWN':
                    added_pixels = np.ones([exceed_pixels, img_matrix_cut.shape[1]]) * 255
                    img_matrix_cut = np.concatenate([img_matrix_cut, added_pixels], axis=0)


            img_matrix_cut = np.tile(np.reshape(img_matrix_cut,
                                                [img_matrix_cut.shape[0], img_matrix_cut.shape[1], 1]),
                                     [1, 1, 3])
            img_cut = Image.fromarray(np.uint8(img_matrix_cut))
            img_resize = img_cut.resize((150, 150), Image.ANTIALIAS)
            img_output = Image.new("RGB", (256, 256), (255, 255, 255))
            img_output.paste(img_resize, (52, 52))
            img_output.resize((64,64), Image.ANTIALIAS)



        return img_output.resize((64,64), Image.ANTIALIAS)

    sample_char_set = get_chars(path='../ContentTxt/SampleChars.txt')
    sample_font = ImageFont.truetype(img_path, size=150)
    counter=0
    for current_char in sample_char_set:
        char_misc = draw_single_char(ch=current_char,font=sample_font)
        char_np = np.asarray(char_misc)[:, :, 0]
        char_np = char_np / GRAYSCALE_AVG - 1
        char_np = np.expand_dims(char_np,axis=2)
        if counter==0:
            style_reference = char_np
        else:
            style_reference = np.concatenate([style_reference,char_np],axis=2)
        counter+=1

    style_num = style_reference.shape[2]
    row_col_num = np.int64(np.ceil(np.sqrt(style_num)))
    resave_paper = matrix_paper_generation(images=np.expand_dims(np.transpose(style_reference, [2, 0, 1]), axis=3),
                                           rows=row_col_num, columns=row_col_num)
    misc.imsave(os.path.join(resave_path, 'InputStyleImg.png'), resave_paper)


    return style_reference

def crop_from_full_handwriting_essay_paper(img_path, resave_path):

    img = cv2.imread(img_path)
    img_misc = misc.imread(img_path)
    misc.imsave(os.path.join(resave_path, 'InputStyleImg.png'), img_misc)

    img_new = img
    img_new[np.where(img < 150)] = 0
    img_new[np.where(img >= 150)] = 255
    img = img_new

    image_list = cc.char_cut(img, 37, 64)
    counter = 0
    style_reference = None
    for im_split in image_list:
        img = np.expand_dims(im_split, axis=2)
        img = img / GRAYSCALE_AVG - 1
        if counter == 0:
            style_reference = img
        else:
            style_reference = np.concatenate([style_reference, img], axis=2)
        counter += 1

    print("In total %d style references are extracted from %s" % (style_reference.shape[2],img_path))

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

def numpy_img_save(img,path):
    imgout=cp.deepcopy(img)
    imgout = imgout * 255
    imgout = np.tile(np.reshape(imgout,
                                [imgout.shape[0],
                                 imgout.shape[1], 1]),
                       [1, 1, 3])
    imgout_misc = Image.fromarray(np.uint8(imgout))
    misc.imsave(path,imgout_misc)

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
