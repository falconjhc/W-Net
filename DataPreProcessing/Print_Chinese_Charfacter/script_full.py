# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import sys
import os


import matplotlib.pyplot as plt
import pylab
import matplotlib.image as mpimg
import numpy as np
import shutil
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

reload(sys)
sys.setdefaultencoding("utf-8")

CN_CHARSET = None
CN_T_CHARSET = None
JP_CHARSET = None
KR_CHARSET = None



input_args = ['--dst_font_dir','/data/Harric/CASIA_Sources/PrintedSources/80_PF/',
              '--src_font','../standard_font.ttf',
              '--paired_img_dir_256','/data/Harric/CASIA_256/PrintedData/PairChars/',
              '--single_img_dir_256', '/data/Harric/CASIA_256/PrintedData/SingleChars/',
              '--paired_img_dir_128', '/data/Harric/CASIA_128/PrintedData/PairChars/',
              '--single_img_dir_128', '/data/Harric/CASIA_128/PrintedData/SingleChars/',
              '--paired_img_dir_64', '/data/Harric/CASIA_64/PrintedData/PairChars/',
              '--single_img_dir_64', '/data/Harric/CASIA_64/PrintedData/SingleChars/'
              ]




parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--src_font', dest='src_font', required=True, help='path of the source font')


parser.add_argument('--dst_font_dir', dest='dst_font_dir', required=True, help='path of the target font dir')

parser.add_argument('--char_size', dest='char_size', type=int, default=150, help='character size')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', dest='x_offset', type=int, default=20, help='x offset')
parser.add_argument('--y_offset', dest='y_offset', type=int, default=20, help='y_offset')


parser.add_argument('--paired_img_dir_256', dest='paired_img_dir_256', required=True,help='directory to save paired imgs')
parser.add_argument('--single_img_dir_256', dest='single_img_dir_256', required=True,help='directory to save paired imgs')
parser.add_argument('--paired_img_dir_128', dest='paired_img_dir_128', required=True,help='directory to save paired imgs')
parser.add_argument('--single_img_dir_128', dest='single_img_dir_128', required=True,help='directory to save paired imgs')
parser.add_argument('--paired_img_dir_64', dest='paired_img_dir_64', required=True,help='directory to save paired imgs')
parser.add_argument('--single_img_dir_64', dest='single_img_dir_64', required=True,help='directory to save paired imgs')

args = parser.parse_args(input_args)







def draw_example(ch, src_font, dst_font, canvas_size, x_offset, y_offset):
    def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
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



        return img_output


    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)

    invalid_img = (np.max(dst_img) == np.min(dst_img))


    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img,dst_img,invalid_img




def font2img(charset,
             character_label_list,
             src,
             dst,
             char_size,
             canvas_size,
             x_offset,
             y_offset,
             pair_save_dir_256,
             single_save_dir_256,
             pair_save_dir_128,
             single_save_dir_128,
             pair_save_dir_64,
             single_save_dir_64,
             pair_label=0):
    src_font = ImageFont.truetype(src, size=char_size)
    dst_font = ImageFont.truetype(dst, size=char_size)

    



    current_pair_dir = ("Font_Pair_No_%04d" % pair_label)
    current_single_dir = ("Font_No_%04d" % pair_label)

    current_pair_dir_256 = os.path.join(pair_save_dir_256,current_pair_dir)
    if not os.path.exists(current_pair_dir_256):
        os.makedirs(current_pair_dir_256)
    current_single_dir_256 = os.path.join(single_save_dir_256, current_single_dir)
    if not os.path.exists(current_single_dir_256):
        os.makedirs(current_single_dir_256)

    current_pair_dir_128 = os.path.join(pair_save_dir_128, current_pair_dir)
    if not os.path.exists(current_pair_dir_128):
        os.makedirs(current_pair_dir_128)
    current_single_dir_128 = os.path.join(single_save_dir_128, current_single_dir)
    if not os.path.exists(current_single_dir_128):
        os.makedirs(current_single_dir_128)

    current_pair_dir_64 = os.path.join(pair_save_dir_64, current_pair_dir)
    if not os.path.exists(current_pair_dir_64):
        os.makedirs(current_pair_dir_64)
    current_single_dir_64 = os.path.join(single_save_dir_64, current_single_dir)
    if not os.path.exists(current_single_dir_64):
        os.makedirs(current_single_dir_64)



    count = 0

    for c in charset:


        if count == len(charset):
            break
        pair_256, single_256, invalid_img = draw_example(c, src_font, dst_font, canvas_size, x_offset, y_offset)
        if pair_256 and (not invalid_img):
            pair_256.save(os.path.join(current_pair_dir_256, "%09d_%s_%05d.jpg" % (count, character_label_list[count],pair_label)))
            single_256.save(os.path.join(current_single_dir_256, "%09d_%s_%05d.png" % (count,character_label_list[count], pair_label)))

            pair_128 = pair_256.resize((256,128),Image.ANTIALIAS)
            single_128 = single_256.resize((128,128),Image.ANTIALIAS)
            pair_128.save(os.path.join(current_pair_dir_128,"%09d_%s_%05d.jpg" % (count, character_label_list[count], pair_label)))
            single_128.save(os.path.join(current_single_dir_128,"%09d_%s_%05d.png" % (count, character_label_list[count], pair_label)))

            pair_64 = pair_256.resize((128, 64), Image.ANTIALIAS)
            single_64 = single_256.resize((64, 64), Image.ANTIALIAS)
            pair_64.save(os.path.join(current_pair_dir_64, "%09d_%s_%05d.jpg" % (count, character_label_list[count], pair_label)))
            single_64.save(os.path.join(current_single_dir_64, "%09d_%s_%05d.png" % (count, character_label_list[count], pair_label)))

            count += 1



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







def find_file_list( path,train_mark):
    file_list=list()
    counter=0

    dir_list=os.listdir(path)
    dir_list.sort()

    for dir in dir_list:
        current_full_path=os.path.join(path,dir)
        files = [f for f in os.listdir(current_full_path) if os.path.isfile(os.path.join(current_full_path, f))]
        files.sort()
        for fn in files:
            if fn.find('.jpg') != -1:
                file_list.append(os.path.join(current_full_path,fn))
                counter = counter + 1



    return file_list




















if __name__ == "__main__":

    dir_serach_list=args.dst_font_dir.split(",")

    dst_file_list=[]
    for current_dir in dir_serach_list:
        for root, dirs, files in os.walk(current_dir):
            for name in files:

                file_path = (os.path.join(root, name))
                file_extension = os.path.splitext(file_path)[1]

                if file_extension == '.TTF' or file_extension == '.OTF' or file_extension == '.ttf' or file_extension == '.otf':
                    dst_file_list.append(os.path.join(root, name))

    dst_file_list.sort()




    charset_level1, character_label_level1 = get_chars_set_from_level1_2(path='../charset/GB2312_Level_1.txt',level=1)
    charset_level2, character_label_level2 = get_chars_set_from_level1_2(path='../charset/GB2312_Level_2.txt',level=2)


    if os.path.exists(args.paired_img_dir_256):
        shutil.rmtree(args.paired_img_dir_256)
    if os.path.exists(args.single_img_dir_256):
        shutil.rmtree(args.single_img_dir_256)
    if os.path.exists(args.paired_img_dir_128):
        shutil.rmtree(args.paired_img_dir_128)
    if os.path.exists(args.single_img_dir_128):
        shutil.rmtree(args.single_img_dir_128)
    if os.path.exists(args.paired_img_dir_64):
        shutil.rmtree(args.paired_img_dir_64)
    if os.path.exists(args.single_img_dir_64):
        shutil.rmtree(args.single_img_dir_64)


    os.makedirs(args.paired_img_dir_256)
    os.makedirs(args.single_img_dir_256)
    os.makedirs(args.paired_img_dir_128)
    os.makedirs(args.single_img_dir_128)
    os.makedirs(args.paired_img_dir_64)
    os.makedirs(args.single_img_dir_64)



    font_counter=0
    for dst_file in dst_file_list:



        file_name=dst_file.split('/')
        file_name=file_name[len(file_name)-1]
        pair_label=file_name.split('_')
        pair_label=int(pair_label[0])

        print("%s, Processing Font-->Pict:%d/%d with PairLabel:%04d" % (dst_file, font_counter, len(dst_file_list),pair_label))

        font2img(charset=charset_level1,
                 character_label_list=character_label_level1,
                 src=args.src_font, dst=dst_file,
                 char_size=args.char_size,canvas_size=args.canvas_size,
                 x_offset=args.x_offset, y_offset=args.y_offset,
                 pair_save_dir_256=os.path.join(args.paired_img_dir_256,'GB2312_L1'),
                 single_save_dir_256=os.path.join(args.single_img_dir_256,'GB2312_L1'),
                 pair_save_dir_128=os.path.join(args.paired_img_dir_128, 'GB2312_L1'),
                 single_save_dir_128=os.path.join(args.single_img_dir_128, 'GB2312_L1'),
                 pair_save_dir_64=os.path.join(args.paired_img_dir_64, 'GB2312_L1'),
                 single_save_dir_64=os.path.join(args.single_img_dir_64, 'GB2312_L1'),
                 pair_label=pair_label)
        font2img(charset=charset_level2,
                 character_label_list=character_label_level2,
                 src=args.src_font, dst=dst_file,
                 char_size=args.char_size,canvas_size=args.canvas_size,
                 x_offset=args.x_offset, y_offset=args.y_offset,
                 pair_save_dir_256=os.path.join(args.paired_img_dir_256, 'GB2312_L2'),
                 single_save_dir_256=os.path.join(args.single_img_dir_256, 'GB2312_L2'),
                 pair_save_dir_128=os.path.join(args.paired_img_dir_128, 'GB2312_L2'),
                 single_save_dir_128=os.path.join(args.single_img_dir_128, 'GB2312_L2'),
                 pair_save_dir_64=os.path.join(args.paired_img_dir_64, 'GB2312_L2'),
                 single_save_dir_64=os.path.join(args.single_img_dir_64, 'GB2312_L2'),
                 pair_label=pair_label)

        font_counter=font_counter+1






