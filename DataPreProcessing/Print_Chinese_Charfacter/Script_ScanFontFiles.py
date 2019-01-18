# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import sys
import os
sys.path.append('..')
sys.path.append('../../')



import matplotlib.pyplot as plt
import pylab
import matplotlib.image as mpimg
import numpy as np
import shutil
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from utilities.utils import image_show, image_revalue

reload(sys)
sys.setdefaultencoding("utf-8")

CN_CHARSET = None
CN_T_CHARSET = None
JP_CHARSET = None
KR_CHARSET = None



input_args = ['--dst_font_dir','/DataA/Harric/ChineseCharacterExp/CASIA_Dataset/Sources/64_FoundContentPrototypeTtfOtfs/Simplified/',
              ]




parser = argparse.ArgumentParser(description='Convert font to images')


parser.add_argument('--dst_font_dir', dest='dst_font_dir', required=True, help='path of the target font dir')
parser.add_argument('--char_size', dest='char_size', type=int, default=150, help='character size')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', dest='x_offset', type=int, default=20, help='x offset')
parser.add_argument('--y_offset', dest='y_offset', type=int, default=20, help='y_offset')

args = parser.parse_args(input_args)







def draw_example(ch, dst_font, canvas_size, x_offset, y_offset):
    def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
        img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
        except IOError:
            return False

        img_matrix = np.asarray(img)[:, :, 0]
        if np.min(img_matrix) == np.max(img_matrix) or (0 not in img_matrix):
            return False
        else:
            return True

    return_mark = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    return return_mark




def font2img(charset,
             character_label_list,
             dst,
             char_size,
             canvas_size,
             x_offset,
             y_offset):
    dst_font = ImageFont.truetype(dst, size=char_size)

    count = 0
    valid_counter = 0
    invalid_counter = 0
    for c in charset:


        if count == len(charset):
            break
        return_mark = draw_example(c, dst_font, canvas_size, x_offset, y_offset)

        if return_mark:
            valid_counter+=1
        else:
            invalid_counter+=1
        count += 1
    return valid_counter,invalid_counter



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




    charset_level1_simplified, character_label_level1_simplified = \
        get_chars_set_from_level1_2(path='../charset/GB2312_Level1_Simplified.txt',level=1)
    charset_level2_simplified, character_label_level2_simplified = \
        get_chars_set_from_level1_2(path='../charset/GB2312_Level2_Simplified.txt',level=2)
    charset_level1_traditional, character_label_level1_traditional = \
        get_chars_set_from_level1_2(path='../charset/GB2312_Level1_Traditional.txt', level=1)
    charset_level2_traditional, character_label_level2_traditional = \
        get_chars_set_from_level1_2(path='../charset/GB2312_Level2_Traditional.txt', level=2)



    font_counter=0
    for dst_file in dst_file_list:

        file_name=dst_file.split('/')
        file_name=file_name[len(file_name)-1]

        valid_l1_simplified,invalid_l1_simplified= \
            font2img(charset=charset_level1_simplified,
                     character_label_list=character_label_level1_simplified, dst=dst_file,
                     char_size=args.char_size,canvas_size=args.canvas_size,
                     x_offset=args.x_offset, y_offset=args.y_offset)
        valid_l2_simplified, invalid_l2_simplified = \
            font2img(charset=charset_level2_simplified,
                     character_label_list=character_label_level2_simplified,
                     dst=dst_file,
                     char_size=args.char_size,canvas_size=args.canvas_size,
                     x_offset=args.x_offset, y_offset=args.y_offset)

        valid_l1_traditional, invalid_l1_traditional = \
            font2img(charset=charset_level1_traditional,
                     character_label_list=character_label_level1_traditional, dst=dst_file,
                     char_size=args.char_size, canvas_size=args.canvas_size,
                     x_offset=args.x_offset, y_offset=args.y_offset)
        valid_l2_traditional, invalid_l2_traditional = \
            font2img(charset=charset_level2_traditional,
                     character_label_list=character_label_level2_traditional,
                     dst=dst_file,
                     char_size=args.char_size, canvas_size=args.canvas_size,
                     x_offset=args.x_offset, y_offset=args.y_offset)

        font_counter=font_counter+1

        dst_file_name = dst_file.split("/")[-1]

        print("%s, ScanningFont:%d/%d: ValidL1_Smp:%d,InvalidL1_Smp:%d; ValidL2_Smp:%d,InvalidL2_Smp:%d;; ValidL1_Trd:%d,InvalidL1_Trd:%d; ValidL2_Trd:%d,InvalidL2_Trd:%d"
              % (dst_file_name, font_counter, len(dst_file_list),
                 valid_l1_simplified, invalid_l1_simplified, valid_l2_simplified, invalid_l2_simplified,
                 valid_l1_traditional, invalid_l1_traditional, valid_l2_traditional, invalid_l2_traditional))






