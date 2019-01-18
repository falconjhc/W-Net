# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import sys
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import collections


import matplotlib.pyplot as plt

reload(sys)
sys.setdefaultencoding("utf-8")

CN_CHARSET = None
CN_T_CHARSET = None
JP_CHARSET = None
KR_CHARSET = None

DEFAULT_CHARSET = "./charset/cjk.json"


def load_global_charset():
    global CN_CHARSET, JP_CHARSET, KR_CHARSET, CN_T_CHARSET
    cjk = json.load(open(DEFAULT_CHARSET))
    CN_CHARSET = cjk["gbk"]
    JP_CHARSET = cjk["jp"]
    KR_CHARSET = cjk["kr"]
    CN_T_CHARSET = cjk["gb2312_t"]

def get_chars_set_for_each_char(path):
    """
    Expect a text file that each line is a char
    """
    chars = list()
    with open(path) as f:
        for line in f:
            line = u"%s" % line
            for charIndex in range(len(line)):
                #char = line.split()[charIndex]
                #char = line.split()[0]
                char = line[charIndex]
                chars.append(char)
    return chars



def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    #draw.text((0, 0), ch, (0, 0, 0), font=font)
    #plt.imshow(img)
    return img


def draw_example(source_font, charset, canvas_size, x_offset, y_offset):

    char_length=len(charset)
    output_img = Image.new("RGB", (int(canvas_size * char_length / 2), int(canvas_size*2)), (255, 255, 255))

    counter=0
    for c in charset:


        curt_char_img = draw_single_char(c, source_font, canvas_size, x_offset, y_offset)
        if counter < len(charset) / 2:
            output_img.paste(curt_char_img, (canvas_size*counter, 0))
        else:
            output_img.paste(curt_char_img, (canvas_size * (counter-len(charset) / 2), canvas_size))
        counter+=1

        #plt.imshow(output_img)
        #break
    # check the filter example in the hashes or not


    # output_img=output_img.resize([int(canvas_size * char_length / 2), int(canvas_size * 2)])
    return output_img


def filter_recurring_hash(charset, font, canvas_size, x_offset, y_offset):
    """ Some characters are missing in a given font, filter them
    by checking the recurring hashes
    """
    _charset = charset[:]
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = collections.defaultdict(int)
    for c in sample:
        img = draw_single_char(c, font, canvas_size, x_offset, y_offset)
        hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]


def font2img(src_path, dst_path, charset, char_size, canvas_size,
             x_offset, y_offset):

    # counter=0
    # char_list=list()
    # for c in charset:
    #     char_list.append(ImageFont.truetype(src, size=char_size))
    #     #font[counter] = ImageFont.truetype(src, size=char_size)
    #     counter=+1

    source_font=ImageFont.truetype(src_path, size=char_size)

    # filter_hashes = set()
    # if filter_by_hash:
    #     filter_hashes = set(filter_recurring_hash(charset, char_list, canvas_size, x_offset, y_offset))
    #     print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))



    char_img = draw_example(source_font,charset, canvas_size, x_offset, y_offset)

    char_img.save(dst_path)
    #
    # count = 0
    #
    # for c in charset:
    #     e = draw_example(c, char_list,  canvas_size, x_offset, y_offset, filter_hashes)
    #     if e:
    #         e.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)))
    #         count += 1
    #         if count % 100 == 0:
    #             print("processed %d chars" % count)


load_global_charset()
parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--source_font_dir', dest='source_font_dir', default='../Font_sources', help='path of the source font dir')
parser.add_argument('--preview_font_dir', dest='preview_font_dir', default='../Font_jpg_preview', help='path of the preview font dir')
# parser.add_argument('--source_font_dir', dest='source_font_dir', default='/Users/harric/Desktop/201402171232540/', help='path of the source font dir')
# parser.add_argument('--preview_font_dir', dest='preview_font_dir', default='/Users/harric/Desktop/201402171232540_1/', help='path of the preview font dir')

parser.add_argument('--filter', dest='filter', type=int, default=1, help='filter recurring characters')
parser.add_argument('--charset', dest='charset', type=str, default='GB2312_Level_1',
                    help='charset, can be either: CN, JP, KR or a one line file')
parser.add_argument('--shuffle', dest='shuffle', type=int, default=0, help='shuffle a charset before processings')
parser.add_argument('--char_size', dest='char_size', type=int, default=150, help='character size')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', dest='x_offset', type=int, default=20, help='x offset')
parser.add_argument('--y_offset', dest='y_offset', type=int, default=20, help='y_offset')
parser.add_argument('--sample_count', dest='sample_count', type=int, default=10000, help='number of characters to draw')
parser.add_argument('--sample_dir', dest='sample_dir', default='./experiment/font_img',help='directory to save examples')
parser.add_argument('--label', dest='label', type=int, default=4, help='label as the prefix of examples')

args = parser.parse_args()

if __name__ == "__main__":
    charset = get_chars_set_for_each_char("./charset/preview_set.txt")
    charset.remove('\n')


    for root,dirs,files in os.walk(args.source_font_dir):
        for filepath in files:
            read_path=os.path.join(root,filepath)
            write_dir_path=root.replace(args.source_font_dir, args.preview_font_dir)

            if os.path.exists(write_dir_path)==False:
                os.makedirs(write_dir_path)

            #write_path = read_path.replace(args.source_font_dir, args.preview_font_dir)

            otf_found=read_path.find('otf')
            ttf_found=read_path.find('ttf')
            OTF_found=read_path.find('OTF')
            TTF_found=read_path.find('TTF')
            if otf_found==-1 and ttf_found==-1 and OTF_found==-1 and TTF_found==-1:
                continue

            write_path=os.path.join(write_dir_path,filepath)
            write_path=write_path.replace(write_path[len(write_path)-3:len(write_path)],'jpg')
            print('Read:%s, WriteDir:%s, WriteName:%s' % (read_path,write_dir_path,filepath))

            font2img(read_path, write_path, charset, args.char_size,
                     args.canvas_size, args.x_offset, args.y_offset)








