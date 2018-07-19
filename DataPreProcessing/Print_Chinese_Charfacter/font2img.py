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

reload(sys)
sys.setdefaultencoding("utf-8")

CN_CHARSET = None
CN_T_CHARSET = None
JP_CHARSET = None
KR_CHARSET = None






def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    return img


def draw_example(ch, src_font, dst_font, canvas_size, x_offset, y_offset):
    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)


    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img


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


def font2img(src, dst, charset, char_size, canvas_size,
             x_offset, y_offset, sample_dir, label=0):
    src_font = ImageFont.truetype(src, size=char_size)
    dst_font = ImageFont.truetype(dst, size=char_size)


    current_pair_dir = ("Font_Pair_No_%d" % label)
    current_pair_dir = os.path.join(sample_dir,current_pair_dir)
    if not os.path.exists(current_pair_dir):
        os.mkdir(current_pair_dir)



    count = 0

    for c in charset:
        if count == len(charset):
            break
        e = draw_example(c, src_font, dst_font, canvas_size, x_offset, y_offset)
        if e:
            e.save(os.path.join(current_pair_dir, "%d_%04d.jpg" % (label, count)))
            count += 1
            if count % 100 == 0:
                print("processed %d chars" % count)
    print("processed %d chars for label no. %d" %(count,label))


def get_chars_set_from_txt(path):
    """
    Expect a text file that each line is a char
    """
    chars = list()
    with open(path) as f:
        for line in f:

            line = u"%s" % line
            char_counter=0
            for char in line:

                #current_char = line.split()[char_counter]
                current_char = line[char_counter]
                chars.append(current_char)
                char_counter+=1
    return chars


parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--src_font', dest='src_font', default='./Font_ttf_otf/正体黑/微软雅黑.ttf', help='path of the source font')
parser.add_argument('--dst_font', dest='dst_font', default='./Font_ttf_otf/手写体正常/邯郸-郭灵霞灵芝体.ttf', help='path of the target font')
parser.add_argument('--char_size', dest='char_size', type=int, default=150, help='character size')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', dest='x_offset', type=int, default=20, help='x offset')
parser.add_argument('--y_offset', dest='y_offset', type=int, default=20, help='y_offset')
parser.add_argument('--sample_dir', dest='sample_dir', default='./Font_Pairs',help='directory to save examples')
parser.add_argument('--label', dest='label', type=int, default=8, help='label as the prefix of examples')

args = parser.parse_args()

if __name__ == "__main__":
    # if args.charset in ['CN', 'JP', 'KR', 'CN_T']:
    #     charset = locals().get("%s_CHARSET" % args.charset)
    # else:
    #     charset = [c for c in open(args.charset).readline()[:-1].decode("utf-8")]
    # if args.shuffle:
    #     np.random.shuffle(charset)

    charset_train = get_chars_set_from_txt('./charset/GB2312_Level_1.txt')
    charset_val = get_chars_set_from_txt('./charset/GB2312_Level_2.txt')

    font2img(args.src_font, args.dst_font, charset_train, args.char_size,
             args.canvas_size, args.x_offset, args.y_offset,
             os.path.join(args.sample_dir,'train'), args.label)

    font2img(args.src_font, args.dst_font, charset_val, args.char_size,
             args.canvas_size, args.x_offset, args.y_offset,
             os.path.join(args.sample_dir, 'val'), args.label)