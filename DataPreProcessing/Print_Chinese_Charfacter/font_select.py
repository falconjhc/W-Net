# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import sys
import numpy as np
import os
import shutil
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import collections


import matplotlib.pyplot as plt

reload(sys)
sys.setdefaultencoding("utf-8")


parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--source_font_dir', dest='source_font_dir', default='./Fonts_TrueType/', help='path of the source font dir')
parser.add_argument('--preview_font_dir', dest='preview_font_dir', default='./Font_Img_new/', help='path of the preview font dir')
parser.add_argument('--new_source_font_dir', dest='new_source_font_dir', default='./Font_new/', help='path of the source font dir')

args = parser.parse_args()


if __name__ == "__main__":

    org_font_file_list=list()
    org_font_file_name_list=list()
    for root, dirs, files in os.walk(args.source_font_dir):
        for filepath in files:
            org_font_file_list.append(os.path.join(root,filepath))
            org_font_file_name_list.append(filepath)

    selected_font_img_file_list = list()
    selected_font_img_name_file_list = list()
    selected_font_img_dir_list = list()
    for root, dirs, files in os.walk(args.preview_font_dir):
        for filepath in files:
            selected_font_img_file_list.append(os.path.join(root, filepath))
            selected_font_img_dir_list.append(root)
            selected_font_img_name_file_list.append(filepath)




    counter=0
    for file_name in selected_font_img_name_file_list:



        # this_img_file=selected_font_img_file_list[counter]
        # this_img_font=this_img_file.replace(args.preview_font_dir,args.source_font_dir)


        found=0
        file_name = file_name.replace('jpg', 'otf')
        if file_name in org_font_file_name_list:
            index=org_font_file_name_list.index(file_name)
            this_img_font_final = file_name
            found=1

        file_name = file_name.replace('otf', 'ttf')
        if file_name in org_font_file_name_list:
            index=org_font_file_name_list.index(file_name)
            this_img_font_final = file_name
            found=1

        file_name = file_name.replace('ttf', 'OTF')
        if file_name in org_font_file_name_list:
            index=org_font_file_name_list.index(file_name)
            this_img_font_final = file_name
            found=1

        file_name = file_name.replace('OTF', 'TTF')
        if file_name in org_font_file_name_list:
            index=org_font_file_name_list.index(file_name)
            this_img_font_final = file_name
            found=1



        org_file = org_font_file_list[index]
        org_file_name=org_font_file_name_list[index]
        new_file_path = selected_font_img_dir_list[counter]
        new_file_path=new_file_path.replace(args.preview_font_dir,args.new_source_font_dir)
        if os.path.exists(new_file_path) == False:
            os.makedirs(new_file_path)
        new_file=os.path.join(new_file_path,org_file_name)



        shutil.copyfile(org_file,new_file)




        print('path:%s,counter:%d' % (new_file,counter))
        counter = counter + 1
