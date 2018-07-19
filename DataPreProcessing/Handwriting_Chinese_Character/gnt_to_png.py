#!/usr/bin/python

import struct
import os
import argparse

import matplotlib.pyplot as plt
from PIL import Image as img
import numpy as np
import shutil
import pylab

print_separater="##########################################################################"

input_args = [
              '--src_gnt_dir','/data/Harric/CASIA_Sources/HandWritingSources/CASIA-HWDB1.1/',
              '--dst_png_dir_256','/data/Harric/CASIA_256/HandWritingData/SingleChars/CASIA-HWDB1.1/',
              '--dst_png_dir_128','/data/Harric/CASIA_128/HandWritingData/SingleChars/CASIA-HWDB1.1/',
              '--dst_png_dir_64','/data/Harric/CASIA_64/HandWritingData/SingleChars/CASIA-HWDB1.1/',
              '--num_writer_each_group','5']


parser = argparse.ArgumentParser(description='Convert gnt to pngs')
parser.add_argument('--src_gnt_dir', dest='src_gnt_dir', required=True, help='path of the source gnt')
parser.add_argument('--dst_png_dir_256', dest='dst_png_dir_256', required=True, help='path of the target png')
parser.add_argument('--dst_png_dir_128', dest='dst_png_dir_128', required=True, help='path of the target png')
parser.add_argument('--dst_png_dir_64', dest='dst_png_dir_64', required=True, help='path of the target png')

parser.add_argument('--num_writer_each_group', dest='num_writer_each_group', required=True, help='num_writer_each_group')


args = parser.parse_args(input_args)



def Gnt_2_Png(src_path,
              saving_path_prefix_256,
              saving_path_prefix_128,
              saving_path_prefix_64,):



    writer = os.path.split(src_path)[1]
    writer_index = writer.find('-c')
    writer_id = int(writer[0:writer_index])

    f = open(src_path, 'rb')
    count = 0
    while f.read(1) != "":
        f.seek(-1, 1)

        length_bytes = struct.unpack('<I', f.read(4))[0]
        tag_code = f.read(2)
        character_id_1 = ord(tag_code[0])
        character_id_2 = ord(tag_code[1])
        character_id = str(character_id_1) + str(character_id_2)

        if int(character_id_1) >= 176 and int(character_id_1) != 255:
            width = struct.unpack('<H', f.read(2))[0]
            height = struct.unpack('<H', f.read(2))[0]

            im = img.new('RGB', (width, height))
            img_array = im.load()
            for x in xrange(0, height):
                for y in xrange(0, width):
                    pixel = struct.unpack('<B', f.read(1))[0]
                    img_array[y, x] = (pixel, pixel, pixel)

            im = im.resize((150, 150))
            im_np_matrix = np.asarray(im)
            im_np_vector = np.reshape(im_np_matrix,[im_np_matrix.shape[0]*im_np_matrix.shape[1]*im_np_matrix.shape[2],1])
            im_np_vector.flags.writeable = True
            im_np_vector[np.where(im_np_vector < 240)] = 0
            im_np_vector[np.where(im_np_vector > 240)] = 255
            im_np_matrix = np.reshape(im_np_matrix,[im_np_matrix.shape[0], im_np_matrix.shape[1], im_np_matrix.shape[2]])
            im = img.fromarray(np.uint8(im_np_matrix))

            im_output_256 = img.new("RGB", (256, 256), (255, 255, 255))
            im_output_256.paste(im,(52,52))
            filename = ("%09d_%s_%05d.png" % (count,character_id, writer_id))


            save_path_with_file_name_256 = os.path.join(saving_path_prefix_256, filename)
            im_output_256.save(save_path_with_file_name_256)

            im_output_128 = im_output_256.resize((128, 128), img.ANTIALIAS)
            save_path_with_file_name_128 = os.path.join(saving_path_prefix_128, filename)
            im_output_128.save(save_path_with_file_name_128)

            im_output_64 = im_output_256.resize((64, 64), img.ANTIALIAS)
            save_path_with_file_name_64 = os.path.join(saving_path_prefix_64, filename)
            im_output_64.save(save_path_with_file_name_64)
            count += 1
        else:
            print("Abnormal:%s",save_path_with_file_name_256)

    f.close()

    return count




if __name__ == "__main__":


    gnt_file_list = list()
    for root, dirs, files in os.walk(args.src_gnt_dir):
        for name in files:
            file_path = (os.path.join(root, name))
            file_extension = os.path.splitext(file_path)[1]
            if file_extension == '.gnt':
                gnt_file_list.append(os.path.join(root,name))
    gnt_file_list.sort()
    last_file = os.path.split(gnt_file_list[len(gnt_file_list)-1])[1]
    last_writer_index = last_file.find('-c')
    last_writer_id = int(last_file[0:last_writer_index])
    print("TotalWriter:%d" % len(gnt_file_list))
    print(print_separater)
    #raw_input("Please enter to continue")
    print(print_separater)





    if os.path.exists(args.dst_png_dir_256):
        shutil.rmtree(args.dst_png_dir_256)
    if os.path.exists(args.dst_png_dir_128):
        shutil.rmtree(args.dst_png_dir_128)
    if os.path.exists(args.dst_png_dir_64):
        shutil.rmtree(args.dst_png_dir_64)

    os.makedirs(args.dst_png_dir_256)
    os.makedirs(args.dst_png_dir_128)
    os.makedirs(args.dst_png_dir_64)


    ii=1
    full_counter=0
    next_writer_id=args.num_writer_each_group
    for individual_path in gnt_file_list:
        file_name = os.path.split(individual_path)[1]
        writer_index = file_name.find('-c')
        writer_id = int(file_name[0:writer_index])

        if ii == 1 or writer_id > next_writer_id:
            prev_writer_id = writer_id
            next_writer_id = writer_id + int(args.num_writer_each_group) - 1
            if next_writer_id > last_writer_id:
                next_writer_id = last_writer_id


            current_save_path = ("writer%05d_To_%05d" % (prev_writer_id,next_writer_id))
            print(current_save_path)

            current_save_path_256 = os.path.join(args.dst_png_dir_256,current_save_path)
            os.makedirs(current_save_path_256)
            current_save_path_128 = os.path.join(args.dst_png_dir_128, current_save_path)
            os.makedirs(current_save_path_128)
            current_save_path_64 = os.path.join(args.dst_png_dir_64, current_save_path)
            os.makedirs(current_save_path_64)



        this_writer_counter = Gnt_2_Png(src_path=individual_path,
                                        saving_path_prefix_256=current_save_path_256,
                                        saving_path_prefix_128=current_save_path_128,
                                        saving_path_prefix_64=current_save_path_64)
        full_counter+=this_writer_counter
        print("Writer:%d, Counter:%d/%d" % (writer_id,this_writer_counter,full_counter))
        print(print_separater)


        ii+=1

