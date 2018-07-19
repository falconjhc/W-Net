import struct
import os
import argparse

import matplotlib.pyplot as plt
import Image as img
import shutil


input_args = [
              '--src_dir','../../Point04_Org/',
              '--dst_pair_dir','../../Point04_Paired/',
              '--dst_single_dir','../../Point04_Single/']


parser = argparse.ArgumentParser(description='Find Point04 Experimental Data')
parser.add_argument('--src_dir', dest='src_dir', required=True, help='path of the source')
parser.add_argument('--dst_pair_dir', dest='dst_pair_dir', required=True, help='path of the target')
parser.add_argument('--dst_single_dir', dest='dst_single_dir', required=True, help='path of the target')

args = parser.parse_args(input_args)




if __name__ == "__main__":

    if os.path.exists(args.dst_pair_dir):
        shutil.rmtree(args.dst_pair_dir)
    os.makedirs(args.dst_pair_dir)
    if os.path.exists(args.dst_single_dir):
        shutil.rmtree(args.dst_single_dir)
    os.makedirs(args.dst_single_dir)


    neutral_file_list=list()
    for root, dirs, files in os.walk(args.src_dir):
        files.sort()
        for file in files:
            file_path = (os.path.join(root, file))
            file_extension = os.path.splitext(file_path)[1]
            if ('+0+0') in file and file_extension == '.jpg':
                neutral_file_list.append(os.path.join(root,file))



    file_counter=0
    for root, dirs, files in os.walk(args.src_dir):



        files.sort()
        for file in files:
            file_path = (os.path.join(root, file))
            file_extension = os.path.splitext(file_path)[1]
            if file_extension == '.jpg':


                if file_path.find("Div01") !=-1:
                    curt_single_dir = os.path.join(args.dst_single_dir, "Div01")
                    curt_pair_dir = os.path.join(args.dst_pair_dir, "Div01")


                if file_path.find("Div02") !=-1:
                    curt_single_dir = os.path.join(args.dst_single_dir, "Div02")
                    curt_pair_dir = os.path.join(args.dst_pair_dir, "Div02")

                if not os.path.exists(curt_single_dir):
                    os.makedirs(curt_single_dir)
                if not os.path.exists(curt_pair_dir):
                    os.makedirs(curt_pair_dir)


                idx_position1=file_path.find('personne')+8+5
                idx_position2=file_path.find(file_extension)

                curt_group_id = file_path[file_path.find('personne'):file_path.find('personne')+8+3]


                split_pos=list()
                counter=0
                person_id = file_path[idx_position1-5:idx_position1-5+2]
                for c in file_path:
                    if c=='+' or c=='-':
                        split_pos.append(counter)



                    counter+=1
                vertical_angle_str = file_path[idx_position1:split_pos[1]]
                horizontal_angle_str = file_path[split_pos[1]:idx_position2]
                vertical_angle = int (vertical_angle_str)
                horizontal_angle = int(horizontal_angle_str)


                if vertical_angle==0:


                    for neutral_file in neutral_file_list:
                        if curt_group_id in neutral_file:
                            curt_neutral_file = neutral_file
                            break


                    img_style = img.open(file_path)
                    img_neutral = img.open(curt_neutral_file)
                    img_style_resized = img_style.resize((256,256))
                    img_neutral_resized = img_neutral.resize((256, 256))

                    pair_img = img.new("RGB",(256*2,256),(255,255,255))
                    pair_img.paste(img_style_resized,(0,0))
                    pair_img.paste(img_neutral_resized,(255,0))


                    file_name_pair = ("%09d_%06d_%05d.jpg" % (file_counter,int(person_id),horizontal_angle))
                    file_name_single = ("%06d_%05d.png" % (int(person_id), horizontal_angle))


                    pair_img.save(os.path.join(curt_pair_dir,file_name_pair))
                    img_style_resized.save(os.path.join(curt_single_dir,file_name_single))
                    print(file_name_pair + '@@@' + file_name_single)


                    file_counter+=1
