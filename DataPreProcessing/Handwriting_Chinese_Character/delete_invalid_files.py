import os
import shutil
checking_path = '../../debug_datasets/writer01001_To_01005'

tgt_dir_list = os.listdir(checking_path)
tgt_dir_list.sort()

dir_valid = os.path.isdir(tgt_dir_list[0])
if dir_valid:
    for curt_dir in tgt_dir_list:
        for root, dirs, files in os.walk(os.path.join(checking_path, curt_dir)):
            for name in files:
                file_path = (os.path.join(root, name))
                file_extension = os.path.splitext(file_path)[1]
                if file_extension == '.png':
                    split_1 = file_path.split('_', file_path.count('_'))

                    character_id = split_1[len(split_1) - 2][
                                   split_1[len(split_1) - 2].rfind('/') + 1:split_1[len(split_1) - 2].rfind('/') + 7]
                    character_id_1 = int(character_id[0:3])
                    character_id_2 = int(character_id[3:])

                    if character_id_1 == 255 or character_id_1 < 176:
                        os.remove(file_path)
                        print("Removed:" + file_path)


else:
    for root, dirs, files in os.walk(os.path.join(checking_path)):
        for name in files:
            file_path = (os.path.join(root, name))
            file_extension = os.path.splitext(file_path)[1]
            if file_extension == '.png':
                split_1 = file_path.split('_', file_path.count('_'))

                character_id = split_1[len(split_1) - 2][
                               split_1[len(split_1) - 2].rfind('/') + 1:split_1[len(split_1) - 2].rfind('/') + 7]
                character_id_1 = int(character_id[0:3])
                character_id_2 = int(character_id[3:])

                if character_id_1 == 255 or character_id_1 < 176:
                    os.remove(file_path)
                    print("Removed:" + file_path)





