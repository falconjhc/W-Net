import os
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import utilities.infer_implementations as inf_tools
from utilities.infer_implementations import draw_single_char
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import scipy.misc as misc

from utilities.utils import image_show



checking_file_list = list()
lost_data_complete = True
lost_data_complete_font_file = '/DataA/Harric/ChineseCharacterExp/CASIA_Dataset/Sources/PrintedSources/CustomalizedFonts/HeiTi_Chinese.ttf'
# lost_data_complete_font_file = '/DataA/Harric/ChineseCharacterExp/CASIA_Dataset/Sources/PrintedSources/CustomalizedFonts/HeiTi_Korean.ttf'
# lost_data_complete_font_file = '/DataA/Harric/ChineseCharacterExp/CASIA_Dataset/Sources/PrintedSources/CustomalizedFonts/Heiti_Jap1.otf'


input_txt = '../ContentTxt/ContentChars_BlancaPython_32.txt'
# input_txt = '../ContentTxt/StyleChars_Paintings_20.txt'

# file_written_dir = '../EvaluationScripts/EvaluateDataFileLists/HandWritingData/'
file_written_dir = '../EvaluationScripts/EvaluateDataFileLists/PrintedData/'

# datapath_prefix = '/DataA/Harric/ChineseCharacterExp/CASIA_Dataset/HandWritingData_240Binarized/CASIA-HWDB1.1/'
# checking_file_list.append('../FileList/HandWritingData/Char_0_3754_Writer_1151_1200_Isolated.txt')
# checking_file_list.append('../FileList/HandWritingData/Char_0_3754_Writer_1101_1150_Isolated.txt')
# checking_file_list.append('../FileList/HandWritingData/Char_0_3754_Writer_1001_1032_Isolated.txt')

# datapath_prefix = '/DataA/Harric/ChineseCharacterExp/CASIA_Dataset/HandWritingData_240Binarized/CASIA-HWDB2.1/'
# checking_file_list.append('../FileList/HandWritingData/Char_0_3754_Writer_1151_1200_Cursive.txt')
# checking_file_list.append('../FileList/HandWritingData/Char_0_3754_Writer_1101_1150_Cursive.txt')
# checking_file_list.append('../FileList/HandWritingData/Char_0_3754_Writer_1001_1032_Cursive.txt')

datapath_prefix = '/DataA/Harric/ChineseCharacterExp/CASIA_Dataset/PrintedData/GB2312_L1/'
# checking_file_list.append('../FileList/PrintedData/Char_0_3754_Font_0_49_GB2312L1.txt')
checking_file_list.append('../FileList/PrintedData/Char_0_3754_Font_50_79_GB2312L1.txt')
# checking_file_list.append('../FileList/PrintedData/Char_0_3754_Writer_Selected32_Printed_Fonts_GB2312L1.txt')





# file_written_file_name = 'tmp.txt'
# file_written_file_name = 'ContentChar_BlancaPython_Writer_1151_1200_Isolated.txt'
# file_written_file_name = 'ContentChar_BlancaPython_Writer_1101_1150_Isolated.txt'
# file_written_file_name = 'ContentChar_BlancaPython_Writer_1001_1032_Isolated.txt'
# file_written_file_name = 'ContentChar_BlancaPython_Writer_1151_1200_Cursive.txt'
# file_written_file_name = 'ContentChar_BlancaPython_Writer_1101_1150_Cursive.txt'
# file_written_file_name = 'ContentChar_BlancaPython_Writer_1001_1032_Cursive.txt'

# file_written_file_name = 'StyleChars_Paintings_Writer_1151_1200_Isolated.txt'
# file_written_file_name = 'StyleChars_Paintings_Writer_1101_1150_Isolated.txt'
# file_written_file_name = 'StyleChars_Paintings_Writer_1001_1032_Isolated.txt'
# file_written_file_name = 'StyleChars_Paintings_Writer_1151_1200_Cursive.txt'
# file_written_file_name = 'StyleChars_Paintings_Writer_1101_1150_Cursive.txt'
# file_written_file_name = 'StyleChars_Paintings_Writer_1001_1032_Cursive.txt'



# file_written_file_name = 'ContentChar_BlancaPython_Font_0_49_GB2312L1.txt'
file_written_file_name = 'ContentChar_BlancaPython_Font_50_79_GB2312L1.txt'
# file_written_file_name = 'ContentChar_BlancaPython_Selected32_Printed_Fonts_GB2312L1.txt'
# file_written_file_name = 'StyleChar_Paintings_Font_0_49_GB2312L1.txt'
# file_written_file_name = 'StyleChar_Paintings_Font_50_79_GB2312L1.txt'
# file_written_file_name = 'StyleChar_Paintings_Writer_Selected32_Printed_Fonts_GB2312L1.txt'




if not os.path.exists(file_written_dir):
    os.makedirs(file_written_dir)


def read_content_from_dir(input_txt,
                          level1_charlist,level2_charlist,
                          level1_labellist,level2_labellist):
    # get label0 for the targeted content input txt
    targeted_chars_list = list()
    targeted_character_label0_list = list()

    train_counter=0
    test_counter=0
    l1_counter=0
    l2_counter=0
    train_chars=list()
    test_chars=list()

    with open(input_txt) as f:
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
                        l1_counter+=1
                        if idx<3000:
                            train_counter+=1
                            train_chars.append(char)
                        else:
                            test_counter+=1
                            test_chars.append(char)
                    elif level2_found == 2:
                        idx = level2_charlist.index(current_char)
                        character_id = level2_labellist[idx]
                        l2_counter+=1
                    else:
                        print("Fails! Didnt find %s" % unicode(char))
                        character_id = 0
                        return -1,-1

                    targeted_character_label0_list.append(str(character_id))
                    targeted_chars_list.append(current_char)
    print("In total %d targeted chars are found in the standard GB2312 with Repeat:%d" %
          (len(targeted_chars_list),
           len(targeted_chars_list)-len(np.unique(targeted_character_label0_list))))
    print("TrainingNum:%d, TestNum:%d, GB2312L1:%d, GB2312L2:%d" %
          (train_counter,test_counter,l1_counter,l2_counter))

    return targeted_character_label0_list, targeted_chars_list



def data_file_list_read(file_list_txt):
    label0_list = list()
    label1_list = list()
    data_list = list()

    for ii in range(len(file_list_txt)):
        file_handle = open(file_list_txt[ii], 'r')
        lines = file_handle.readlines()

        for line in lines:
            curt_line = line.split('@')
            label1_list.append(curt_line[2])
            label0_list.append(curt_line[1])
            curt_data = curt_line[3].split('\n')[0]
            if curt_data[0] == '/':
                curt_data = curt_data[1:]
            # curt_data_path = os.path.join(file_data_dir, curt_data)
            data_list.append(curt_data)
        file_handle.close()

    return label1_list, label0_list, data_list


def find_related_data_list(dataset_label1_list,
                           dataset_label0_list,
                           dataset_data_list,
                           label0_list_to_be_checked,
                           char_list,
                           lost_data_complete,lost_data_complete_font_file):
    def _diff(list1,list2):
        c = set(list1).union(set(list2))
        d = set(list1).intersection(set(list2))
        output = list(c-d)
        return output

    dataset_label1_vec=np.unique(dataset_label1_list)
    found_label0_list=list()
    found_label1_list=list()
    found_data_list=list()
    sample_font = ImageFont.truetype(lost_data_complete_font_file, size=150)


    tmpfile_counter=0
    invalid_dataset=False
    tmpchar_save_dir = os.path.join(datapath_prefix, 'TmpChars')
    if not os.path.exists(tmpchar_save_dir):
        os.makedirs(tmpchar_save_dir)


    for label1 in dataset_label1_vec:
        current_found_label0_list=list()
        current_found_label1_list=list()
        current_found_data_list=list()
        found_indices = [ii for ii in range(len(dataset_label1_list)) if dataset_label1_list[ii] == label1]

        current_possible_label0_list = list()
        current_possible_label1_list = list()
        current_possible_data_list = list()
        for ii in found_indices:
            current_possible_label0_list.append(dataset_label0_list[ii])
            current_possible_label1_list.append(dataset_label1_list[ii])
            current_possible_data_list.append(dataset_data_list[ii])

        label0_travel_counter = 0
        for current_checking_label0 in label0_list_to_be_checked:
            if current_checking_label0 in current_possible_label0_list:
                found_indices = [ii for ii in range(len(current_possible_label0_list)) if current_possible_label0_list[ii] == current_checking_label0]
                if len(found_indices)==0:
                    break
                for ii in found_indices:
                    current_found_label0_list.append(current_possible_label0_list[ii])
                    current_found_label1_list.append(current_possible_label1_list[ii])
                    current_found_data_list.append(current_possible_data_list[ii])
            else:
                lost_char = char_list[label0_travel_counter]

                if not lost_data_complete:
                    print("Error: Character:%s(%s) not found in Style:%s"
                          % (current_checking_label0, lost_char, label1))
                    invalid_dataset=True
                else:
                    print("Error: Character:%s(%s) not found in Style:%s, But saved a replacement:%d"
                          % (current_checking_label0, lost_char, label1, tmpfile_counter+1))
                    tmp_single_char_completed = draw_single_char(ch=lost_char,
                                                                 font=sample_font,
                                                                 canvas_size=64)

                    tmp_file_name = "%09d_%s_%s.png"  % (tmpfile_counter, current_checking_label0, label1)
                    misc.imsave(os.path.join(tmpchar_save_dir, tmp_file_name),
                                tmp_single_char_completed)

                    current_found_data_list.append('TmpChars/'+tmp_file_name)
                    current_found_label1_list.append(label1)
                    current_found_label0_list.append(current_checking_label0)

                    tmpfile_counter+=1

            label0_travel_counter+=1

        found_label0_list.extend(current_found_label0_list)
        found_label1_list.extend(current_found_label1_list)
        found_data_list.extend(current_found_data_list)

    print("In Total Styles:%d, Lack Char Style:%d;" % (len(np.unique(dataset_label1_list)),len(np.unique(dataset_label1_list))-len(np.unique(found_label1_list))))
    if not invalid_dataset:
        return found_label0_list, found_label1_list, found_data_list
    else:
        return -1,-1,-1


def write_to_file(path,data_list,label0_list,label1_list,mark):

    file_handle = open(path,'w')
    full_line_num = len(data_list)
    for ii in range(full_line_num):
        if mark:
            write_info = str(1) +'@'+ str(label0_list[ii]) + '@' + str(label1_list[ii]) + '@' + data_list[ii]
        else:
            write_info = str(-1) + '@'+ str(label0_list[ii]) + '@' + str(label1_list[ii]) + '@' + data_list[ii]
        file_handle.write(write_info)
        file_handle.write('\n')
    file_handle.close()
    print("Write to File: %s" % path)
    print("FileNum:%d" % len(data_list))

charset_level1, character_label_level1 = \
    inf_tools.get_chars_set_from_level1_2(path='../ContentTxt/GB2312_Level_1.txt',
                                          level=1)
charset_level2, character_label_level2 = \
    inf_tools.get_chars_set_from_level1_2(path='../ContentTxt/GB2312_Level_2.txt',
                                          level=2)

input_label0_list,input_char_list= \
    read_content_from_dir(input_txt=input_txt,
                          level1_charlist=charset_level1,
                          level2_charlist=charset_level2,
                          level1_labellist=character_label_level1,
                          level2_labellist=character_label_level2)

dataset_label1_list, dataset_label0_list, dataset_data_list = \
    data_file_list_read(file_list_txt=checking_file_list)

found_label0_list, found_label1_list, found_data_list = \
    find_related_data_list(dataset_label1_list=dataset_label1_list,
                           dataset_label0_list=dataset_label0_list,
                           dataset_data_list=dataset_data_list,
                           label0_list_to_be_checked=input_label0_list,
                           char_list = input_char_list,
                           lost_data_complete=lost_data_complete,
                           lost_data_complete_font_file=lost_data_complete_font_file)

write_to_file(path=os.path.join(file_written_dir,file_written_file_name),
              data_list=found_data_list,
              label0_list=found_label0_list,
              label1_list=found_label1_list,
              mark=True)


