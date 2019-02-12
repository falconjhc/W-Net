import os
import numpy as np

file_list_txt = '../FileList/HandWritingData/Char_0_3754_Writer_1151_1200_Cursive.txt'
file_written_dir = '../TrainTestFileList/HandWritingData/'
file_written_file_name = 'Char_0_3754_Writer_1151_1200_Cursive'

if not os.path.exists(file_written_dir):
    os.makedirs(file_written_dir)


label0_split_train_test_label = 3000

def data_file_list_read(file_list_txt):
    label0_list = list()
    label1_list = list()
    data_list = list()

    file_handle = open(file_list_txt, 'r')
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

def train_test_split(label0_split_train_test_label,
                     label0_list,label1_list,data_list):
    train_label0_list=list()
    train_label1_list=list()
    train_data_list=list()
    test_label0_list = list()
    test_label1_list = list()
    test_data_list = list()

    for ii in range(len(data_list)):
        current_label0 = label0_list[ii]
        label0_1 = current_label0[0:3]
        label0_2 = current_label0[3:6]
        label0_id = (int(label0_1)-160-16)*94 + (int(label0_2)-160-1)
        current_label1 = label1_list[ii]
        current_data = data_list[ii]
        if label0_id<label0_split_train_test_label:
            train_label0_list.append(current_label0)
            train_label1_list.append(current_label1)
            train_data_list.append(current_data)
        else:
            test_label0_list.append(current_label0)
            test_label1_list.append(current_label1)
            test_data_list.append(current_data)
        # if label0_id>3754:
        #     a=1
    return train_label0_list,train_label1_list,train_data_list,test_label0_list,test_label1_list,test_data_list


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


label1_list, label0_list, data_list = \
    data_file_list_read(file_list_txt=file_list_txt)

train_label0_list,\
train_label1_list,\
train_data_list,\
test_label0_list,\
test_label1_list,\
test_data_list = \
    train_test_split(label0_split_train_test_label=label0_split_train_test_label,
                     label0_list=label0_list,
                     label1_list=label1_list,
                     data_list=data_list)

print("TrainFileNum:%d, TestFileNum:%d" % (len(train_data_list), len(test_data_list)))
print("TrainLabel0:%d, TestLabel0:%d" % (len(np.unique(train_label0_list)),len(np.unique(test_label0_list))))
print("TrainLabel1:%d, TestLabel1:%d" % (len(np.unique(train_label1_list)),len(np.unique(test_label1_list))))


write_to_file(path=os.path.join(file_written_dir,file_written_file_name+'_Train.txt'),
              data_list=train_data_list,
              label0_list=train_label0_list,
              label1_list=train_label1_list,
              mark=True)

write_to_file(path=os.path.join(file_written_dir,file_written_file_name+'_Test.txt'),
              data_list=test_data_list,
              label0_list=test_label0_list,
              label1_list=test_label1_list,
              mark=True)

