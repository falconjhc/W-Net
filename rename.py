import os
path = '/Data_HDD/Harric/ChineseCharacterExp/CASIA_Dataset/PrintedData_64Fonts/'

def list_all_files(rootdir):
    _files = []
    list = os.listdir(rootdir)
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _files.append(path)
    return _files



# for dir_name in os.listdir(path):
#     if os.path.isdir(os.path.join(path,dir_name))==True:
#
#         id = int(dir_name)
#         new_name = 100+int(id)
#         new_name = '%05d' % new_name
#         os.rename(os.path.join(path,dir_name),os.path.join(path,new_name))
#         print file,'ok'


file_list = list_all_files(path)
for file in file_list:

    split_txt = file.split('_')
    split_len = len(split_txt)

    left_str = ''
    for ii in range(split_len-1):
        if ii ==0:
            left_str = split_txt[ii]
        else:
            left_str = left_str + '_' + split_txt[ii]

    right_str = '.'+split_txt[split_len-1].split('.')[1]
    file_id = split_txt[split_len-1].split('.')[0]
    new_file_id = "%05d" % (int(file_id)+100)
    new_file_name = left_str + '_' + new_file_id+right_str

    os.rename(file, new_file_name)


    print new_file_name,'ok'
