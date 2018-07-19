# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import glob
import os
import cPickle as pickle
import random


def pickle_examples(paths, save_path,train_mark):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory
    """
    with open(save_path, 'wb') as ft:
        for p in paths:
            label = int(os.path.basename(p).split("_")[0])
            with open(p, 'rb') as f:
                if train_mark == True:
                    print("Train: img2bny %s" % p, label)
                else:
                    print("Val: img2bny %s" % p, label)
                img_bytes = f.read()
                r = random.random()
                example = (label, img_bytes)
                pickle.dump(example, ft)


def find_file_list( path,train_mark):
    file_list=list()
    counter=0
    for root, dirs, files in os.walk(path):
        for fn in files:
            if fn.find('.jpg')!=-1:
                this_file_with_path=os.path.join(root,fn)
                if train_mark==True:
                    print('Train File Finding: File_Path:%s, Counter:%d' % (this_file_with_path, counter))
                else:
                    print('Val File Finding: File_Path:%s, Counter:%d' % (this_file_with_path, counter))
                file_list.append(this_file_with_path)
                counter=counter+1



    return file_list


parser = argparse.ArgumentParser(description='Compile list of images into a pickled object for training')
parser.add_argument('--dir_train', dest='dir_train', default='./Font_Pairs/train', help='path of training examples')
parser.add_argument('--dir_val', dest='dir_val', default='./Font_Pairs/val', help='path of validation examples')
parser.add_argument('--save_dir', dest='save_dir', default='./', help='path to save pickled files')




args = parser.parse_args()

if __name__ == "__main__":

    train_name = 'train'+'.obj'
    val_name='val'+'.obj'
    train_path = os.path.join(args.save_dir, train_name)
    val_path = os.path.join(args.save_dir, val_name)

    train_file_list=find_file_list(path=args.dir_train,train_mark=True)
    val_file_list = find_file_list(path=args.dir_val, train_mark=False)



    pickle_examples(val_file_list, save_path=val_path, train_mark=False)
    pickle_examples(train_file_list, save_path=train_path, train_mark=True)

