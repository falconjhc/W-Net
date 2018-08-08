import numpy as np
import random
import tensorflow as tf
import os

import sys
import gc
sys.path.append('..')
from utilities.utils import image_show

from utilities.utils import shift_and_resize_image
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random as rnd
import scipy.misc as misc
import time
import multiprocessing as multi_thread
print_separator = "#################################################################"
from tensorflow.python.client import device_lib
import copy as cpy

GRAYSCALE_AVG = 127.5
class Dataset(object):
    def __init__(self,
                 data_list,
                 label0_list,label1_list,
                 sorted_by_label0,
                 print_marks,
                 info_print_interval):
        self.data_list = data_list
        self.label0_list = label0_list
        self.label1_list = label1_list

        if sorted_by_label0:
            self.sorted_data_by_label0(print_marks=print_marks,
                                       info_print_interval=info_print_interval)



        
    def sorted_data_by_label0(self,print_marks,info_print_interval):
        print(print_separator)
        label0_vec = np.unique(self.label0_list)
        sorted_data_list=list()
        sorted_label0_list = list()
        sorted_label1_list = list()



        sort_start = time.time()
        counter=0
        for label0 in label0_vec:
            found_indices = [ii for ii in range(len(self.label0_list)) if self.label0_list[ii] == label0]
            for ii in found_indices:
                sorted_data_list.append(self.data_list[ii])
                sorted_label0_list.append(self.label0_list[ii])
                sorted_label1_list.append(self.label1_list[ii])

            if time.time()-sort_start > info_print_interval or label0==label0_vec[0] or counter == len(label0_vec)-1:
                print(print_marks+'SortingForLabel0:%d/%d' % (counter+1,len(label0_vec)))
                sort_start=time.time()
            counter+=1
        self.data_list = sorted_data_list
        self.label0_list = sorted_label0_list
        self.label1_list = sorted_label1_list
        print(print_separator)




    def read_style_file_list(self,data_path_list_txt_file,data_dir):

        self.label0_list = list()
        self.label1_list = list()
        self.data_list = list()

        for ii in range(len(data_path_list_txt_file)):

            file_handle = open(data_path_list_txt_file[ii], 'r')
            lines = file_handle.readlines()

            for line in lines:
                curt_line = line.split('@')
                self.label0_list.append(int(curt_line[1]))
                self.label1_list.append(int(curt_line[2]))
                curt_data = curt_line[3].split('\n')[0]
                if curt_data[0] == '/':
                    curt_data = curt_data[1:]
                curt_data_path = os.path.join(data_dir[ii], curt_data)
                self.data_list.append(curt_data_path)

            file_handle.close()


class Dataset_Iterator(object):
    def __init__(self,thread_num,
                 batch_size,style_input_num,
                 input_width, input_channel,
                 true_style,
                 style_reference_list,content_prototype_list,
                 info_print_interval,print_marks,
                 augment=False
                 ):
        self.batch_size = batch_size
        self.input_width = input_width
        self.input_filters = input_channel
        self.true_style = true_style
        self.style_reference_list = style_reference_list

        self.thread_num = thread_num
        self.augment = augment
        self.style_input_num = style_input_num
        self.content_input_num = len(content_prototype_list)
        self.label0_vec = np.unique(self.true_style.label0_list)

        self.content_data_list_alignment_with_true_style_data(content_prototype_list=content_prototype_list,
                                                              print_marks=print_marks,
                                                              info_print_interval=info_print_interval)


    def content_data_list_alignment_with_true_style_data(self, content_prototype_list, print_marks,info_print_interval):

        find_start = time.time()
        label0_counter = 0
        content_data_list = list()
        content_label0_list = list()
        content_label1_list = list()
        for ii in range(len(content_prototype_list)):
            content_data_list.append(cpy.deepcopy(self.true_style.data_list))
            content_label0_list.append(cpy.deepcopy(self.true_style.label0_list))
            content_label1_list.append(cpy.deepcopy(self.true_style.label1_list))

        label0_vec = np.unique(self.true_style.label0_list)
        delete_label_add = 0
        delete_data_add = 0
        for label0 in label0_vec:
            current_label0_indices_on_the_style_data = [ii for ii in range(len(self.true_style.label0_list)) if self.true_style.label0_list[ii] == label0]

            valid_label0 = True
            for kk in range(len(content_prototype_list)):
                if not label0 in content_prototype_list[kk].label0_list:
                    valid_label0 = False
                    break
            if not valid_label0:
                delete_label_add += 1
                delete_data_add += len(current_label0_indices_on_the_style_data)
                delete_counter = 0
                current_label0_indices_on_the_style_data.sort()
                for iii in current_label0_indices_on_the_style_data:
                    del self.true_style.label0_list[iii-delete_counter]
                    del self.true_style.label1_list[iii-delete_counter]
                    del self.true_style.data_list[iii-delete_counter]
                    for jjj in range(len(self.style_reference_list)):
                        del self.style_reference_list[jjj].label0_list[iii-delete_counter]
                        del self.style_reference_list[jjj].label1_list[iii-delete_counter]
                        del self.style_reference_list[jjj].data_list[iii-delete_counter]
                    for jjj in range(len(content_prototype_list)):
                        del content_data_list[jjj][iii - delete_counter]
                        del content_label0_list[jjj][iii - delete_counter]
                        del content_label1_list[jjj][iii - delete_counter]
                    delete_counter += 1

                continue
            else:
                current_label0_indices_on_the_style_data = [ii for ii in range(len(self.true_style.label0_list)) if self.true_style.label0_list[ii] == label0]
                for ii in range(len(content_prototype_list)):
                    current_prototype_dataset = content_prototype_list[ii]
                    current_label0_index_on_the_content_prototype_data = current_prototype_dataset.label0_list.index(label0)
                    for jj in current_label0_indices_on_the_style_data:
                        content_data_list[ii][jj] = current_prototype_dataset.data_list[current_label0_index_on_the_content_prototype_data]
                        content_label0_list[ii][jj] = current_prototype_dataset.label0_list[current_label0_index_on_the_content_prototype_data]
                        content_label1_list[ii][jj] = current_prototype_dataset.label1_list[current_label0_index_on_the_content_prototype_data]


                if time.time() - find_start > info_print_interval or label0 == label0_vec[0] or label0_counter == len(label0_vec) - 1:
                    print(print_marks + ' FindingCorrespondendingContentPrototype_BasedOnLabel0:%d/%d' %
                          (label0_counter + 1, len(label0_vec)))
                    print(print_marks + ' Deleted %d Label0s with %d samples' %
                          (delete_label_add, delete_data_add))
                    find_start = time.time()
            label0_counter += 1


        self.content_prototype_list = content_prototype_list
        for ii in range(len(content_prototype_list)):
            self.content_prototype_list[ii].data_list = content_data_list[ii]
            self.content_prototype_list[ii].label0_list = content_label0_list[ii]
            self.content_prototype_list[ii].label1_list = content_label1_list[ii]



    def reproduce_dataset_lists(self, info, shuffle,info_print_interval):

        if shuffle:

            old_content_prototype_list = cpy.deepcopy(self.content_prototype_list)
            old_true_style_data_list = self.true_style.data_list
            old_true_style_label0_list = self.true_style.label0_list
            old_true_style_label1_list = self.true_style.label1_list

            self.true_style.data_list=list()
            self.true_style.label0_list=list()
            self.true_style.label1_list=list()
            for ii in range(self.content_input_num):
                self.content_prototype_list[ii].data_list = list()
                self.content_prototype_list[ii].label0_list = list()
                self.content_prototype_list[ii].label1_list = list()

            indices_shuffled = np.random.permutation(len(old_true_style_data_list))
            for ii in indices_shuffled:
                self.true_style.data_list.append(old_true_style_data_list[ii])
                self.true_style.label0_list.append(old_true_style_label0_list[ii])
                self.true_style.label1_list.append(old_true_style_label1_list[ii])
                for jj in range(self.content_input_num):
                    self.content_prototype_list[jj].data_list.append(old_content_prototype_list[jj].data_list[ii])
                    self.content_prototype_list[jj].label0_list.append(old_content_prototype_list[jj].label0_list[ii])
                    self.content_prototype_list[jj].label1_list.append(old_content_prototype_list[jj].label1_list[ii])



        time_start = time.time()
        label1_vec = np.unique(self.true_style.label1_list)
        label1_counter = 0
        for label1 in label1_vec:
            found_indices = [ii for ii in range(len(self.true_style.label1_list)) if
                             self.true_style.label1_list[ii] == label1]
            for jj in range(self.style_input_num):
                current_new_indices = rnd.sample(found_indices, len(found_indices))
                for kk in range(len(found_indices)):
                    self.style_reference_list[jj].data_list[found_indices[kk]] = \
                        self.true_style.data_list[current_new_indices[kk]]
                    self.style_reference_list[jj].label0_list[found_indices[kk]] = \
                        self.true_style.label0_list[current_new_indices[kk]]
                    self.style_reference_list[jj].label1_list[found_indices[kk]] = \
                        self.true_style.label1_list[current_new_indices[kk]]
            label1_counter += 1
            if time.time() - time_start > info_print_interval or label1_counter == len(
                    label1_vec) or label1_counter == 1:
                time_start = time.time()
                print('%s:DatasetReInitialization@CurrentLabel1:%d(%d)/%d' % (
                info, label1_counter, label1, len(label1_vec)))

    def iterator_reset(self,sess):
        sess.run(self.true_style_iterator.initializer,
                 feed_dict={self.true_style_data_list_input_op:self.true_style.data_list,
                            self.true_style_label0_list_input_op:self.true_style.label0_list,
                            self.true_style_label1_list_input_op:self.true_style.label1_list})
        for ii in range(self.content_input_num):
            sess.run(self.prototype_iterator_list[ii].initializer,
                     feed_dict={self.prototype_data_list_input_op_list[ii]:self.content_prototype_list[ii].data_list,
                                self.prototype_label0_list_input_op_list[ii]:self.content_prototype_list[ii].label0_list,
                                self.prototype_label1_list_input_op_list[ii]:self.content_prototype_list[ii].label1_list})
        for ii in range(self.style_input_num):
            sess.run(self.reference_iterator_list[ii].initializer,
                     feed_dict={self.reference_data_list_input_op_list[ii]: self.style_reference_list[ii].data_list,
                                self.reference_label0_list_input_op_list[ii]: self.style_reference_list[ii].label0_list,
                                self.reference_label1_list_input_op_list[ii]: self.style_reference_list[ii].label1_list})


    def create_dataset_op(self):
        def _get_tensor_slice():
            data_list_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
            label0_list_placeholder = tf.placeholder(dtype=tf.float32, shape=[None])
            label1_list_placeholder = tf.placeholder(dtype=tf.float32, shape=[None])
            dataset_op = tf.data.Dataset.from_tensor_slices((data_list_placeholder,
                                                             label0_list_placeholder,
                                                             label1_list_placeholder))
            return dataset_op, \
                   data_list_placeholder, \
                   label0_list_placeholder, label1_list_placeholder



        def _parser_for_data(file_list,label0_list,label1_list):
            image_string = tf.read_file(file_list)
            image_decoded = tf.image.decode_image(contents=image_string, channels=1)
            image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, self.input_width, self.input_width)
            img_output = tf.slice(image_resized,
                                  [0, 0, 0],
                                  [self.input_width, self.input_width, self.input_filters])
            img_output = tf.subtract(tf.divide(tf.cast(img_output, tf.float32), tf.constant(127.5, tf.float32)),
                                     tf.constant(1, tf.float32))
            return img_output, label0_list, label1_list




        # for true style image
        true_style_dataset, \
        true_style_data_list_input_op, \
        true_style_label0_list_input_op, true_style_label1_list_input_op = \
            _get_tensor_slice()

        true_style_dataset = \
            true_style_dataset.map(map_func=_parser_for_data,
                                   num_parallel_calls=self.thread_num).apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat(3)
        true_style_iterator = true_style_dataset.make_initializable_iterator()
        true_style_img_tensor, true_style_label0_tensor, true_style_label1_tensor = \
            true_style_iterator.get_next()
        output_img_tensor = true_style_img_tensor

        self.true_style_iterator = true_style_iterator
        self.true_style_data_list_input_op = true_style_data_list_input_op
        self.true_style_label0_list_input_op = true_style_label0_list_input_op
        self.true_style_label1_list_input_op = true_style_label1_list_input_op


        # for prototype images
        prototype_iterator_list = list()
        prototype_data_list_input_op_list = list()
        prototype_label0_list_input_op_list = list()
        prototype_label1_list_input_op_list = list()
        for ii in range(self.content_input_num):
            prototype_dataset, \
            prototype_data_list_input_op, \
            prototype_label0_list_input_op, prototype_label1_list_input_op = \
                _get_tensor_slice()
            prototype_dataset = \
                prototype_dataset.map(map_func=_parser_for_data,
                                      num_parallel_calls=self.thread_num).apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat(3)
            prototype_iterator = prototype_dataset.make_initializable_iterator()
            prototype_img_tensor, prototype_label0_tensor, prototype_label1_tensor = prototype_iterator.get_next()
            prototype_iterator_list.append(prototype_iterator)
            prototype_data_list_input_op_list.append(prototype_data_list_input_op)
            prototype_label0_list_input_op_list.append(prototype_label0_list_input_op)
            prototype_label1_list_input_op_list.append(prototype_label1_list_input_op)

            output_img_tensor = tf.concat([output_img_tensor,prototype_img_tensor], axis=3)

        self.prototype_iterator_list = prototype_iterator_list
        self.prototype_data_list_input_op_list = prototype_data_list_input_op_list
        self.prototype_label0_list_input_op_list = prototype_label0_list_input_op_list
        self.prototype_label1_list_input_op_list = prototype_label1_list_input_op_list


        # for reference images
        reference_iterator_list = list()
        reference_data_list_input_op_list = list()
        reference_label0_list_input_op_list = list()
        reference_label1_list_input_op_list = list()
        for ii in range(self.style_input_num):
            reference_dataset, \
            reference_data_list_input_op, \
            reference_label0_list_input_op, reference_label1_list_input_op = \
                _get_tensor_slice()

            reference_dataset = reference_dataset.map(map_func=_parser_for_data,
                                                      num_parallel_calls=self.thread_num).apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat(3)
            reference_iterator = reference_dataset.make_initializable_iterator()
            reference_img_tensor, reference_label0_tensor, reference_label1_tensor = reference_iterator.get_next()

            reference_iterator_list.append(reference_iterator)
            reference_data_list_input_op_list.append(reference_data_list_input_op)
            reference_label0_list_input_op_list.append(reference_label0_list_input_op)
            reference_label1_list_input_op_list.append(reference_label1_list_input_op)

            output_img_tensor = tf.concat([output_img_tensor, reference_img_tensor], axis=3)

        self.reference_iterator_list=reference_iterator_list
        self.reference_data_list_input_op_list = reference_data_list_input_op_list
        self.reference_label0_list_input_op_list = reference_label0_list_input_op_list
        self.reference_label1_list_input_op_list = reference_label1_list_input_op_list

        self.output_batch_image_tensor = output_img_tensor
        self.output_batch_label0_tensor = true_style_label0_tensor
        self.output_batch_label1_tensor = true_style_label1_tensor




    def get_next_batch(self, sess):
        img, label0, label1 = sess.run([self.output_batch_image_tensor,
                                        self.output_batch_label0_tensor,
                                        self.output_batch_label1_tensor])
        return img, label1, label0

class DataProvider(object):
    def __init__(self,
                 epoch,
                 batch_size,
                 input_width,
                 input_filters,style_input_num,
                 info_print_interval,
                 file_list_txt_content, file_list_txt_style_train, file_list_txt_style_validation,
                 content_data_dir, style_train_data_dir,style_validation_data_dir,
                 augment_train_data=True):

        local_device_protos = device_lib.list_local_devices()
        gpu_device = [x.name for x in local_device_protos if x.device_type == 'GPU']
        if len(gpu_device) == 0:
            self.thread_num = multi_thread.cpu_count()
        else:
            self.thread_num = int(multi_thread.cpu_count() / len(gpu_device))

        self.batch_size = batch_size
        self.augment_train_data=augment_train_data
        self.input_width = input_width
        self.input_filters = input_filters

        self.style_input_num=style_input_num

        self.dataset_iterator_create(content_data_dir=content_data_dir,
                                     file_list_txt_content=file_list_txt_content,
                                     style_train_data_dir=style_train_data_dir,
                                     file_list_txt_style_train=file_list_txt_style_train,
                                     style_validation_data_dir=style_validation_data_dir,
                                     file_list_txt_style_validation=file_list_txt_style_validation,
                                     info_print_interval=info_print_interval)

    def dataset_reinitialization(self, sess, init_for_val, info_interval):
        self.train_iterator.reproduce_dataset_lists(info="TrainData", shuffle=True, info_print_interval=info_interval)
        self.train_iterator.iterator_reset(sess=sess)
        if init_for_val:
            self.validate_iterator.reproduce_dataset_lists(info="ValData", shuffle=False, info_print_interval=info_interval)
            self.validate_iterator.iterator_reset(sess=sess)
        print(print_separator)

    def data_file_list_read(self,file_list_txt,file_data_dir):

        label0_list = list()
        label1_list = list()
        data_list = list()

        for ii in range(len(file_list_txt)):

            file_handle = open(file_list_txt[ii], 'r')
            lines = file_handle.readlines()

            for line in lines:
                curt_line = line.split('@')
                label1_list.append(int(curt_line[2]))
                label0_list.append(int(curt_line[1]))
                curt_data = curt_line[3].split('\n')[0]
                if curt_data[0] == '/':
                    curt_data = curt_data[1:]
                curt_data_path = os.path.join(file_data_dir[ii], curt_data)
                data_list.append(curt_data_path)
            file_handle.close()
        return label1_list, label0_list, data_list



    def dataset_iterator_create(self,info_print_interval,
                                content_data_dir,file_list_txt_content,
                                style_train_data_dir, file_list_txt_style_train,
                                style_validation_data_dir, file_list_txt_style_validation):

        def _filter_current_label1_data(current_label1, full_data_list, full_label1_list,full_label0_list):
            selected_indices = [ii for ii in range(len(full_label1_list)) if full_label1_list[ii] == current_label1]
            selected_data_list=list()
            selected_label0_list = list()
            selected_label1_list = list()

            for ii in selected_indices:
                selected_data_list.append(full_data_list[ii])
                selected_label0_list.append(full_label0_list[ii])
                selected_label1_list.append(full_label1_list[ii])
            return selected_data_list, selected_label0_list,selected_label1_list

        # building for content data set
        content_label1_list, content_label0_list, content_data_path_list = \
            self.data_file_list_read(file_list_txt=file_list_txt_content,
                                     file_data_dir=content_data_dir)
        self.content_label1_vec = np.unique(content_label1_list)
        self.content_label0_vec = np.unique(content_label0_list)
        self.content_input_num = len(np.unique(content_label1_list))

        content_prototype_list = list()
        for content_label1 in self.content_label1_vec:
            current_data_list, current_label0_list, current_label1_list = \
            _filter_current_label1_data(current_label1=content_label1,
                                        full_data_list=content_data_path_list,
                                        full_label0_list=content_label0_list,
                                        full_label1_list=content_label1_list)
            train_content_dataset = Dataset(data_list=cpy.deepcopy(current_data_list),
                                            label0_list=cpy.deepcopy(current_label0_list),
                                            label1_list=cpy.deepcopy(current_label1_list),
                                            sorted_by_label0=False,
                                            print_marks='ForOriginalContentData:',
                                            info_print_interval=info_print_interval)
            content_prototype_list.append(train_content_dataset)



        # building for style data set for train
        train_style_label1_list, train_style_label0_list, train_style_data_path_list = \
            self.data_file_list_read(file_list_txt=file_list_txt_style_train,
                                     file_data_dir=style_train_data_dir)
        # self.style_label1_vec = np.unique(train_style_label1_list)
        # self.style_label0_vec = np.unique(train_style_label0_list)

        train_style_reference_list = list()
        for ii in range(self.style_input_num):
            train_style_dataset = Dataset(data_list=cpy.deepcopy(train_style_data_path_list),
                                          label0_list=cpy.deepcopy(train_style_label0_list),
                                          label1_list=cpy.deepcopy(train_style_label1_list),
                                          sorted_by_label0=False,
                                          print_marks='ForStyleReferenceTrainData:',
                                          info_print_interval=info_print_interval)
            train_style_reference_list.append(train_style_dataset)

        # building for true style data set for train
        train_true_style_dataset = Dataset(data_list=cpy.deepcopy(train_style_data_path_list),
                                           label0_list=cpy.deepcopy(train_style_label0_list),
                                           label1_list=cpy.deepcopy(train_style_label1_list),
                                           sorted_by_label0=False,
                                           print_marks='ForTrueStyleTrainData:',
                                           info_print_interval=info_print_interval)

        # construct the train iterator
        self.train_iterator = Dataset_Iterator(batch_size=self.batch_size,
                                               thread_num=self.thread_num,
                                               input_width=self.input_width,
                                               input_channel=self.input_filters,
                                               true_style=train_true_style_dataset,
                                               style_reference_list=cpy.deepcopy(train_style_reference_list),
                                               content_prototype_list=cpy.deepcopy(content_prototype_list),
                                               augment=self.augment_train_data,
                                               style_input_num=self.style_input_num,
                                               info_print_interval=info_print_interval,
                                               print_marks='ForTrainIterator:')
        self.style_label1_vec = np.unique(self.train_iterator.true_style.label1_list)
        self.style_label0_vec = self.train_iterator.label0_vec

        # building for style data set for validation
        validation_style_label1_list, validation_style_label0_list, validation_style_data_path_list = \
            self.data_file_list_read(file_list_txt=file_list_txt_style_validation,
                                     file_data_dir=style_validation_data_dir)


        validation_style_reference_list = list()
        for ii in range(self.style_input_num):
            validation_style_dataset = Dataset(data_list=cpy.deepcopy(validation_style_data_path_list),
                                               label0_list=cpy.deepcopy(validation_style_label0_list),
                                               label1_list=cpy.deepcopy(validation_style_label1_list),
                                               sorted_by_label0=False,
                                               print_marks='ForStyleReferenceValidationData:',
                                               info_print_interval=info_print_interval)
            validation_style_reference_list.append(validation_style_dataset)

        # building for true style data set for validation
        validation_true_style_dataset = Dataset(data_list=cpy.deepcopy(validation_style_data_path_list),
                                                label0_list=cpy.deepcopy(validation_style_label0_list),
                                                label1_list=cpy.deepcopy(validation_style_label1_list),
                                                sorted_by_label0=True,
                                                print_marks='ForTrueStyleValidationData:',
                                                info_print_interval=info_print_interval)

        # construct the validation iterator
        self.validate_iterator = Dataset_Iterator(batch_size=self.batch_size,
                                                  thread_num=self.thread_num,
                                                  input_width=self.input_width,
                                                  input_channel=self.input_filters,
                                                  true_style=validation_true_style_dataset,
                                                  style_reference_list=cpy.deepcopy(validation_style_reference_list),
                                                  content_prototype_list=cpy.deepcopy(content_prototype_list),
                                                  augment=False,
                                                  style_input_num=self.style_input_num,
                                                  info_print_interval=info_print_interval,
                                                  print_marks='ForValidationIterator:')
        self.train_iterator.create_dataset_op()
        self.validate_iterator.create_dataset_op()

        print(print_separator)

    def get_involved_label_list(self):
        return self.style_label0_vec,self.style_label1_vec

    def compute_total_batch_num(self):
        """Total padded batch num"""
        return int(np.ceil(len(self.train_iterator.true_style.data_list) / float(self.batch_size)))
