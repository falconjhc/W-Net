# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import


from collections import namedtuple
import tensorflow as tf
import numpy as np
import os
import shutil
import time

import sys
sys.path.append('..')
from dataset.dataset_classification import DataProvider
from utilities.utils import scale_back_for_img, merge, correct_ckpt_path
from utilities.utils import image_show

from tensorflow.python.client import device_lib




from model.feature_extractor_networks import alexnet as alexnet
from model.feature_extractor_networks import vgg_16_net as vgg16
from model.feature_extractor_networks import vgg_16_net_no_bn as vgg16_nobn
from model.feature_extractor_networks import vgg_11_net as vgg11
from model.feature_extractor_networks import encoder_8_layers as encoder8
from model.feature_extractor_networks import encoder_6_layers as encoder6




network_dict = {'alexnet':alexnet,
                'vgg11net':vgg11,
                'vgg16net': vgg16,
                'vgg16net_nobn': vgg16_nobn,
                'encoder8layers':encoder8,
                'encoder6layers':encoder6}


lr_decay_factor = 0.999
moving_avg_decay = 0.9999


InputHandle = namedtuple("InputHandle", ["batch_images", "batch_label1_labels","batch_label0_labels"])
EvalHandle = namedtuple("InputHandle", ["batch_images", "batch_label1_labels","batch_label0_labels"])
SummaryHandle = namedtuple("SummaryHandle",["CrossEntropy_Loss","TrainingAccuracy","TestAccuracy"])
eps= 1e-3
print_separater="#########################################################"

def get_model_id_and_create_dirs(experiment_id,experiment_dir,
                                 log_dir,
                                 extra_net,
                                 train_resume_mode):
    model_id = ("Exp%s_%s" % (experiment_id,extra_net))



    ckpt_root_dir = os.path.join(experiment_dir,'checkpoint')
    ckpt_root_dir = os.path.join(ckpt_root_dir, model_id)
    ckpt_model_variable_dir = os.path.join(ckpt_root_dir, 'variables')
    ckpt_model_framework_dir = os.path.join(ckpt_root_dir, 'frameworks')

    log_dir=os.path.join(log_dir,model_id)
    if os.path.exists(log_dir) and train_resume_mode !=1:
        shutil.rmtree(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if train_resume_mode !=1:
        if os.path.exists(ckpt_model_variable_dir):
            shutil.rmtree(ckpt_model_variable_dir)
        os.makedirs(ckpt_model_variable_dir)
        if os.path.exists(ckpt_model_framework_dir):
            shutil.rmtree(ckpt_model_framework_dir)
        os.makedirs(ckpt_model_framework_dir)





    return model_id, ckpt_model_variable_dir, ckpt_model_framework_dir, log_dir

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    cpu_device=[x.name for x in local_device_protos if x.device_type == 'CPU']
    gpu_device=[x.name for x in local_device_protos if x.device_type == 'GPU']
    print("Available CPU:%s with number:%d" % (cpu_device, len(cpu_device)))
    print("Available GPU:%s with number:%d" % (gpu_device, len(gpu_device)))
    return cpu_device, gpu_device,len(cpu_device),len(gpu_device)

def train_procedures(args_input):

    weight_decay = True
    initializer = 'XavierInit'


    args_input.label1_loss=args_input.label1_loss+eps
    args_input.label0_loss=args_input.label0_loss+eps
    if args_input.label1_loss<10*eps and args_input.label0_loss<=10*eps:
        print("Error: Both training targets are N/A!!!")
        return

    with tf.Graph().as_default():
        summary_seconds = 30
        print_info_seconds = summary_seconds * 5
        summary_start = time.time()
        print_info_start = time.time()


        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)


        batch_size=args_input.batch_size
        experiment_dir=args_input.experiment_dir
        experiment_id=args_input.experiment_id
        epoch_num=args_input.epoch_num
        extra_net = network_dict[args_input.network]

        data_provider = DataProvider(batch_size=batch_size,
                                     image_width=args_input.image_size,
                                     data_dir_train_path=args_input.data_dir_train_path,
                                     data_dir_validation_path=args_input.data_dir_validation_path,
                                     epoch_num=epoch_num,
                                     input_filters=args_input.image_filters,
                                     file_list_txt_path_train=args_input.file_list_txt_train,
                                     file_list_txt_path_validation=args_input.file_list_txt_validation,
                                     cheat_mode=args_input.cheat_mode, sess=sess)

        cpu, gpu, cpu_num, gpu_num = get_available_gpus()
        if gpu_num==0:
            run_device=cpu[0]
        else:
            run_device=gpu[0]
        print("RunningOn:%s" % run_device)
        print(print_separater)

        with tf.device(run_device):

            model_id, \
            ckpt_variables_dir, ckpt_framework_dir,\
            log_dir = get_model_id_and_create_dirs(experiment_id=experiment_id,
                                                   experiment_dir=experiment_dir,
                                                   log_dir=args_input.log_dir,
                                                   extra_net=args_input.network,
                                                   train_resume_mode=args_input.train_resume_mode)
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False,dtype=tf.int64)
            epoch_step = tf.get_variable('epoch_step', [], initializer=tf.constant_initializer(0), trainable=False,dtype=tf.int64)
            epoch_step_increase_one_op = tf.assign(epoch_step, epoch_step + 1)
            framework_var_list = list()
            framework_var_list.append(global_step)
            framework_var_list.append(epoch_step)

            init_lr = args_input.init_lr
            learning_rate = tf.train.exponential_decay(learning_rate=init_lr,
                                                       global_step=global_step,
                                                       decay_steps=data_provider.iters_for_each_epoch,
                                                       decay_rate=lr_decay_factor,
                                                       staircase=True)
            learning_rate_summary = tf.summary.scalar('LearningRate', learning_rate)

            with tf.variable_scope(tf.get_variable_scope()):
                with tf.device(run_device):


                    merged_loss_summary, loss_optimization,\
                    label0_loss_ce,label1_loss_ce,label0_loss_ct,label1_loss_ct,\
                        input_handle,center_update_op, center_vars\
                        = build_model(batch_size=batch_size,
                                      learning_rate=learning_rate,
                                      input_filters=args_input.image_filters,
                                      logits_length_label1=len(data_provider.label1_vec),
                                      logits_length_label0=len(data_provider.label0_vec),
                                      device=run_device,
                                      extra_net=extra_net,
                                      args_input=args_input,
                                      weight_decay=weight_decay,
                                      initializer=initializer)

                    train_acry_summary, test_acry_summary, \
                    train_enpy_summary, test_enpy_summary, \
                    acry_0_lgt, acry_1_lgt,\
                    label0_acry,label1_acry,label0_enpy,label1_enpy,\
                        eval_handle= \
                        network_inference(batch_size=batch_size,
                                          logits_length_label1=len(data_provider.label1_vec),
                                          logits_length_label0=len(data_provider.label0_vec),
                                          device=run_device,
                                          image_width=args_input.image_size,
                                          extra_net=extra_net,
                                          input_filters=args_input.image_filters,
                                          initializer=initializer)



                    print(
                        "Initialization model building for %s completed;"% (run_device))

            #####################################################################
            #####################################################################
            ############# VARIABLE INITIALIZATION MUST BE RUN ###################
            ############# BEFORE MODEL RESTORE !!!!!!!!!!!!!! ###################
            ############# FXCK TENSORFLOW GRAPH CODING !!!!!! ###################
            #####################################################################
            #####################################################################
            t_vars_for_train = tf.trainable_variables()
            t_vars_for_save = find_bn_avg_var(t_vars_for_train)
            t_vars_for_save.extend(center_vars)
            framework_saver = tf.train.Saver(max_to_keep=1, var_list=framework_var_list)
            saver_full_model = tf.train.Saver(max_to_keep=1, var_list=t_vars_for_save)
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

            # initialize variables that are trained from stratch
            # (rather than those variables loaded)
            if not args_input.train_resume_mode == 1:
                tf.variables_initializer(var_list=t_vars_for_save).run(session=sess)
                tf.variables_initializer(var_list=framework_var_list).run(session=sess)

            else:
                ckpt = tf.train.get_checkpoint_state(ckpt_variables_dir)
                saver_full_model.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
                print('Full model restored from %s' % ckpt.model_checkpoint_path)

                ckpt = tf.train.get_checkpoint_state(ckpt_framework_dir)
                corrected_ckpt_path = correct_ckpt_path(real_dir=ckpt_framework_dir,
                                                        maybe_path=ckpt.model_checkpoint_path)
                framework_saver.restore(sess=sess, save_path=corrected_ckpt_path)
                print("Framework restored from:%s" % corrected_ckpt_path)







            batch_train_val_image = tf.placeholder(tf.float32,
                                                   [1,
                                                    batch_size  * args_input.image_size,
                                                    args_input.image_size*2,
                                                    3])
            batch_train_val_image_summary = tf.summary.image('Batch_Train_Image', batch_train_val_image)



            optimizer =  tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=loss_optimization,
                                                                                   var_list=t_vars_for_train,
                                                                                   global_step=global_step)


            print("TrainSampleNum:%d,ValidateSampleNum:%d,Involvedlabel1Num(Train):%d,Involvedlabel0Num(Train):%d" %
                  (len(data_provider.train.data_list),
                   len(data_provider.val.data_list),
                   len(data_provider.label1_vec),
                   len(data_provider.label0_vec)))
            print("BatchSize:%d,  ItrsNum:%d, EpochNum:%d" %
                  (batch_size, data_provider.iters, data_provider.epoch))
            print("CrossEntropyLoss:Label0/1:%.3f/%.3f" %
                  (args_input.label0_loss,args_input.label1_loss))
            print("CenterLoss:Label0/1:%.3f/%.3f" %
                  (args_input.label0_loss*args_input.center_loss_penalty_rate, args_input.label1_loss*args_input.center_loss_penalty_rate))

            ei_start = epoch_step.eval(session=sess)
            global_step_start = global_step.eval(session=sess)
            ei_ranges = range(ei_start, data_provider.epoch, 1)
            learningrate_start_new = learning_rate.eval(session=sess)
            print("Epoch:%d, GlobalStep:%d, LearningRate:%.5f" % (ei_start,global_step_start,learningrate_start_new))

            print("Initialization completed.")
            if not args_input.debug_mode == 0:
                summary_seconds = 3
                print_info_seconds = 3
            process_pause(args_input)



            print("Training start.")
            start_time = time.time()
            print(print_separater)
            print(print_separater)
            print(print_separater)
            tf.train.start_queue_runners(sess=sess)





            label0_hightest_accuracy = -1
            label1_hightest_accuracy = -1
            record_print_info = list()






            for ei in ei_ranges:

                for bid in range(data_provider.iters_for_each_epoch):
                    this_itr_start = time.time()

                    read_data_start = time.time()
                    batch_images_train, batch_label1_labels_train,batch_label0_labels_train\
                        = data_provider.train.get_next_batch(sess=sess,augment=True)







                    batch_label1_labels_train = dense_to_one_hot(input_label=batch_label1_labels_train,
                                                                 batch_size=batch_size,
                                                                 involved_label_list=data_provider.label1_vec)
                    batch_label0_labels_train = dense_to_one_hot(input_label=batch_label0_labels_train,
                                                                 batch_size=batch_size,
                                                                 involved_label_list=data_provider.label0_vec)
                    read_data_consumed = time.time()-read_data_start

                    feed_for_model_train = batch_feed(batch_images=batch_images_train,
                                                      batch_label0=batch_label0_labels_train,
                                                      batch_label1=batch_label1_labels_train,
                                                      handle=input_handle)
                    feed_for_model_infer = batch_feed(batch_images=batch_images_train,
                                                      batch_label0=batch_label0_labels_train,
                                                      batch_label1=batch_label1_labels_train,
                                                      handle=eval_handle)
                    model_feed = feed_for_model_train
                    model_feed.update(feed_for_model_infer)

                    optimizing_start = time.time()
                    _,_, \
                    batch_label1_loss_ce, \
                    batch_label0_loss_ce, \
                    batch_label1_loss_ct, \
                    batch_label0_loss_ct, \
                    train_accuracy_label1, \
                    train_accuracy_label0, \
                    train_entropy_label1, \
                    train_entropy_label0, \
                    merged_loss_summary_output, \
                    learning_rate_summary_output, \
                    accuracy_summary_train_output, \
                    entropy_summary_train_output \
                        = sess.run([optimizer,center_update_op,
                                    label1_loss_ce,
                                    label0_loss_ce,
                                    label1_loss_ct,
                                    label0_loss_ct,
                                    label1_acry,
                                    label0_acry,
                                    label1_enpy,
                                    label0_enpy,
                                    merged_loss_summary,
                                    learning_rate_summary,
                                    train_acry_summary,
                                    train_enpy_summary],
                                   feed_dict=model_feed)
                    optimizing_consumed = time.time()-optimizing_start





                    ## print info for test on validation test
                    if time.time() - summary_start > summary_seconds or global_step.eval(session=sess)==global_step_start+1:
                        summary_start=time.time()

                        ## summary for train
                        summary_writer.add_summary(merged_loss_summary_output, global_step.eval(session=sess))
                        summary_writer.add_summary(learning_rate_summary_output, global_step.eval(session=sess))
                        summary_writer.add_summary(accuracy_summary_train_output, global_step.eval(session=sess))
                        summary_writer.add_summary(entropy_summary_train_output, global_step.eval(session=sess))

                        ## print info for train
                        current_time = time.strftime('%Y-%m-%d@%H:%M:%S', time.localtime())
                        #print("LearningRate:%.7f" % learning_rate.eval(session=sess))
                        passed_full = time.time() - start_time
                        passed_itr = time.time() - this_itr_start
                        print("Time:%s,Epoch:%d/%d,Itr:%d/%d;" %
                              (current_time, ei + 1, data_provider.epoch, bid + 1, data_provider.iters_for_each_epoch))
                        print("ItrDuration:%.2fses,FullDuration:%.2fhrs(%.2fdays);" %
                              (passed_itr, passed_full / 3600, passed_full / (3600 * 24)))

                        percentage_completed = float(global_step.eval(session=sess)) / float(
                            data_provider.epoch * data_provider.iters_for_each_epoch) * 100
                        percentage_to_be_fulfilled = 100 - percentage_completed
                        hrs_estimated_remaining = (float(passed_full) / (
                            percentage_completed + eps)) * percentage_to_be_fulfilled / 3600
                        print("CompletePctg:%.2f,TimeRemainingEstm:%.2fhrs(%.2fdays)" % (
                            percentage_completed, hrs_estimated_remaining,
                            hrs_estimated_remaining / 24))
                        print("ReadData:%f,Optimization:%f" % (read_data_consumed,optimizing_consumed))


                        ## test on validation
                        batch_images_val, batch_label1_labels_val, batch_label0_labels_val = \
                            data_provider.val.get_next_batch(sess=sess,augment=False)
                        batch_label1_labels_val = dense_to_one_hot(input_label=batch_label1_labels_val,
                                                                   batch_size=batch_size,
                                                                   involved_label_list=data_provider.label1_vec)
                        batch_label0_labels_val = dense_to_one_hot(input_label=batch_label0_labels_val,
                                                                   batch_size=batch_size,
                                                                   involved_label_list=data_provider.label0_vec)


                        test_accuracy_label1,test_accuracy_label0,\
                        test_entropy_label1,test_entropy_label0,\
                        accuracy_summary_test_output,entropy_summary_test_output = \
                            sess.run([label1_acry,label0_acry,
                                      label1_enpy,label0_enpy,
                                      test_acry_summary,test_enpy_summary],
                                     feed_dict=batch_feed(batch_images=batch_images_val,
                                                          batch_label0=batch_label0_labels_val,
                                                          batch_label1=batch_label1_labels_val,
                                                          handle=eval_handle))

                        # validation info output
                        summary_writer.add_summary(accuracy_summary_test_output, global_step.eval(session=sess))
                        summary_writer.add_summary(entropy_summary_test_output, global_step.eval(session=sess))

                        print("TrainAccuracy_onlabel1/_onlabel0:%f/%f" % (train_accuracy_label1, train_accuracy_label0))
                        print("TrainEntropy_onlabel1/_onlabel0:%f/%f" % (train_entropy_label1, train_entropy_label0))
                        print("TestAccuracy_onlabel1/_onlabel0:%f/%f" % (test_accuracy_label1,test_accuracy_label0))
                        print("TestEntropy_onlabel1/_onlabel0:%f/%f" % (test_entropy_label1, test_entropy_label0))
                        print(print_separater)
                        print("CrossEntropyLoss:%f/%f" % (batch_label1_loss_ce,batch_label0_loss_ce))
                        print("CenterLoss:%f/%f" % (batch_label1_loss_ct,batch_label0_loss_ct))
                        print(print_separater)





                        # batch train / validation images summary
                        merged_batch_train_img = merge(scale_back_for_img(batch_images_train),[batch_size,1])
                        merged_batch_val_img = merge(scale_back_for_img(batch_images_val), [batch_size, 1])
                        merged_batch_train_val_img = np.concatenate([merged_batch_train_img, merged_batch_val_img], axis=1)
                        batch_train_val_image_summary_output=sess.run(batch_train_val_image_summary,
                                                                      feed_dict={batch_train_val_image:
                                                                                 np.reshape(merged_batch_train_val_img,
                                                                                            (1,
                                                                                             merged_batch_train_val_img.shape[0],
                                                                                             merged_batch_train_val_img.shape[1],
                                                                                             merged_batch_train_val_img.shape[2]))})
                        summary_writer.add_summary(batch_train_val_image_summary_output, global_step.eval(session=sess))
                        summary_writer.flush()


                    if time.time() - print_info_start > print_info_seconds or global_step.eval(session=sess) == global_step_start+1:
                        print_info_start = time.time()
                        for ii in record_print_info:
                            print(ii)
                        print(print_separater)


                label0_hightest_accuracy, label1_hightest_accuracy = \
                    performance_evaluation(sess=sess,
                                           data_provider=data_provider,
                                           batch_size=batch_size,
                                           evalHandle=eval_handle,
                                           batch_label0_logits_op=acry_0_lgt,
                                           batch_label1_logits_op=acry_1_lgt,
                                           ei=epoch_step.eval(session=sess),
                                           label0_highest=label0_hightest_accuracy,
                                           label1_highest=label1_hightest_accuracy,
                                           record=record_print_info,
                                           print_info_second=summary_seconds)

                checkpoint(sess=sess,
                           saver=saver_full_model,
                           model_dir=ckpt_variables_dir,
                           model_name=args_input.network,
                           counter=global_step.eval(session=sess))
                checkpoint(sess=sess,
                           saver=framework_saver,
                           model_dir=ckpt_framework_dir,
                           model_name='framework',
                           counter=global_step.eval(session=sess))


                sess.run(epoch_step_increase_one_op)
                print("Epoch:%d is completed." % (ei+1))



def process_pause(args_input):
    if args_input.debug_mode == 0:
        raw_input("Press Enter to Coninue")
        print(print_separater)


def find_bn_avg_var(var_list):
    var_list_new=list()
    for ii in var_list:
        var_list_new.append(ii)


    all_vars = tf.global_variables()
    bn_var_list = [var for var in var_list if 'bn' in var.name]
    output_avg_var = list()
    for bn_var in bn_var_list:
        if 'gamma' in bn_var.name:
            continue
        bn_var_name = bn_var.name
        variance_name = bn_var_name.replace('beta','moving_variance')
        average_name = bn_var_name.replace('beta','moving_mean')
        variance_var = [var for var in all_vars if variance_name in var.name][0]
        average_var = [var for var in all_vars if average_name in var.name][0]
        output_avg_var.append(variance_var)
        output_avg_var.append(average_var)

    var_list_new.extend(output_avg_var)

    output=list()
    for ii in var_list_new:
        if ii not in output:
            output.append(ii)

    return output




def dense_to_one_hot(input_label,
                     batch_size,
                     involved_label_list):



    # for abnormal label process:
    # for those labels in the test list not occurring in the training
    abnormal_marker_indices =list()
    for ii in range(len(input_label)):
        if not input_label[ii] in involved_label_list:
            abnormal_marker_indices.append(ii)
    data_indices = np.arange(len(input_label))
    data_indices_list = data_indices.tolist()
    for ii in abnormal_marker_indices:
        data_indices_list.remove(data_indices[ii])
    data_indices = np.array(data_indices_list)


    label_length = len(involved_label_list)
    input_label_matrix = np.tile(np.asarray(input_label), [len(involved_label_list), 1])
    fine_tune_martix = np.transpose(np.tile(involved_label_list, [batch_size, 1]))
    diff=input_label_matrix-fine_tune_martix
    find_positions = np.argwhere(np.transpose(diff) == 0)
    input_label_indices=np.transpose(find_positions[:,1:]).tolist()

    output_one_hot_label=np.zeros((len(input_label), label_length),dtype=np.float32)
    output_one_hot_label[data_indices,input_label_indices]=1
    return output_one_hot_label

def network_inference(batch_size,
                      device,
                      input_filters,
                      logits_length_label1,
                      logits_length_label0,
                      image_width,
                      extra_net,
                      initializer):
    batch_images = tf.placeholder(tf.float32,
                                  [batch_size, image_width, image_width, input_filters],
                                  name='batch_image_validation')

    batch_label1_labels = tf.placeholder(tf.float32,
                                         [batch_size, logits_length_label1],
                                         name='batch_label1_validation')

    batch_label0_labels = tf.placeholder(tf.float32,
                                         [batch_size, logits_length_label0],
                                         name='batch_label0_validation')


    batch_label1_logits, batch_label0_logits = \
        extra_net(image=batch_images,
                  batch_size=batch_size,
                  device=device,
                  logits_length_font=logits_length_label1,
                  logits_length_character=logits_length_label0,
                  is_training=False,
                  reuse=True,
                  weight_decay=False,
                  initializer=initializer,
                  name_prefix='ExtraNet')


    correct_prediction_label1 = tf.equal(tf.argmax(batch_label1_logits,axis=1), tf.argmax(batch_label1_labels,axis=1))
    accuracy_label1 = tf.reduce_mean(tf.cast(correct_prediction_label1, tf.float32)) * 100

    correct_prediction_label0 = tf.equal(tf.argmax(batch_label0_logits, axis=1), tf.argmax(batch_label0_labels, axis=1))
    accuracy_label0 = tf.reduce_mean(tf.cast(correct_prediction_label0, tf.float32)) * 100


    entropy_label1 = tf.nn.softmax_cross_entropy_with_logits(logits=batch_label1_logits,
                                                             labels=tf.nn.softmax(batch_label1_logits))
    entropy_label0 = tf.nn.softmax_cross_entropy_with_logits(logits=batch_label0_logits,
                                                             labels=tf.nn.softmax(batch_label0_logits))
    entropy_label1 = tf.reduce_mean(entropy_label1)
    entropy_label0 = tf.reduce_mean(entropy_label0)


    current_eval_handle = EvalHandle(batch_images=batch_images,
                                     batch_label1_labels=batch_label1_labels,
                                     batch_label0_labels=batch_label0_labels)


    acry_train_label0_summary = tf.summary.scalar("Accuracy_Train_Label0", accuracy_label0)
    acry_train_label1_summary = tf.summary.scalar("Accuracy_Train_Label1", accuracy_label1)
    acry_test_label0_summary = tf.summary.scalar("Accuracy_Test_Label0", accuracy_label0)
    acry_test_label1_summary = tf.summary.scalar("Accuracy_Test_Label1", accuracy_label1)

    enpy_train_label0_summary = tf.summary.scalar("Entropy_Train_Label0", entropy_label0)
    enpy_train_label1_summary = tf.summary.scalar("Entropy_Train_Label1", entropy_label1)
    enpy_test_label0_summary = tf.summary.scalar("Entropy_Test_Label0", entropy_label0)
    enpy_test_label1_summary = tf.summary.scalar("Entropy_Test_Label1", entropy_label1)

    merged_acry_train_summary = tf.summary.merge([acry_train_label0_summary, acry_train_label1_summary])
    merged_acry_test_summary = tf.summary.merge([acry_test_label0_summary, acry_test_label1_summary])
    merged_enpy_train_summary = tf.summary.merge([enpy_train_label0_summary,enpy_train_label1_summary])
    merged_enpy_test_summary = tf.summary.merge([enpy_test_label0_summary,enpy_test_label1_summary])

    return merged_acry_train_summary,\
           merged_acry_test_summary, \
           merged_enpy_train_summary, \
           merged_enpy_test_summary,\
           batch_label0_logits,\
           batch_label1_logits,\
           accuracy_label0,\
           accuracy_label1, \
           entropy_label0, \
           entropy_label1, \
           current_eval_handle

def batch_feed(batch_images,
               batch_label0,
               batch_label1,
               handle):
    output_dict = {}
    output_dict.update({handle.batch_images:batch_images})
    output_dict.update({handle.batch_label0_labels:batch_label0})
    output_dict.update({handle.batch_label1_labels:batch_label1})

    return output_dict



def check_class_centralization(sess,
                               data_provider,
                               evalHandle,
                               batch_size,
                               batch_logits_op,
                               train_label1_vec, check_centralization_label1_vec,
                               ei,
                               print_info_secs):
    def find_groups(label_vec,full_label,full_logits):

        logits_with_class = list()
        for ii in label_vec:
            curt_indices = [int(tmp) for tmp, v in enumerate(full_label) if v == ii]
            selected_logits = full_logits[curt_indices,:]
            logits_with_class.append(selected_logits)
        return logits_with_class


    def calculate_inter_intra_class_centralization(input_logit_group_list):
        intra_distance_all = np.zeros([len(input_logit_group_list),1])
        mean_pos_all = np.zeros([len(input_logit_group_list), input_logit_group_list[0].shape[1]])
        counter=0


        # calculate intra_class distance
        for logits in input_logit_group_list:
            mean_pos = np.mean(logits, axis=0)
            mean_pos_repeated = np.tile(np.reshape(mean_pos,[1, mean_pos.shape[0]]), [logits.shape[0],1])
            intra_distance = np.mean(np.sqrt(np.sum(np.square(logits-mean_pos_repeated),axis=1)))
            intra_distance_all[counter] = intra_distance
            mean_pos_all[counter,:] = mean_pos
            counter+=1
        intra_distance_avg = np.mean(intra_distance_all)


        # calculate inter class distance
        inter_distance_all = np.zeros([len(input_logit_group_list), 1])
        for ii in range(len(input_logit_group_list)):
            this_checking_logit = mean_pos_all[ii,:]

            inter_distance_for_this_checking_logit = np.zeros([len(input_logit_group_list)-1, 1])
            counter_for_this_checking_logit=0
            for jj in range(len(input_logit_group_list)):
                this_to_be_checking_logit = mean_pos_all[jj,:]
                if ii == jj:
                    continue

                inter_distance = np.sqrt(np.sum(np.square(this_checking_logit-this_to_be_checking_logit)))
                inter_distance_for_this_checking_logit[counter_for_this_checking_logit]=inter_distance
                counter_for_this_checking_logit+=1
            inter_distance_avg_for_this_checking_logit = np.mean(inter_distance_for_this_checking_logit)
            inter_distance_all[ii]=inter_distance_avg_for_this_checking_logit
        inter_distance_avg=np.mean(inter_distance_all)
        intra_2_inter = intra_distance_avg / inter_distance_avg

        return intra_2_inter


    print_info_start = time.time()
    print(print_separater)
    iter_num = len(data_provider.eval.data_list) / (batch_size) + 1


    full_logits_list = list()
    for ii in batch_logits_op:
        full_logits_list.append(np.zeros([iter_num * batch_size,int(ii.shape[1]) ]))
    full_label1 = np.zeros([iter_num * batch_size])

    counter=0
    for ii in range(iter_num):
        time_start=time.time()
        batch_images, batch_label1, batch_label0 \
            = data_provider.eval.get_next_batch(sess=sess, augment=False)

        batch_logits = sess.run(batch_logits_op,
                                feed_dict={evalHandle.batch_images:batch_images})

        for jj in range(len(batch_logits)):
            full_logits_list[jj][ii * batch_size:(ii + 1) * batch_size,:] = batch_logits[jj]
        full_label1[ii * batch_size:(ii + 1) * batch_size] = batch_label1

        counter+=batch_size

        if time.time()-print_info_start > print_info_secs or ii == 0 or ii ==iter_num-1:
            print_info_start=time.time()
            print("CheckingCentralization:Eval@Epoch:%d,Iter:%d/%d,Elps:%.3f" % (
            ei, ii + 1, iter_num,  time.time() - time_start))
    print(print_separater)

    intra2inter_for_training_list=list()
    intra2inter_for_all_list = list()
    for ii in range(len(full_logits_list)):
        group_logit_list_for_training = find_groups(label_vec=train_label1_vec,
                                                    full_label=full_label1,
                                                    full_logits=full_logits_list[ii])
        intra2inter_for_training = calculate_inter_intra_class_centralization(input_logit_group_list=group_logit_list_for_training)
        print('ForTraining@Layer%d,intra2inter:%.5f' % (ii, intra2inter_for_training))
        intra2inter_for_training_list.append(intra2inter_for_training)
    print(print_separater)

    for ii in range(len(full_logits_list)):
        group_logit_list_for_centralization = find_groups(label_vec=check_centralization_label1_vec,
                                                          full_label=full_label1,
                                                          full_logits=full_logits_list[ii])
        intra2inter_for_checking_centralization = calculate_inter_intra_class_centralization(input_logit_group_list=group_logit_list_for_centralization)
        print('ForCheckingCentralization@Layer%d,intra2inter:%.5f' % (ii, intra2inter_for_checking_centralization))
        intra2inter_for_all_list.append(intra2inter_for_checking_centralization)
    print(print_separater)


    return intra2inter_for_training_list, intra2inter_for_all_list







def performance_evaluation(sess,
                           data_provider,
                           batch_size,evalHandle,
                           batch_label0_logits_op,batch_label1_logits_op,
                           ei,
                           label0_highest,label1_highest,record,
                           print_info_second):

    print_info_start = time.time()
    print(print_separater)
    iter_num = len(data_provider.val.data_list) / (batch_size) + 1

    full_logits_label0 = np.zeros([iter_num * batch_size, len(data_provider.label0_vec)])
    full_logits_label1 = np.zeros([iter_num * batch_size, len(data_provider.label1_vec)])
    true_label0 = np.zeros([iter_num * batch_size])
    true_label1 = np.zeros([iter_num * batch_size])

    counter=0
    for ii in range(iter_num):
        time_start=time.time()
        batch_images, batch_label1, batch_label0 \
            = data_provider.val.get_next_batch(sess=sess, augment=False)
        batch_label1_labels_one_hot = dense_to_one_hot(input_label=batch_label1,
                                                       batch_size=batch_size,
                                                       involved_label_list=data_provider.label1_vec)

        batch_label0_labels_one_hot = dense_to_one_hot(input_label=batch_label0,
                                                       batch_size=batch_size,
                                                       involved_label_list=data_provider.label0_vec)

        batch_label0_logits, batch_label1_logits, = sess.run([batch_label0_logits_op,batch_label1_logits_op],
                                                             feed_dict=batch_feed(batch_images=batch_images,
                                                                                  batch_label0=batch_label0_labels_one_hot,
                                                                                  batch_label1=batch_label1_labels_one_hot,
                                                                                  handle=evalHandle))

        full_logits_label0[ii * batch_size:(ii + 1) * batch_size, :] = batch_label0_logits
        full_logits_label1[ii * batch_size:(ii + 1) * batch_size, :] = batch_label1_logits
        true_label0[ii * batch_size:(ii + 1) * batch_size] = batch_label0
        true_label1[ii * batch_size:(ii + 1) * batch_size] = batch_label1
        counter=counter+batch_size


        if time.time()-print_info_start > print_info_second or ii == 0 or ii ==iter_num-1:
            print_info_start=time.time()

            batch_estm_indices_label0 = np.argmax(full_logits_label0[0:counter,:], axis=1)
            batch_estm_label0 = data_provider.label0_vec[batch_estm_indices_label0]
            batch_diff_label0 = true_label0[0:counter] - batch_estm_label0
            batch_correct_label0 = [i for i, v in enumerate(batch_diff_label0) if v == 0]
            batch_arcy_label0 = float(len(batch_correct_label0)) / float(counter) * 100

            batch_estm_indices_label1 = np.argmax(full_logits_label1[0:counter,:], axis=1)
            batch_estm_label1 = data_provider.label1_vec[batch_estm_indices_label1]
            batch_diff_label1 = true_label1[0:counter] - batch_estm_label1
            batch_correct_label0 = [i for i, v in enumerate(batch_diff_label1) if v == 0]
            batch_arcy_label1 = float(len(batch_correct_label0)) / float(counter) * 100

            print("Eval@Epoch:%d,Iter:%d/%d,Acry:%.3f/%.3f,Elps:%.3f" % (ei,ii+1,iter_num,batch_arcy_label0,batch_arcy_label1,
                                                                     time.time()-time_start))


    full_logits_label0 = full_logits_label0[0:len(data_provider.val.data_list), :]
    full_logits_label1 = full_logits_label1[0:len(data_provider.val.data_list), :]
    true_label0 = true_label0[0:len(data_provider.val.data_list)]
    true_label1 = true_label1[0:len(data_provider.val.data_list)]

    label0_estm_indices = np.argmax(full_logits_label0, axis=1)
    label1_estm_indices = np.argmax(full_logits_label1, axis=1)
    label0_estm = data_provider.label0_vec[label0_estm_indices]
    label1_estm = data_provider.label1_vec[label1_estm_indices]
    diff_label0 = label0_estm - true_label0
    diff_label1 = label1_estm - true_label1
    correct_label0 = [i for i, v in enumerate(diff_label0) if v == 0]
    correct_label1 = [i for i, v in enumerate(diff_label1) if v == 0]
    acry_label0 = float(len(correct_label0)) / float(len(data_provider.val.data_list)) * 100
    acry_label1 = float(len(correct_label1)) / float(len(data_provider.val.data_list)) * 100

    print("Eval@Epoch:%d, Acry_Label0/Label1:%.3f/%.3f" % (ei,acry_label0,acry_label1))
    print(print_separater)

    if acry_label0>label0_highest or acry_label1 > label1_highest:
        if acry_label0>label0_highest:
            label0_highest = acry_label0

        if acry_label1>label1_highest:
            label1_highest = acry_label1

        current_accuracy_info = time.strftime('%Y-%m-%d@%H:%M:%S', time.localtime())
        current_accuracy_info = current_accuracy_info + (", Epoch:%d, Round:%d, CurtHighestAcry_Label0/1:%.3f/%.3f" %
                                                         (ei,
                                                          len(record) + 1,
                                                          acry_label0,
                                                          acry_label1))
        print("New record found: %s, and model saved" % current_accuracy_info)
        print(print_separater)
        record.append(current_accuracy_info)




    return label0_highest,label1_highest

def center_loss(features,num_classes,labels,alpha,prefix):
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers_'+prefix, [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.argmax(labels,axis=1)
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)
    center_loss = tf.nn.l2_loss(features - centers_batch)
    diff = centers_batch - features

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff
    centers_update_op = tf.scatter_sub(centers, labels, diff)


    return center_loss,centers_update_op, centers

def build_model(batch_size,learning_rate,
                input_filters,
                logits_length_label1,
                logits_length_label0,
                device,
                extra_net,
                args_input,
                weight_decay,
                initializer):

    batch_images = tf.placeholder(tf.float32,
                                  [batch_size, args_input.image_size, args_input.image_size,input_filters],
                                  name='batch_image_train')

    batch_label1_labels = tf.placeholder(tf.float32,
                                  [batch_size, logits_length_label1],
                                  name='batch_label1_train')

    batch_label0_labels = tf.placeholder(tf.float32,
                                         [batch_size, logits_length_label0],
                                         name='batch_label0_train')

    batch_label1_logits, batch_label0_logits = extra_net(image=batch_images,
                                                         batch_size=batch_size,
                                                         device=device,
                                                         logits_length_font=logits_length_label1,
                                                         logits_length_character=logits_length_label0,
                                                         is_training=True,
                                                         reuse=False,
                                                         weight_decay=weight_decay,
                                                         initializer=initializer,
                                                         name_prefix='ExtraNet')



    cross_entropy_label1 = tf.nn.softmax_cross_entropy_with_logits(labels=batch_label1_labels,
                                                                   logits=batch_label1_logits,
                                                                   name='batch_cross_entropy_label1')
    cross_entropy_label0 = tf.nn.softmax_cross_entropy_with_logits(labels=batch_label0_labels,
                                                                   logits=batch_label0_logits,
                                                                   name='batch_cross_entropy_label0')

    cross_entropy_label1_avg = tf.reduce_mean(cross_entropy_label1, name='cross_entropy_label1') * args_input.label1_loss
    cross_entropy_label0_avg = tf.reduce_mean(cross_entropy_label0, name='cross_entropy_label0') * args_input.label0_loss

    category_label0_loss_summary = tf.summary.scalar("Loss_CategoryLoss0", cross_entropy_label0_avg/args_input.label0_loss)
    category_label1_loss_summary = tf.summary.scalar("Loss_CategoryLoss1", cross_entropy_label1_avg/args_input.label1_loss)



    center_loss_label0, center_update_label0_op, label0_centers = center_loss(features=batch_label0_logits,
                                                                              num_classes=logits_length_label0,
                                                                              labels=batch_label0_labels,
                                                                              alpha=learning_rate,
                                                                              prefix='label0')
    center_loss_label1, center_update_label1_op, label1_centers = center_loss(features=batch_label1_logits,
                                                                              num_classes=logits_length_label1,
                                                                              labels=batch_label1_labels,
                                                                              alpha=learning_rate,
                                                                              prefix='label1')
    center_loss_label0_penalty = args_input.label0_loss * args_input.center_loss_penalty_rate + eps
    center_loss_label1_penalty = args_input.label1_loss * args_input.center_loss_penalty_rate + eps
    center_loss_label0 = center_loss_label0 * center_loss_label0_penalty
    center_loss_label1 = center_loss_label1 * center_loss_label1_penalty
    center_update_op = [center_update_label0_op,center_update_label1_op]
    center_vars = [label0_centers,label1_centers]

    center_label0_loss_summary = tf.summary.scalar("Loss_Center_Label0",
                                                   center_loss_label0 / center_loss_label0_penalty)
    center_label1_loss_summary = tf.summary.scalar("Loss_Center_Label1",
                                                   center_loss_label1 / center_loss_label1_penalty)



    merged_loss_summary = tf.summary.merge([category_label1_loss_summary, category_label0_loss_summary,
                                            center_label0_loss_summary, center_label1_loss_summary])

    current_input_handle = InputHandle(batch_images=batch_images,
                                       batch_label1_labels=batch_label1_labels,
                                       batch_label0_labels=batch_label0_labels)


    loss_optimization = 0
    if args_input.label0_loss > 10 * eps:
        loss_optimization+=cross_entropy_label0_avg
        if args_input.center_loss_penalty_rate > eps:
            loss_optimization += center_loss_label0
    if args_input.label1_loss > 10 * eps:
        loss_optimization+=cross_entropy_label1_avg
        if args_input.center_loss_penalty_rate > eps:
            loss_optimization += center_loss_label1

    weight_decay_loss_list = tf.get_collection('ExtraNet_weight_decay')
    if weight_decay_loss_list:
        weight_decay_loss = 0
        for ii in weight_decay_loss_list:
            weight_decay_loss += ii
        weight_decay_loss = weight_decay_loss / len(weight_decay_loss_list)
        weight_decay_loss_summary = tf.summary.scalar("Loss_WeightDecay",
                                                      tf.abs(weight_decay_loss))
        loss_optimization = loss_optimization + weight_decay_loss
        merged_loss_summary = tf.summary.merge([merged_loss_summary,weight_decay_loss_summary])

    return merged_loss_summary,loss_optimization,\
           cross_entropy_label0_avg,cross_entropy_label1_avg,center_loss_label0,center_loss_label1,\
           current_input_handle,center_update_op,center_vars


def checkpoint(sess, saver,model_dir,model_name,counter):
    model_name = model_name+".model"
    saver.save(sess, os.path.join(model_dir, model_name), global_step=int(counter))


def find_vars_from_trained_model(input_full_vars,vars_you_want_to_find):


    if vars_you_want_to_find == 'ALL':
        output_vars = input_full_vars
    else:

        vars_you_want_to_find = vars_you_want_to_find.split(',')
        output_vars = list()
        for individual_var in vars_you_want_to_find:
            var_found = [var for var in input_full_vars if individual_var in var.name]


            if not var_found:
                print("Didn't find var:%s" % individual_var)
                return -1, False
            else:
                output_vars.extend(var_found)

    return output_vars,True


