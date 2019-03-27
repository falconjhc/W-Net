# -*- coding: utf-8 -*-


import sys
reload(sys)
sys.setdefaultencoding('utf8')
from pyExcelerator import *

import os
import numpy as np
import matplotlib.pyplot as plt


reading_result_path = '/Data_HDD/Harric/ChineseCharacterExp/EvalResult/EvaluationResult_201901/'
saving_path = '../EvaluationResults/EvaluationLineGraphs_201901/'


checking_dataset = 'StyleHw50'
checking_mode = 'ContentUnKnown-StyleUnKnown'

saving_path = os.path.join(saving_path,checking_dataset)
saving_path = os.path.join(saving_path,checking_mode)
if not os.path.exists(saving_path):
    os.makedirs(saving_path)



color_list = list()
color_list.append('red')
color_list.append('green')
color_list.append('blue')
color_list.append('lightcoral')
color_list.append('chocolate')
color_list.append('gold')
color_list.append('lime')
color_list.append('darkslategrey')
color_list.append('darkred')
color_list.append('slategray')
color_list.append('y')
color_list.append('lightgreen')
color_list.append('hotpink')



def list_sub_dir(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            if (os.path.isdir(m)):
                h = os.path.split(m)
                # print h[1]
                list.append(h[1])
        return list

def find_excel_table_mse_vn(current_result_read_path, mark):

    mse_avg = np.loadtxt(os.path.join(current_result_read_path,'Avg_Feature'+mark+'.csv'), dtype=np.str, delimiter=",")
    mse_std = np.loadtxt(os.path.join(current_result_read_path, 'Std_Feature'+mark+'.csv'), dtype=np.str, delimiter=",")

    true_fake_avg_list = list()
    random_content_avg_list = list()
    same_content_avg_list = list()
    random_style_avg_list = list()
    same_style_avg_list = list()
    if mark == 'MSE':
        layer_num = 7
    elif mark == 'VN':
        layer_num = 5
    for layer in range(layer_num):
        true_fake_avg = mse_avg[0, layer]
        true_fake_std = mse_std[0, layer]
        true_fake_avg = "%.3f" % float(true_fake_avg)
        true_fake_std = "+/-%.5f" % float(true_fake_std)
        true_fake_avg_list.append(true_fake_avg)

        random_content_avg = mse_avg[1, layer]
        random_content_std = mse_std[1, layer]
        random_content_std_avg = mse_avg[2, layer]
        random_content_avg = "%.3f" % float(random_content_avg)
        random_content_std = "+/-%.5f" % float(random_content_std)
        random_content_std_avg = "+/-%.5f" % float(random_content_std_avg)
        random_content_avg_list.append(random_content_avg)

        same_content_avg = mse_avg[3, layer]
        same_content_std = mse_std[3, layer]
        same_content_avg = "%.3f" % float(same_content_avg)
        same_content_std = "+/-%.5f" % float(same_content_std)
        same_content_avg_list.append(same_content_avg)

        random_style_avg = mse_avg[4, layer]
        random_style_std = mse_std[4, layer]
        random_style_std_avg = mse_avg[5, layer]
        random_style_avg = "%.3f" % float(random_style_avg)
        random_style_std = "+/-%.5f" % float(random_style_std)
        random_style_std_avg = "+/-%.5f" % float(random_style_std_avg)
        random_style_avg_list.append(random_style_avg)

        same_style_avg = mse_avg[6, layer]
        same_style_std = mse_std[6, layer]
        same_style_avg = "%.3f" % float(same_style_avg)
        same_style_std = "+/-%.5f" % float(same_style_std)
        same_style_avg_list.append(same_style_avg)


    return true_fake_avg_list, random_content_avg_list, same_content_avg_list, random_style_avg_list, same_style_avg_list



def find_excel_table_pixel(current_result_read_path):

    avg = np.loadtxt(os.path.join(current_result_read_path,'Avg_PixelDiff.csv'), dtype=np.str, delimiter=",")
    std = np.loadtxt(os.path.join(current_result_read_path, 'Std_PixelDiff.csv'), dtype=np.str, delimiter=",")

    same_l1_avg = '%.3f' % float(avg[0,0])
    same_l1_std = '%.5f'% float(std[0,0])
    same_mse_avg = '%.3f' % float(avg[0,2])
    same_mse_std = '%.5f'% float(std[0,2])
    same_pdar_avg = '%.3f' % float(avg[0,4])
    same_pdar_std = '%.5f'%  float(std[0,4])

    random_content_l1_avg = '%.3f' % float(avg[1,0])
    random_content_l1_std = '%.5f'% float(std[1,0])
    random_content_l1_std_avg = '%.5f'% float(avg[1,1])
    random_content_mse_avg = '%.3f' % float(avg[1, 2])
    random_content_mse_std = '%.5f' % float(std[1, 2])
    random_content_mse_std_avg = '%.5f' % float(avg[1, 3])
    random_content_pdar_avg = '%.3f' % float(avg[1, 4])
    random_content_pdar_std = '%.5f' % float(std[1, 4])
    random_content_pdar_std_avg = '%.5f' % float(avg[1, 5])

    random_style_l1_avg = '%.3f' % float(avg[2, 0])
    random_style_l1_std = '%.5f' % float(std[2, 0])
    random_style_l1_std_avg = '%.5f' % float(avg[2, 1])
    random_style_mse_avg = '%.3f' % float(avg[2, 2])
    random_style_mse_std = '%.5f' % float(std[2, 2])
    random_style_mse_std_avg = '%.5f' % float(avg[2, 3])
    random_style_pdar_avg = '%.3f' % float(avg[2, 4])
    random_style_pdar_std = '%.5f' % float(std[2, 4])
    random_style_pdar_std_avg = '%.5f' % float(avg[2, 5])


    return same_l1_avg, same_mse_avg, same_pdar_avg, \
           random_content_l1_avg, random_content_mse_avg, random_content_pdar_avg, \
           random_style_l1_avg, random_style_mse_avg, random_style_pdar_avg




def find_real_exp_num(exp_list):
    valid_exp_name_list = list()
    exp_full_list = list()
    for exp in exp_list:
        exp_name = exp
        if not (('WNet' in exp_name) and (checking_dataset in exp_name) and (checking_mode in exp_name)):
            continue
        keyword_find = [ii for ii in range(len(exp_name)) if exp_name.startswith('Style', ii)]
        for keyword in keyword_find:
            keyword_str = exp_name[keyword:keyword + 7]
            if keyword_str[-1].isdigit():
                style_info = keyword_str
                break
        exp_name_modified = exp_name.replace(style_info,'')
        if not exp_name_modified in valid_exp_name_list:
            valid_exp_name_list.append(exp_name_modified)
            exp_full_list.append(list())
            exp_full_list[-1].append(exp_name)
        else:
            found_index = valid_exp_name_list.index(exp_name_modified)
            exp_full_list[found_index].append(exp_name)
    return exp_full_list,valid_exp_name_list


def line_graph_data_collect(exp_list):

    #pixele_l1_list = np.zeros(len(exp_list)).tolist()
    style_num_list = list()
    pixel_same_l1_list = list()
    pixel_same_mse_list = list()
    pixel_same_pdar_list = list()
    pixel_random_content_l1_list = list()
    pixel_random_content_mse_list = list()
    pixel_random_content_pdar_list = list()
    pixel_random_style_l1_list = list()
    pixel_random_style_mse_list = list()
    pixel_random_style_pdar_list = list()
    mse_true_fake_avg_list = list()
    mse_random_content_avg_list = list()
    mse_same_content_avg_list = list()
    mse_random_style_avg_list = list()
    mse_same_style_avg_list = list()
    vn_true_fake_avg_list = list()
    vn_random_content_avg_list = list()
    vn_same_content_avg_list = list()
    vn_random_style_avg_list = list()
    vn_same_style_avg_list = list()
    for each_exp in exp_list:

        # get style num
        exp_name = each_exp.split('/')[-1]
        keyword_find = [ii for ii in range(len(exp_name)) if exp_name.startswith('Style', ii)]
        for keyword in keyword_find:
            keyword_str = exp_name[keyword:keyword + 7]
            if keyword_str[-1].isdigit():
                sheet_name = keyword_str
                break
        style_num = int(sheet_name[5:])
        style_num_list.append(style_num)


        # get pixel result
        current_path = os.path.join(reading_result_path,each_exp)
        same_l1_avg, same_mse_avg, same_pdar_avg, \
        random_content_l1_avg, random_content_mse_avg, random_content_pdar_avg, \
        random_style_l1_avg, random_style_mse_avg, random_style_pdar_avg = \
            find_excel_table_pixel(current_result_read_path=current_path)
        pixel_same_l1_list.append(same_l1_avg)
        pixel_same_mse_list.append(same_mse_avg)
        pixel_same_pdar_list.append(same_pdar_avg)
        pixel_random_content_l1_list.append(random_content_l1_avg)
        pixel_random_content_mse_list.append(random_content_mse_avg)
        pixel_random_content_pdar_list.append(random_content_pdar_avg)
        pixel_random_style_l1_list.append(random_style_l1_avg)
        pixel_random_style_mse_list.append(random_style_mse_avg)
        pixel_random_style_pdar_list.append(random_style_pdar_avg)


        # get deep mse
        mse_true_fake_avg, \
        mse_random_content_avg, mse_same_content_avg, \
        mse_random_style_avg, mse_same_style_avg = \
            find_excel_table_mse_vn(current_result_read_path=current_path, mark='MSE')
        mse_true_fake_avg_list.append(mse_true_fake_avg)
        mse_random_content_avg_list.append(mse_random_content_avg)
        mse_same_content_avg_list.append(mse_same_content_avg)
        mse_random_style_avg_list.append(mse_random_style_avg)
        mse_same_style_avg_list.append(mse_same_style_avg)

        # get deep vn
        vn_true_fake_avg, \
        vn_random_content_avg, vn_same_content_avg, \
        vn_random_style_avg, vn_same_style_avg = \
            find_excel_table_mse_vn(current_result_read_path=current_path, mark='VN')
        vn_true_fake_avg_list.append(vn_true_fake_avg)
        vn_random_content_avg_list.append(vn_random_content_avg)
        vn_same_content_avg_list.append(vn_same_content_avg)
        vn_random_style_avg_list.append(vn_random_style_avg)
        vn_same_style_avg_list.append(vn_same_style_avg)


    return style_num_list, \
           pixel_same_l1_list, pixel_same_mse_list, pixel_same_pdar_list, \
           pixel_random_content_l1_list, pixel_random_content_mse_list, pixel_random_content_pdar_list, \
           pixel_random_style_l1_list, pixel_random_style_mse_list, pixel_random_style_pdar_list, \
           mse_true_fake_avg_list, mse_random_content_avg_list, mse_same_content_avg_list, mse_random_style_avg_list, mse_same_style_avg_list, \
           vn_true_fake_avg_list, vn_random_content_avg_list, vn_same_content_avg_list, vn_random_style_avg_list, vn_same_style_avg_list



def draw_line_func_for_pixel(draw_y, style_num, exp_name_list, y_axis_name, fig_saving_info):
    plt.figure(figsize=(10, 7))
    counter=0
    for y,x, exp_label in zip(draw_y, style_num, exp_name_list):
        current_exp_label = exp_label[0]
        x_sorted, y_sorted = zip(*sorted(zip(x,y)))
        plt.plot(x_sorted, y_sorted, color=color_list[counter], label='Exp%02d' % (counter+1))
        counter+=1
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Number of Style References', fontsize=20)
    plt.ylabel(y_axis_name, fontsize=20)
    plt.legend(fontsize=12, loc=4)
    #plt.show()
    fig_saving_info = y_axis_name + '-' + fig_saving_info
    current_saving_path = os.path.join(saving_path, fig_saving_info)+'.pdf'
    plt.savefig(current_saving_path)

def draw_line_func_for_deep_feature(draw_y, style_num, exp_name_list, y_axis_name, fig_saving_info):
    found_index = y_axis_name.find('Layer')
    layer_select = int(y_axis_name[found_index+5])
    actual_draw_y_list = list()
    for exp_counter in range(len(draw_y)):
        current_value_list = list()
        for style_num_counter in range(len(draw_y[exp_counter])):
            current_value = draw_y[exp_counter][style_num_counter][layer_select-1]
            current_value_list.append(current_value)
        actual_draw_y_list.append(current_value_list)

    plt.figure(figsize=(10, 7))
    counter = 0
    for y, x, exp_label in zip(actual_draw_y_list, style_num, exp_name_list):
        current_exp_label = exp_label[0]
        x_sorted, y_sorted = zip(*sorted(zip(x, y)))
        plt.plot(x_sorted, y_sorted, color=color_list[counter], label='Exp%02d' % (counter + 1))
        counter += 1
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Number of Style References', fontsize=20)
    plt.ylabel(y_axis_name, fontsize=20)
    plt.legend(fontsize=12, loc=4)
    # plt.show()
    fig_saving_info = y_axis_name + '-' + fig_saving_info
    current_saving_path = os.path.join(saving_path, fig_saving_info) + '.pdf'
    plt.savefig(current_saving_path)

def write_to_txt(exp_name_list,txt_name):
    file_handle = open(os.path.join(saving_path, txt_name+'.txt'), 'w')
    len_counter = 0
    for curt_name in exp_name_list:
        write_info = 'Exp:%d  | %s' % (len_counter + 1, curt_name)
        file_handle.write(write_info)
        file_handle.write('\n')
        len_counter += 1
    file_handle.close()



def main():

    exp_list  = list_sub_dir(path=reading_result_path)
    exp_list.sort()
    exp_full_list_structured,valid_exp_name_list = find_real_exp_num(exp_list=exp_list)

    style_num_list = list()

    pixel_same_l1_list = list()
    pixel_same_mse_list = list()
    pixel_same_pdar_list = list()

    pixel_random_content_l1_list = list()
    pixel_random_content_mse_list = list()
    pixel_random_content_pdar_list = list()

    pixel_random_style_l1_list = list()
    pixel_random_style_mse_list = list()
    pixel_random_style_pdar_list = list()

    mse_true_fake_avg_list = list()
    mse_random_content_avg_list = list()
    mse_same_content_avg_list = list()
    mse_random_style_avg_list = list()
    mse_same_style_avg_list = list()

    vn_true_fake_avg_list = list()
    vn_random_content_avg_list = list()
    vn_same_content_avg_list = list()
    vn_random_style_avg_list = list()
    vn_same_style_avg_list = list()
    for each_exp_list in exp_full_list_structured:
        style_num, \
        pixel_same_l1, pixel_same_mse, pixel_same_pdar, \
        pixel_random_content_l1, pixel_random_content_mse, pixel_random_content_pdar, \
        pixel_random_style_l1, pixel_random_style_mse, pixel_random_style_pdar, \
        mse_true_fake_avg, mse_random_content_avg, mse_same_content_avg, mse_random_style_avg, mse_same_style_avg, \
        vn_true_fake_avg, vn_random_content_avg, vn_same_content_avg, vn_random_style_avg, vn_same_style_avg = \
            line_graph_data_collect(exp_list=each_exp_list)

        style_num_list.append(style_num)

        pixel_same_l1_list.append(pixel_same_l1)
        pixel_same_mse_list.append(pixel_same_mse)
        pixel_same_pdar_list.append(pixel_same_pdar)

        pixel_random_content_l1_list.append(pixel_random_content_l1)
        pixel_random_content_mse_list.append(pixel_random_content_mse)
        pixel_random_content_pdar_list.append(pixel_random_content_pdar)

        pixel_random_style_l1_list.append(pixel_random_style_l1)
        pixel_random_style_mse_list.append(pixel_random_style_mse)
        pixel_random_style_pdar_list.append(pixel_random_content_pdar)

        mse_true_fake_avg_list.append(mse_true_fake_avg)
        mse_random_content_avg_list.append(mse_random_content_avg)
        mse_same_content_avg_list.append(mse_same_content_avg)
        mse_random_style_avg_list.append(mse_random_style_avg)
        mse_same_style_avg_list.append(mse_same_style_avg)

        vn_true_fake_avg_list.append(vn_true_fake_avg)
        vn_random_content_avg_list.append(vn_random_content_avg)
        vn_same_content_avg_list.append(vn_same_content_avg)
        vn_random_style_avg_list.append(vn_random_style_avg)
        vn_same_style_avg_list.append(vn_same_style_avg)

    write_to_txt(exp_name_list=valid_exp_name_list,
                 txt_name=checking_dataset + '-' + checking_mode)

    # draw pixel curves
    draw_line_func_for_pixel(draw_y=pixel_same_l1_list,
                             style_num=style_num_list,
                             exp_name_list=exp_full_list_structured,
                             y_axis_name='SameCharacter-Pixel-L1',
                             fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_pixel(draw_y=pixel_same_mse_list,
                             style_num=style_num_list,
                             exp_name_list=exp_full_list_structured,
                             y_axis_name='SameCharacter-Pixel-MSE',
                             fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_pixel(draw_y=pixel_same_pdar_list,
                             style_num=style_num_list,
                             exp_name_list=exp_full_list_structured,
                             y_axis_name='SameCharacter-Pixel-PDAR',
                             fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)


    # draw deep mse curves
    draw_line_func_for_deep_feature(draw_y=mse_true_fake_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameCharacter-DeepFeature-Layer5-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_true_fake_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameCharacter-DeepFeature-Layer4-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_true_fake_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameCharacter-DeepFeature-Layer3-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_true_fake_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameCharacter-DeepFeature-Layer2-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_true_fake_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameCharacter-DeepFeature-Layer1-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)

    draw_line_func_for_deep_feature(draw_y=mse_random_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomContent-DeepFeature-Layer5-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_random_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomContent-DeepFeature-Layer4-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_random_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomContent-DeepFeature-Layer3-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_random_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomContent-DeepFeature-Layer2-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_random_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomContent-DeepFeature-Layer1-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_same_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameContent-DeepFeature-Layer5-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_same_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameContent-DeepFeature-Layer4-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_same_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameContent-DeepFeature-Layer3-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_same_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameContent-DeepFeature-Layer2-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_same_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameContent-DeepFeature-Layer1-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)

    draw_line_func_for_deep_feature(draw_y=mse_random_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomStyle-DeepFeature-Layer5-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_random_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomStyle-DeepFeature-Layer4-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_random_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomStyle-DeepFeature-Layer3-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_random_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomStyle-DeepFeature-Layer2-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_random_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomStyle-DeepFeature-Layer1-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)

    draw_line_func_for_deep_feature(draw_y=mse_same_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameStyle-DeepFeature-Layer5-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_same_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameStyle-DeepFeature-Layer4-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_same_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameStyle-DeepFeature-Layer3-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_same_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameStyle-DeepFeature-Layer2-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=mse_same_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameStyle-DeepFeature-Layer1-MSE',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)

    # draw deep vn curves
    draw_line_func_for_deep_feature(draw_y=vn_true_fake_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameCharacter-DeepFeature-Layer5-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_true_fake_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameCharacter-DeepFeature-Layer4-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_true_fake_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameCharacter-DeepFeature-Layer3-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_true_fake_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameCharacter-DeepFeature-Layer2-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_true_fake_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameCharacter-DeepFeature-Layer1-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)

    draw_line_func_for_deep_feature(draw_y=vn_random_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomContent-DeepFeature-Layer5-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_random_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomContent-DeepFeature-Layer4-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_random_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomContent-DeepFeature-Layer3-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_random_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomContent-DeepFeature-Layer2-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_random_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomContent-DeepFeature-Layer1-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_same_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameContent-DeepFeature-Layer5-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_same_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameContent-DeepFeature-Layer4-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_same_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameContent-DeepFeature-Layer3-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_same_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameContent-DeepFeature-Layer2-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_same_content_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameContent-DeepFeature-Layer1-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)

    draw_line_func_for_deep_feature(draw_y=vn_random_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomStyle-DeepFeature-Layer5-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_random_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomStyle-DeepFeature-Layer4-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_random_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomStyle-DeepFeature-Layer3-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_random_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomStyle-DeepFeature-Layer2-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_random_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='RandomStyle-DeepFeature-Layer1-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)

    draw_line_func_for_deep_feature(draw_y=vn_same_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameStyle-DeepFeature-Layer5-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_same_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameStyle-DeepFeature-Layer4-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_same_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameStyle-DeepFeature-Layer3-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_same_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameStyle-DeepFeature-Layer2-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)
    draw_line_func_for_deep_feature(draw_y=vn_same_style_avg_list,
                                    style_num=style_num_list,
                                    exp_name_list=exp_full_list_structured,
                                    y_axis_name='SameStyle-DeepFeature-Layer1-VN',
                                    fig_saving_info='W-Net' + '-' + checking_dataset + '-' + checking_mode)





if __name__ == "__main__":
    main()