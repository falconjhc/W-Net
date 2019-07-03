# -*- coding: utf-8 -*-


import sys
reload(sys)
sys.setdefaultencoding('utf8')
from pyExcelerator import *

import os
import numpy as np

from xlrd import open_workbook
from xlutils.copy import copy

# reading_result_path = '/Data_HDD/Harric/ChineseCharacterExp/EvalResult/EvaluationResult_201901/'
# saving_path = '../EvaluationResults/EvaluationResult_201901/'

reading_result_path = '/Users/harric/Downloads/Eval_201904/'
saving_path = '/Users/harric/Desktop/Eval_201904/EvaluationResult_201901/'


if not os.path.exists(saving_path):
    os.makedirs(saving_path)

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

def create_excel_mse_vn_table_head(mark):
    w = Workbook()
    for case in range(16):
        if case == 0:
            sheet_name = 'Pf50-Sty1-CntKN-StyKN'
        elif case == 1:
            sheet_name = 'Pf50-Sty1-CntUnKN-StyKN'
        elif case == 2:
            sheet_name = 'Pf50-Sty1-CntKN-StyUnKN'
        elif case == 3:
            sheet_name = 'Pf50-Sty1-CntUnKN-StyUnKN'
        elif case == 4:
            sheet_name = 'Pf50-Sty4-CntKN-StyKN'
        elif case == 5:
            sheet_name = 'Pf50-Sty4-CntUnKN-StyKN'
        elif case == 6:
            sheet_name = 'Pf50-Sty4-CntKN-StyUnKN'
        elif case == 7:
            sheet_name = 'Pf50-Sty4-CntUnKN-StyUnKN'
        elif case == 8:
            sheet_name = 'Hw50-Sty1-CntKN-StyKN'
        elif case == 9:
            sheet_name = 'Hw50-Sty1-CntUnKN-StyKN'
        elif case == 10:
            sheet_name = 'Hw50-Sty1-CntKN-StyUnKN'
        elif case == 11:
            sheet_name = 'Hw50-Sty1-CntUnKN-StyUnKN'
        elif case == 12:
            sheet_name = 'Hw50-Sty4-CntKN-StyKN'
        elif case == 13:
            sheet_name = 'Hw50-Sty4-CntUnKN-StyKN'
        elif case == 14:
            sheet_name = 'Hw50-Sty4-CntKN-StyUnKN'
        elif case == 15:
            sheet_name = 'Hw50-Sty4-CntUnKN-StyUnKN'

        w_style1 = w.add_sheet(sheet_name)  #
        w_style1.write(0, 0,'')
        if mark == 'MSE':
            layer_num = 7
        elif mark=='VN':
            layer_num = 5
        for ii in range(layer_num):
            w_style1.write(1, 1+ii*12, 'Layer%d-TrueFake-Avg' % (ii + 1))
            w_style1.write(1, 2+ii*12, 'Layer%d-TrueFake-Std' % (ii + 1))
            w_style1.write(0, 1 + ii * 12, 1+ii*12)
            w_style1.write(0, 2 + ii * 12, 2+ii*12)

            w_style1.write(1, 3+ii*12, 'Layer%d-RandomContent-Avg' % (ii + 1))
            w_style1.write(1, 4+ii*12, 'Layer%d-RandomContent-Std'% (ii + 1))
            w_style1.write(1, 5+ii*12, 'Layer%d-RandomContent-Std-Std' % (ii + 1))
            w_style1.write(0, 3 + ii * 12, 3+ii*12)
            w_style1.write(0, 4 + ii * 12, 4+ii*12)
            w_style1.write(0, 5 + ii * 12, 5+ii*12)

            w_style1.write(1, 6+ii*12, 'Layer%d-SameContent-Avg' % (ii + 1))
            w_style1.write(1, 7+ii*12, 'Layer%d-SameContent-Std' % (ii + 1))
            w_style1.write(0, 6 + ii * 12, 6+ii*12)
            w_style1.write(0, 7 + ii * 12, 7+ii*12)

            w_style1.write(1, 8+ii*12, 'Layer%d-RandomStyle-Avg' % (ii + 1))
            w_style1.write(1, 9+ii*12, 'Layer%d-RandomStyle-Std' % (ii + 1))
            w_style1.write(1, 10+ii*12, 'Layer%d-RandomStyle-Std-Std' % (ii + 1))
            w_style1.write(0, 8 + ii * 12, 8+ii*12)
            w_style1.write(0, 9 + ii * 12, 9+ii*12)
            w_style1.write(0, 10 + ii * 12, 10+ii*12)

            w_style1.write(1, 11+ii*12, 'Layer%d-SameStyle-Avg' % (ii + 1))
            w_style1.write(1, 12+ii*12, 'Layer%d-SameStyle-Std' % (ii + 1))
            w_style1.write(0, 11 + ii * 12, 11+ii*12)
            w_style1.write(0, 12 + ii * 12, 12+ii*12)

        w.save(os.path.join(saving_path, mark+'.xls'))

def create_excel_pixel_table_head():
    w = Workbook()
    for case in range(16):
        if case == 0:
            sheet_name = 'Pf50-Sty1-CntKN-StyKN'
        elif case == 1:
            sheet_name = 'Pf50-Sty1-CntUnKN-StyKN'
        elif case == 2:
            sheet_name = 'Pf50-Sty1-CntKN-StyUnKN'
        elif case == 3:
            sheet_name = 'Pf50-Sty1-CntUnKN-StyUnKN'
        elif case == 4:
            sheet_name = 'Pf50-Sty4-CntKN-StyKN'
        elif case == 5:
            sheet_name = 'Pf50-Sty4-CntUnKN-StyKN'
        elif case == 6:
            sheet_name = 'Pf50-Sty4-CntKN-StyUnKN'
        elif case == 7:
            sheet_name = 'Pf50-Sty4-CntUnKN-StyUnKN'
        elif case == 8:
            sheet_name = 'Hw50-Sty1-CntKN-StyKN'
        elif case == 9:
            sheet_name = 'Hw50-Sty1-CntUnKN-StyKN'
        elif case == 10:
            sheet_name = 'Hw50-Sty1-CntKN-StyUnKN'
        elif case == 11:
            sheet_name = 'Hw50-Sty1-CntUnKN-StyUnKN'
        elif case == 12:
            sheet_name = 'Hw50-Sty4-CntKN-StyKN'
        elif case == 13:
            sheet_name = 'Hw50-Sty4-CntUnKN-StyKN'
        elif case == 14:
            sheet_name = 'Hw50-Sty4-CntKN-StyUnKN'
        elif case == 15:
            sheet_name = 'Hw50-Sty4-CntUnKN-StyUnKN'

        work_sheet = w.add_sheet(sheet_name)  #
        work_sheet.write(0, 0,'')
        work_sheet.write(1, 1, 'Same-Pixel-L1-Avg')
        work_sheet.write(1, 2, 'Same-Pixel-L1-Std')
        work_sheet.write(1, 3, 'Same-Pixel-MSE-Avg')
        work_sheet.write(1, 4, 'Same-Pixel-MSE-Std')
        work_sheet.write(1, 5, 'Same-Pixel-PDAR-Avg')
        work_sheet.write(1, 6, 'Same-Pixel-PDAR-Std')
        work_sheet.write(1, 7, 'RandomContent-Pixel-L1-Avg')
        work_sheet.write(1, 8, 'RandomContent-Pixel-L1-Std')
        work_sheet.write(1, 9, 'RandomContent-Pixel-L1-Std-Avg')
        work_sheet.write(1, 10, 'RandomContent-Pixel-MSE-Avg')
        work_sheet.write(1, 11, 'RandomContent-Pixel-MSE-Std')
        work_sheet.write(1, 12, 'RandomContent-Pixel-MSE-Std-Avg')
        work_sheet.write(1, 13, 'RandomContent-Pixel-PDAR-Avg')
        work_sheet.write(1, 14, 'RandomContent-Pixel-PDAR-Std')
        work_sheet.write(1, 15, 'RandomContent-Pixel-PDAR-Std-Avg')
        work_sheet.write(1, 16, 'RandomStyle-Pixel-L1-Avg')
        work_sheet.write(1, 17, 'RandomStyle-Pixel-L1-Std')
        work_sheet.write(1, 18, 'RandomStyle-Pixel-L1-Std-Avg')
        work_sheet.write(1, 19, 'RandomStyle-Pixel-MSE-Avg')
        work_sheet.write(1, 20, 'RandomStyle-Pixel-MSE-Std')
        work_sheet.write(1, 21, 'RandomStyle-Pixel-MSE-Std-Avg')
        work_sheet.write(1, 22, 'RandomStyle-Pixel-PDAR-Avg')
        work_sheet.write(1, 23, 'RandomStyle-Pixel-PDAR-Std')
        work_sheet.write(1, 24, 'RandomStyle-Pixel-PDAR-Std-Avg')

        work_sheet.write(0, 1, 1)
        work_sheet.write(0, 2, 2)
        work_sheet.write(0, 3, 3)
        work_sheet.write(0, 4, 4)
        work_sheet.write(0, 5, 5)
        work_sheet.write(0, 6, 6)
        work_sheet.write(0, 7, 7)
        work_sheet.write(0, 8, 8)
        work_sheet.write(0, 9, 9)
        work_sheet.write(0, 10, 10)
        work_sheet.write(0, 11, 11)
        work_sheet.write(0, 12, 12)
        work_sheet.write(0, 13, 13)
        work_sheet.write(0, 14, 14)
        work_sheet.write(0, 15, 15)
        work_sheet.write(0, 16, 16)
        work_sheet.write(0, 17, 17)
        work_sheet.write(0, 18, 18)
        work_sheet.write(0, 19, 19)
        work_sheet.write(0, 20, 20)
        work_sheet.write(0, 21, 21)
        work_sheet.write(0, 22, 22)
        work_sheet.write(0, 23, 23)
        work_sheet.write(0, 24, 24)
        w.save(os.path.join(saving_path, 'Pixel.xls'))






def fillin_excel_table_mse_vn(current_result_read_path, mark):
    sheet_name = '-1'
    exp_name = current_result_read_path.split('/')[-1]
    exp_name = exp_name[12:]
    keyword_find = [ii for ii in range(len(exp_name)) if exp_name.startswith('Style', ii)]
    for keyword in keyword_find:
        keyword_str = exp_name[keyword:keyword + 7]
        str_len = len(keyword_str)
        if (not keyword_str[str_len - 1].isdigit()) and (keyword_str[str_len - 2].isdigit()) or len(keyword_str) == 6:
            if not len(keyword_str) == 6:
                sheet_name_tmp = keyword_str[0:str_len - 1]
                if sheet_name_tmp == 'Style1' or sheet_name_tmp == 'Style4':
                    sheet_name = sheet_name_tmp
                    break
            elif keyword_str == 'Style1' or keyword_str == 'Style4':
                sheet_name = keyword_str
                break
    if not sheet_name == '-1':
        mse_avg = np.loadtxt(os.path.join(current_result_read_path, 'Avg_Feature' + mark + '.csv'), dtype=np.str,
                             delimiter=",")
        mse_std = np.loadtxt(os.path.join(current_result_read_path, 'Std_Feature' + mark + '.csv'), dtype=np.str,
                             delimiter=",")

        if not exp_name.find('StylePf50') == -1:
            if sheet_name == 'Style1':
                if (not exp_name.find('ContentKnown') == -1) and (not exp_name.find('StyleKnown') == -1):
                    sheet_id = 0
                elif (not exp_name.find('ContentUnKnown') == -1) and (not exp_name.find('StyleKnown') == -1):
                    sheet_id = 1
                elif (not exp_name.find('ContentKnown') == -1) and (not exp_name.find('StyleUnKnown') == -1):
                    sheet_id = 2
                elif (not exp_name.find('ContentUnKnown') == -1) and (not exp_name.find('StyleUnKnown') == -1):
                    sheet_id = 3
            elif sheet_name == 'Style4':
                if (not exp_name.find('ContentKnown') == -1) and (not exp_name.find('StyleKnown') == -1):
                    sheet_id = 4
                elif (not exp_name.find('ContentUnKnown') == -1) and (not exp_name.find('StyleKnown') == -1):
                    sheet_id = 5
                elif (not exp_name.find('ContentKnown') == -1) and (not exp_name.find('StyleUnKnown') == -1):
                    sheet_id = 6
                elif (not exp_name.find('ContentUnKnown') == -1) and (not exp_name.find('StyleUnKnown') == -1):
                    sheet_id = 7
        elif not exp_name.find('StyleHw50') == -1:
            if sheet_name == 'Style1':
                if (not exp_name.find('ContentKnown') == -1) and (not exp_name.find('StyleKnown') == -1):
                    sheet_id = 8
                elif (not exp_name.find('ContentUnKnown') == -1) and (not exp_name.find('StyleKnown') == -1):
                    sheet_id = 9
                elif (not exp_name.find('ContentKnown') == -1) and (not exp_name.find('StyleUnKnown') == -1):
                    sheet_id = 10
                elif (not exp_name.find('ContentUnKnown') == -1) and (not exp_name.find('StyleUnKnown') == -1):
                    sheet_id = 11
            elif sheet_name == 'Style4':
                if (not exp_name.find('ContentKnown') == -1) and (not exp_name.find('StyleKnown') == -1):
                    sheet_id = 12
                elif (not exp_name.find('ContentUnKnown') == -1) and (not exp_name.find('StyleKnown') == -1):
                    sheet_id = 13
                elif (not exp_name.find('ContentKnown') == -1) and (not exp_name.find('StyleUnKnown') == -1):
                    sheet_id = 14
                elif (not exp_name.find('ContentUnKnown') == -1) and (not exp_name.find('StyleUnKnown') == -1):
                    sheet_id = 15

        rexcel = open_workbook(os.path.join(saving_path, mark + '.xls'))
        rows = rexcel.sheets()[sheet_id].nrows
        excel = copy(rexcel)
        table = excel.get_sheet(sheet_id)
        table.write(rows, 0, exp_name)

        if mark == 'MSE':
            layer_num = 7
        elif mark == 'VN':
            layer_num = 5
        for layer in range(layer_num):
            true_fake_avg = mse_avg[0, layer]
            true_fake_std = mse_std[0, layer]
            true_fake_avg = "%.3f" % float(true_fake_avg)
            true_fake_std = "+/-%.5f" % float(true_fake_std)

            random_content_avg = mse_avg[1, layer]
            random_content_std = mse_std[1, layer]
            random_content_std_avg = mse_avg[2, layer]
            random_content_avg = "%.3f" % float(random_content_avg)
            random_content_std = "+/-%.5f" % float(random_content_std)
            random_content_std_avg = "+/-%.5f" % float(random_content_std_avg)

            same_content_avg = mse_avg[3, layer]
            same_content_std = mse_std[3, layer]
            same_content_avg = "%.3f" % float(same_content_avg)
            same_content_std = "+/-%.5f" % float(same_content_std)

            random_style_avg = mse_avg[4, layer]
            random_style_std = mse_std[4, layer]
            random_style_std_avg = mse_avg[5, layer]
            random_style_avg = "%.3f" % float(random_style_avg)
            random_style_std = "+/-%.5f" % float(random_style_std)
            random_style_std_avg = "+/-%.5f" % float(random_style_std_avg)

            same_style_avg = mse_avg[6, layer]
            same_style_std = mse_std[6, layer]
            same_style_avg = "%.3f" % float(same_style_avg)
            same_style_std = "+/-%.5f" % float(same_style_std)

            table.write(rows, 1 + 12 * layer, true_fake_avg)
            table.write(rows, 2 + 12 * layer, true_fake_std)

            table.write(rows, 3 + 12 * layer, random_content_avg)
            table.write(rows, 4 + 12 * layer, random_content_std)
            table.write(rows, 5 + 12 * layer, random_content_std_avg)

            table.write(rows, 6 + 12 * layer, same_content_avg)
            table.write(rows, 7 + 12 * layer, same_content_std)

            table.write(rows, 8 + 12 * layer, random_style_avg)
            table.write(rows, 9 + 12 * layer, random_style_std)
            table.write(rows, 10 + 12 * layer, random_style_std_avg)

            table.write(rows, 11 + 12 * layer, same_style_avg)
            table.write(rows, 12 + 12 * layer, same_style_std)

        excel.save(os.path.join(saving_path, mark + '.xls'))



def fillin_excel_table_pixel(current_result_read_path):

    sheet_name='-1'
    exp_name = current_result_read_path.split('/')[-1]
    exp_name = exp_name[12:]
    keyword_find = [ii for ii in range(len(exp_name)) if exp_name.startswith('Style', ii)]
    for keyword in keyword_find:
        keyword_str = exp_name[keyword:keyword+7]
        str_len = len(keyword_str)
        if (not keyword_str[str_len-1].isdigit()) and (keyword_str[str_len-2].isdigit()) or len(keyword_str)==6:
            if not len(keyword_str)==6:
                sheet_name_tmp = keyword_str[0:str_len - 1]
                if sheet_name_tmp == 'Style1' or sheet_name_tmp == 'Style4':
                    sheet_name = sheet_name_tmp
                    break
            elif keyword_str == 'Style1' or keyword_str == 'Style4':
                sheet_name = keyword_str
                break



    if not sheet_name == '-1':
        avg = np.loadtxt(os.path.join(current_result_read_path, 'Avg_PixelDiff.csv'), dtype=np.str, delimiter=",")
        std = np.loadtxt(os.path.join(current_result_read_path, 'Std_PixelDiff.csv'), dtype=np.str, delimiter=",")

        if not exp_name.find('StylePf50') == -1:
            if sheet_name == 'Style1':
                if (not exp_name.find('ContentKnown') == -1) and (not exp_name.find('StyleKnown') == -1):
                    sheet_id = 0
                elif (not exp_name.find('ContentUnKnown') == -1) and (not exp_name.find('StyleKnown') == -1):
                    sheet_id = 1
                elif (not exp_name.find('ContentKnown') == -1) and (not exp_name.find('StyleUnKnown') == -1):
                    sheet_id = 2
                elif (not exp_name.find('ContentUnKnown') == -1) and (not exp_name.find('StyleUnKnown') == -1):
                    sheet_id = 3
            elif sheet_name == 'Style4':
                if (not exp_name.find('ContentKnown') == -1) and (not exp_name.find('StyleKnown') == -1):
                    sheet_id = 4
                elif (not exp_name.find('ContentUnKnown') == -1) and (not exp_name.find('StyleKnown') == -1):
                    sheet_id = 5
                elif (not exp_name.find('ContentKnown') == -1) and (not exp_name.find('StyleUnKnown') == -1):
                    sheet_id = 6
                elif (not exp_name.find('ContentUnKnown') == -1) and (not exp_name.find('StyleUnKnown') == -1):
                    sheet_id = 7
        elif not exp_name.find('StyleHw50') == -1:
            if sheet_name == 'Style1':
                if (not exp_name.find('ContentKnown') == -1) and (not exp_name.find('StyleKnown') == -1):
                    sheet_id = 8
                elif (not exp_name.find('ContentUnKnown') == -1) and (not exp_name.find('StyleKnown') == -1):
                    sheet_id = 9
                elif (not exp_name.find('ContentKnown') == -1) and (not exp_name.find('StyleUnKnown') == -1):
                    sheet_id = 10
                elif (not exp_name.find('ContentUnKnown') == -1) and (not exp_name.find('StyleUnKnown') == -1):
                    sheet_id = 11
            elif sheet_name == 'Style4':
                if (not exp_name.find('ContentKnown') == -1) and (not exp_name.find('StyleKnown') == -1):
                    sheet_id = 12
                elif (not exp_name.find('ContentUnKnown') == -1) and (not exp_name.find('StyleKnown') == -1):
                    sheet_id = 13
                elif (not exp_name.find('ContentKnown') == -1) and (not exp_name.find('StyleUnKnown') == -1):
                    sheet_id = 14
                elif (not exp_name.find('ContentUnKnown') == -1) and (not exp_name.find('StyleUnKnown') == -1):
                    sheet_id = 15

        rexcel = open_workbook(os.path.join(saving_path, 'Pixel.xls'))
        rows = rexcel.sheets()[sheet_id].nrows
        excel = copy(rexcel)
        table = excel.get_sheet(sheet_id)
        table.write(rows, 0, exp_name)

        same_l1_avg = '%.3f' % float(avg[0, 0])
        same_l1_std = '%.5f' % float(std[0, 0])
        same_mse_avg = '%.3f' % float(avg[0, 2])
        same_mse_std = '%.5f' % float(std[0, 2])
        same_pdar_avg = '%.3f' % float(avg[0, 4])
        same_pdar_std = '%.5f' % float(std[0, 4])

        random_content_l1_avg = '%.3f' % float(avg[1, 0])
        random_content_l1_std = '%.5f' % float(std[1, 0])
        random_content_l1_std_avg = '%.5f' % float(avg[1, 1])
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

        table.write(rows, 1, same_l1_avg)
        table.write(rows, 2, same_l1_std)
        table.write(rows, 3, same_mse_avg)
        table.write(rows, 4, same_mse_std)
        table.write(rows, 5, same_pdar_avg)
        table.write(rows, 6, same_pdar_std)

        table.write(rows, 7, random_content_l1_avg)
        table.write(rows, 8, random_content_l1_std)
        table.write(rows, 9, random_content_l1_std_avg)
        table.write(rows, 10, random_content_mse_avg)
        table.write(rows, 11, random_content_mse_std)
        table.write(rows, 12, random_content_mse_std_avg)
        table.write(rows, 13, random_content_pdar_avg)
        table.write(rows, 14, random_content_pdar_std)
        table.write(rows, 15, random_content_pdar_std_avg)

        table.write(rows, 16, random_style_l1_avg)
        table.write(rows, 17, random_style_l1_std)
        table.write(rows, 18, random_style_l1_std_avg)
        table.write(rows, 19, random_style_mse_avg)
        table.write(rows, 20, random_style_mse_std)
        table.write(rows, 21, random_style_mse_std_avg)
        table.write(rows, 22, random_style_pdar_avg)
        table.write(rows, 23, random_style_pdar_std)
        table.write(rows, 24, random_style_pdar_std_avg)

        excel.save(os.path.join(os.path.join(saving_path, 'Pixel.xls')))




def main():

    exp_list  = list_sub_dir(path=reading_result_path)
    exp_list.sort()
    create_excel_pixel_table_head()
    create_excel_mse_vn_table_head(mark='MSE')
    create_excel_mse_vn_table_head(mark='VN')
    exp_counter = 0
    for current_exp in exp_list:
        #current_exp = exp_list[106]
        print("%d/%d: %s" % (exp_counter+1, len(exp_list), current_exp))
        fillin_excel_table_pixel(current_result_read_path=os.path.join(reading_result_path,current_exp))
        fillin_excel_table_mse_vn(current_result_read_path=os.path.join(reading_result_path,current_exp), mark='MSE')
        fillin_excel_table_mse_vn(current_result_read_path=os.path.join(reading_result_path, current_exp), mark='VN')

        exp_counter+=1








if __name__ == "__main__":
    main()