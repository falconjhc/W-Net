import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd
import os




def read_excel(file, sheet_index=0):

    workbook = xlrd.open_workbook(file)
    sheet = workbook.sheet_by_index(sheet_index)
    data = []
    for i in range(0, sheet.nrows):
        data.append(sheet.row_values(i))
    return data

def read_and_draw_excel_mse_vn(mark):
    for sheet_counter in range(8):

        if sheet_counter==0:
            sheet_name = 'Style1-ContentKnown-StyleKnown'
        elif sheet_counter==1:
            sheet_name = 'Style1-ContentUnKnown-StyleKnown'
        elif sheet_counter == 2:
            sheet_name = 'Style1-ContentKnown-StyleUnKnown'
        elif sheet_counter == 3:
            sheet_name = 'Style1-ContentUnKnown-StyleUnKnown'
        elif sheet_counter==4:
            sheet_name = 'Style4-ContentKnown-StyleKnown'
        elif sheet_counter==5:
            sheet_name = 'Style4-ContentUnKnown-StyleKnown'
        elif sheet_counter == 6:
            sheet_name = 'Style4-ContentKnown-StyleUnKnown'
        elif sheet_counter == 7:
            sheet_name = 'Style4-ContentUnKnown-StyleUnKnown'


        if mark == 'MSE':
            excel_content = read_excel('../EvaluationResults/SummarizedResults/MSE.xls', sheet_index=sheet_counter)
            layer_num = 7
        elif mark=='VN':
            excel_content = read_excel('../EvaluationResults/SummarizedResults/VN.xls', sheet_index=sheet_counter)
            layer_num = 5

        exp_num = len(excel_content)-2
        avg_matrix = np.zeros(shape=[exp_num, layer_num * 5], dtype=np.float64)
        std_matrix = np.zeros(shape=[exp_num, layer_num * 5], dtype=np.float64)
        std_avg_matrix = np.zeros(shape=[exp_num, layer_num * 5], dtype=np.float64)
        exp_name_list = list()
        for exp_counter in range(exp_num):
            current_exp_name = excel_content[exp_counter + 2][0]
            for layer_counter in range(layer_num):
                avg_matrix[exp_counter, 0 + layer_counter * 5] = excel_content[exp_counter + 2][1 + 12 * layer_counter] # true_fake-avg
                std_matrix[exp_counter, 0 + layer_counter * 5] = excel_content[exp_counter + 2][2 + 12 * layer_counter][3:] # true_fake-std
                std_avg_matrix[exp_counter, 0 + layer_counter * 5] = 0 # true_fake-std-avg

                avg_matrix[exp_counter, 1 + layer_counter * 5] = excel_content[exp_counter + 2][3 + 12 * layer_counter]  # randomContent-avg
                std_matrix[exp_counter, 1 + layer_counter * 5] = excel_content[exp_counter + 2][4 + 12 * layer_counter][3:]  # randomContent-std
                std_avg_matrix[exp_counter, 1 + layer_counter * 5] = excel_content[exp_counter + 2][5+ 12 * layer_counter][3:]  # randomContent-std-avg

                avg_matrix[exp_counter, 2 + layer_counter * 5] = excel_content[exp_counter + 2][6 + 12 * layer_counter]  # sameContent-avg
                std_matrix[exp_counter, 2 + layer_counter * 5] = excel_content[exp_counter + 2][7 + 12 * layer_counter][3:]  # sameContent-std
                std_avg_matrix[exp_counter, 2 + layer_counter * 5] = 0  # sameContent-std-avg

                avg_matrix[exp_counter, 3 + layer_counter * 5] = excel_content[exp_counter + 2][8 + 12 * layer_counter]  # randomStyle-avg
                std_matrix[exp_counter, 3 + layer_counter * 5] = excel_content[exp_counter + 2][9 + 12 * layer_counter][3:]  # randomStyle-std
                std_avg_matrix[exp_counter, 3 + layer_counter * 5] = excel_content[exp_counter + 2][10 + 12 * layer_counter][3:]  # randomStyle-std-avg

                avg_matrix[exp_counter, 4 + layer_counter * 5] = excel_content[exp_counter + 2][11 + 12 * layer_counter]  # sameStyle-avg
                std_matrix[exp_counter, 4 + layer_counter * 5] = excel_content[exp_counter + 2][12 + 12 * layer_counter][3:]  # sameStyle-std
                std_avg_matrix[exp_counter, 4 + layer_counter * 5] = 0  # sameContent-std-avg

            exp_name_list.append(current_exp_name)

        # draw excel to bars
        x_series = np.arange(len(exp_name_list))
        total_width, n = 0.9, 2
        width = total_width / n
        x_series = x_series - (total_width - width) / 2

        color_full = np.zeros(shape=[len(exp_name_list), 3],
                              dtype=np.float32)
        exp_list = list()
        for color_counter in range(len(exp_name_list)):
            current_random_color = np.random.uniform(low=0.5, high=1.0, size=[1, 3])
            color_full[color_counter, :] = current_random_color
            exp_list.append("Exp:%d" % (color_counter + 1))

        for layer_counter in range(layer_num):
            for measurement_count in range(5):
                if measurement_count == 0:
                    measurement_name = 'Feature-TrueFake'
                elif measurement_count == 1:
                    measurement_name = 'Feature-RandomContent'
                elif measurement_count == 2:
                    measurement_name = 'Feature-SameContent'
                elif measurement_count == 3:
                    measurement_name = 'Feature-RandomStyle'
                elif measurement_count == 4:
                    measurement_name = 'Feature-SameStyle'


                fig = plt.figure()
                current_avg = avg_matrix[:, layer_counter*5+measurement_count]
                current_std = std_matrix[:, layer_counter*5+measurement_count]
                current_std_avg = std_avg_matrix[:, layer_counter*5+measurement_count]
                plt.bar(x_series - width / 2, current_avg, yerr=current_std, align='center', alpha=0.5, width=width,
                        color=color_full)
                plt.bar(x_series + width / 2, current_avg, yerr=current_std_avg, align='center', alpha=0.5, width=width,
                        color=color_full)
                max_shift = np.max(np.concatenate([current_std, current_std_avg], axis=0))
                max_value = np.max(current_avg)
                for x, y in zip(x_series, current_avg):
                    plt.text(x, y - max_shift - 0.015 * max_value * 5, '%.3f' % y, ha='center', va='bottom')
                for x, y, y_org in zip(x_series, current_std, current_avg):
                    plt.text(x, y_org - max_shift - 0.025 * max_value * 5, '(+/-%.3f)' % y, ha='center', va='bottom')
                for x, y, y_org in zip(x_series, current_std_avg, current_avg):
                    plt.text(x, y_org - max_shift - 0.035 * max_value * 5, '(+/-%.3f)' % y, ha='center', va='bottom')

                plt.legend()
                plt.xticks(x_series, exp_list, rotation='vertical')
                saving_path = os.path.join('../EvaluationResults/SummarizedResults', sheet_name)
                saving_path = os.path.join(saving_path, 'Bar-VggDeepFeatures-'+mark)
                saving_path = os.path.join(saving_path,'Layer%d' % (layer_counter+1))
                if not os.path.exists(saving_path):
                    os.makedirs(saving_path)
                plt.savefig(os.path.join(saving_path, measurement_name + '.eps'))
                plt.close('all')
        print("%s: %s (%d/%d) completed;" % (mark,sheet_name,sheet_counter+1,8))


def read_and_draw_excel_pixel():
    for sheet_counter in range(8):

        if sheet_counter==0:
            sheet_name = 'Style1-ContentKnown-StyleKnown'
        elif sheet_counter==1:
            sheet_name = 'Style1-ContentUnKnown-StyleKnown'
        elif sheet_counter == 2:
            sheet_name = 'Style1-ContentKnown-StyleUnKnown'
        elif sheet_counter == 3:
            sheet_name = 'Style1-ContentUnKnown-StyleUnKnown'
        elif sheet_counter==4:
            sheet_name = 'Style4-ContentKnown-StyleKnown'
        elif sheet_counter==5:
            sheet_name = 'Style4-ContentUnKnown-StyleKnown'
        elif sheet_counter == 6:
            sheet_name = 'Style4-ContentKnown-StyleUnKnown'
        elif sheet_counter == 7:
            sheet_name = 'Style4-ContentUnKnown-StyleUnKnown'
        saving_path = os.path.join('../EvaluationResults/SummarizedResults', sheet_name)
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        excel_content = read_excel('../EvaluationResults/SummarizedResults/Pixel.xls', sheet_index=sheet_counter)
        exp_num = len(excel_content)-2
        avg_matrix = np.zeros(shape=[exp_num, 9], dtype=np.float64)
        std_matrix = np.zeros(shape=[exp_num, 9], dtype=np.float64)
        std_avg_matrix = np.zeros(shape=[exp_num, 9], dtype=np.float64)
        exp_name_list = list()
        for exp_counter in range(exp_num):
            current_exp_name = excel_content[exp_counter+2][0]

            avg_matrix[exp_counter, 0] = excel_content[exp_counter + 2][1] # same-pixel-l1-avg
            std_matrix[exp_counter, 0] = excel_content[exp_counter + 2][2] # same-pixel-l1-std
            std_avg_matrix[exp_counter, 0] = 0 # same-pixel-l1-std-avg

            avg_matrix[exp_counter, 1] = excel_content[exp_counter + 2][3]  # same-pixel-mse-avg
            std_matrix[exp_counter, 1] = excel_content[exp_counter + 2][4]  # same-pixel-mse-std
            std_avg_matrix[exp_counter, 1] = 0  # same-pixel-mse-std-avg

            avg_matrix[exp_counter, 2] = excel_content[exp_counter + 2][5]  # same-pixel-pdar-avg
            std_matrix[exp_counter, 2] = excel_content[exp_counter + 2][6]  # same-pixel-pdar-std
            std_avg_matrix[exp_counter, 2] = 0  # same-pixel-pdar-std-avg

            avg_matrix[exp_counter, 3] = excel_content[exp_counter + 2][7]  # randomContent-pixel-l1-avg
            std_matrix[exp_counter, 3] = excel_content[exp_counter + 2][8]  # randomContent-pixel-l1-std
            std_avg_matrix[exp_counter, 3] = excel_content[exp_counter + 2][9] # randomContent-pixel-l1-std-avg

            avg_matrix[exp_counter, 4] = excel_content[exp_counter + 2][10]  # randomContent-pixel-mse-avg
            std_matrix[exp_counter, 4] = excel_content[exp_counter + 2][11]  # randomContent-pixel-mse-std
            std_avg_matrix[exp_counter, 4] = excel_content[exp_counter + 2][12]  # randomContent-pixel-mse-std-avg

            avg_matrix[exp_counter, 5] = excel_content[exp_counter + 2][13]  # randomContent-pixel-pdar-avg
            std_matrix[exp_counter, 5] = excel_content[exp_counter + 2][14]  # randomContent-pixel-pdar-std
            std_avg_matrix[exp_counter, 5] = excel_content[exp_counter + 2][15]  # randomContent-pixel-pdar-std-avg

            avg_matrix[exp_counter, 6] = excel_content[exp_counter + 2][16]  # randomStyle-pixel-l1-avg
            std_matrix[exp_counter, 6] = excel_content[exp_counter + 2][17]  # randomStyle-pixel-l1-std
            std_avg_matrix[exp_counter, 6] = excel_content[exp_counter + 2][18]  # randomStyle-pixel-l1-std-avg

            avg_matrix[exp_counter, 7] = excel_content[exp_counter + 2][19]  # randomStyle-pixel-mse-avg
            std_matrix[exp_counter, 7] = excel_content[exp_counter + 2][20]  # randomStyle-pixel-mse-std
            std_avg_matrix[exp_counter, 7] = excel_content[exp_counter + 2][21]  # randomStyle-pixel-mse-std-avg

            avg_matrix[exp_counter, 8] = excel_content[exp_counter + 2][22]  # randomStyle-pixel-pdar-avg
            std_matrix[exp_counter, 8] = excel_content[exp_counter + 2][23]  # randomStyle-pixel-pdar-std
            std_avg_matrix[exp_counter, 8] = excel_content[exp_counter + 2][24]  # randomStyle-pixel-pdar-std-avg

            exp_name_list.append(current_exp_name)

        # write experiment names
        file_handle = open(os.path.join(saving_path,'ExpNames.txt'),'w')
        length = len(exp_name_list)
        len_counter = 0
        for curt_name in exp_name_list:
            write_info = 'Exp:%d  | %s' % (len_counter+1, curt_name)
            file_handle.write(write_info)
            file_handle.write('\n')
            len_counter+=1
        file_handle.close()



        # draw excel to bars
        x_series = np.arange(len(exp_name_list))
        total_width, n = 0.9, 2
        width = total_width / n
        x_series = x_series - (total_width - width) / 2

        color_full = np.zeros(shape=[len(exp_name_list), 3],
                              dtype=np.float32)
        exp_list = list()
        for color_counter in range(len(exp_name_list)):
            current_random_color = np.random.uniform(low=0.5, high=1.0, size=[1, 3])
            color_full[color_counter, :] = current_random_color
            exp_list.append("Exp:%d" % (color_counter + 1))

        for measurement_count in range(9):
            if measurement_count == 0:
                measurement_name = 'Same-Pixel-L1'
            elif measurement_count == 1:
                measurement_name = 'Same-Pixel-MSE'
            elif measurement_count == 2:
                measurement_name = 'Same-Pixel-PDAR'
            elif measurement_count == 3:
                measurement_name = 'RandomContent-Pixel-L1'
            elif measurement_count == 4:
                measurement_name = 'RandomContent-Pixel-MSE'
            elif measurement_count == 5:
                measurement_name = 'RandomContent-Pixel-PDAR'
            elif measurement_count == 6:
                measurement_name = 'RandomStyle-Pixel-L1'
            elif measurement_count == 7:
                measurement_name = 'RandomStyle-Pixel-MSE'
            elif measurement_count == 8:
                measurement_name = 'RandomStyle-Pixel-PDAR'

            fig = plt.figure()
            current_avg = avg_matrix[:, measurement_count]
            current_std = std_matrix[:, measurement_count]
            current_std_avg = std_avg_matrix[:, measurement_count]
            plt.bar(x_series - width / 2, current_avg, yerr=current_std, align='center', alpha=0.5, width=width,
                    color=color_full)
            plt.bar(x_series + width / 2, current_avg, yerr=current_std_avg, align='center', alpha=0.5, width=width,
                    color=color_full)
            max_shift = np.max(np.concatenate([current_std, current_std_avg], axis=0))
            max_value = np.max(current_avg)
            for x, y in zip(x_series, current_avg):
                plt.text(x, y - max_shift - 0.015*max_value*5, '%.3f' % y, ha='center', va='bottom')
            for x, y, y_org in zip(x_series, current_std, current_avg):
                plt.text(x, y_org - max_shift - 0.025*max_value*5, '(+/-%.3f)' % y, ha='center', va='bottom')
            for x, y, y_org in zip(x_series, current_std_avg, current_avg):
                plt.text(x, y_org - max_shift - 0.035*max_value*5, '(+/-%.3f)' % y, ha='center', va='bottom')

            plt.legend()
            plt.xticks(x_series, exp_list, rotation='vertical')
            # plt.show()

            current_saving_path = os.path.join(saving_path, 'Bar-Pixel/')
            if not os.path.exists(current_saving_path):
                os.makedirs(current_saving_path)
            plt.savefig(os.path.join(current_saving_path, measurement_name + '.eps'))
            plt.close('all')
        print("%s: %s (%d/%d) completed;" % ('Pixel', sheet_name, sheet_counter + 1, 8))

def main():
    read_and_draw_excel_pixel()
    read_and_draw_excel_mse_vn(mark='VN')
    read_and_draw_excel_mse_vn(mark='MSE')
    print("Complete All !")



if __name__ == "__main__":
    main()