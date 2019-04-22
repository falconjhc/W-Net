# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


def _cut(im, ax):
    im = cv.medianBlur(im, 5)

    ret, im_bw = cv.threshold(im, 200, 1, cv.THRESH_BINARY)
    im_bw[im_bw == 1] = 2
    im_bw[im_bw == 0] = 1
    im_bw[im_bw == 2] = 0

    im_h = np.sum(im_bw, axis=ax)

    seg = list()
    blank = True
    start = 0
    for i, p in enumerate(im_h):
        if blank is True:
            if p != 0:
                start = i
                blank = False
        else:
            if p == 0 and start != -1:
                seg.append([start, i - 1, i - 1 - start])
                start = -1
                blank = True
    if start != -1:
        seg.append([start, len(im_h), len(im_h) - start])
    if len(seg) < 2:
        seg_new = seg
    else:
        seg_new = list()
        gap = 0
        for i in range(len(seg) - 1):
            gap = gap + (seg[i + 1][0] - seg[i][1])
        mean_gap = gap / (len(seg) - 1)
        last_n = 1
        while len(seg) > 0:
            if len(seg) == last_n:
                seg_new.append([seg[0][0], seg[last_n - 1][1]])
                for j in range(last_n):
                    seg.pop(0)
            for i in range(len(seg) - 1):
                if i == len(seg) - 1:
                    seg_new.append([seg[0][0], seg[last_n - 1][1]])
                    for j in range(i + last_n):
                        seg.pop(0)
                    last_n = 1
                    break
                else:
                    gap = seg[i + 1][0] - seg[i][1]
                    if gap > mean_gap / 2:
                        seg_new.append([seg[0][0], seg[i][1]])
                        for j in range(i + 1):
                            seg.pop(0)
                        last_n = 1
                        break
                    else:
                        last_n = last_n + 1
    im_list = list()
    for s in seg_new:
        if ax == 0:
            im_split = im[:, s[0]:s[1]]
        else:
            im_split = im[s[0]:s[1], :]
        im_list.append(im_split)
    return im_list


def padding(im, center_size, final_size):
    ret, im_bw = cv.threshold(im, 200, 1, cv.THRESH_BINARY)
    im_bw[im_bw == 1] = 2
    im_bw[im_bw == 0] = 1
    im_bw[im_bw == 2] = 0
    im_h = np.sum(im_bw, axis=0)
    im_w = np.sum(im_bw, axis=1)
    w_start = 0
    w_end = 0
    h_start = 0
    h_end = 0
    for i, p in enumerate(im_h):
        if p != 0:
            w_start = i
            break

    for i in range(len(im_h) - 1, -1, -1):
        p = im_h[i]
        if p != 0:
            w_end = i
            break

    for i, p in enumerate(im_w):
        if p != 0:
            h_start = i
            break

    for i in range(len(im_w) - 1, -1, -1):
        p = im_w[i]
        if p != 0:
            h_end = i
            break

    im = im[h_start:h_end, w_start:w_end]

    h, w = im.shape[:2]
    if h > w:
        w = int((float(center_size) / h) * w)
        h = int(center_size)
    else:
        h = int((float(center_size) / w) * h)
        w = int(center_size)

    w = int(w / 2) * 2
    h = int(h / 2) * 2

    im = cv.resize(im, (w, h))
    imo = np.zeros((final_size, final_size), dtype=np.uint8) + 255

    imo[int(final_size / 2) - int(h / 2):int(final_size / 2) + int(h / 2), int(final_size / 2) - int(w / 2):int(final_size / 2) + int(w / 2)] = im
    return imo


def char_cut(im, center_size, final_size):
    im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    im_list = list()
    row_list = _cut(im, ax=1)
    for row in row_list:
        col_list = _cut(row, ax=0)
        for col in col_list:
            h, w = col.shape[:2]
            if h / w > 2 or w / h > 2:
                col_row_list = _cut(col, ax=1)
                for col_row in col_row_list:
                    im_list.append(padding(col_row, center_size, final_size))
            else:
                im_list.append(padding(col, center_size, final_size))
    return im_list


def main():
    im = cv.imread('image/9.jpg')
    ims = char_cut(im, center_size=37, final_size=64)   # center_size 是汉字长边的最宽宽度，final_size是图片最终长宽
    for i, im in enumerate(ims):
        cv.imwrite('result/%d.jpg' % i, im)


if __name__ == '__main__':
    main()
