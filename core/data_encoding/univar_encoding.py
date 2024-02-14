# -*- coding: utf-8 -*-
"""
Created on 2021/11/22 14:16:24

@File -> data_encoding.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据离散编码
"""

import pandas as pd
import numpy as np


# ---- 单个箱子 -------------------------------------------------------------------------------------

class Box(object):
    """单个分箱"""

    def __init__(self, x: np.ndarray, freq_n: int, bounds: list):
        self.x = x
        self.freq_n = freq_n
        self.bounds = bounds


def sum_boxs_freqs(boxes: dict):
    sum_ = 0 
    for k in boxes.keys():
        sum_ += boxes[k].freq_n
    return sum_ 


# ---- 数据分箱 -------------------------------------------------------------------------------------

class DataBinning(object):
    """数据分箱"""

    def __init__(self, x: np.ndarray):
        self.x = x.flatten()
        self.x_bounds = [self.x.min(), self.x.max()]

    def _exec_isometric_binning(self, bins: int):
        freq_ns, _intervals = np.histogram(self.x, bins, range=self.x_bounds)
        labels = _intervals[1:]  # 以每个箱子的右边界为label
        return freq_ns, labels

    def init_boxes(self, bins: int):
        freq_ns, labels = self._exec_isometric_binning(bins)  # 左闭右开

        self.boxes = {}
        for i in range(bins):
            x_min = self.x.min() if i == 0 else labels[i - 1]
            x_max = labels[i]
            if i != bins - 1:
                x = self.x[(x_min <= self.x) & (self.x < x_max)]  # 左闭右开
            else:
                x = self.x[(x_min <= self.x) & (self.x <= x_max)]  # 左闭右闭
            self.boxes[i] = Box(x, freq_ns[i], [x_min, x_max])

    def merge_empty_boxes(self):
        keys = sorted(self.boxes.keys())
        bins = len(keys)
        while True:
            for i in range(bins - 1):
                key_i, key_j = keys[i], keys[i + 1]
                box_i, box_j = self.boxes[key_i], self.boxes[key_j]
                if (box_i.freq_n == 0) & (box_j.freq_n == 0):
                    self.boxes[key_i] = Box(
                        np.hstack((box_i.x, box_j.x)),
                        box_i.freq_n + box_j.freq_n, 
                        [box_i.bounds[0], box_j.bounds[1]]
                    )
                    self.boxes.pop(key_j)
                    break
            
            keys = sorted(self.boxes.keys())
            if len(keys) == bins:
                break
            else:
                bins = len(self.boxes.keys())

    def get_divide_values(self):
        divide_values = []
        for k in self.boxes.keys():
            if self.boxes[k].freq_n == 0:
                divide_values.append(self.boxes[k].bounds)
        return divide_values


def encode(x: np.ndarray):
    x = x.copy()

    # 确定数据分割值.
    self = DataBinning(x)
    self.init_boxes(bins = 10)  # FIXME: 这个值不能太大
    self.merge_empty_boxes()
    divide_values = self.get_divide_values()

    # 对数据进行分割.
    x_subs = []
    if divide_values != []:
        for i in range(len(divide_values)):
            if i == 0:
                loc_r = divide_values[i][0]
                idxs = np.where(x <= loc_r)[0]
                x_subs.append([x[idxs], idxs])
            else:
                loc_l, loc_r = divide_values[i - 1][1], divide_values[i][0]
                idxs = np.where((x <= loc_r) & (x >= loc_l))[0]
                x_subs.append([x[idxs], idxs])

            if i == len(divide_values) - 1:
                loc_l = divide_values[i][1]
                idxs = np.where(x >= loc_l)[0]
                x_subs.append([x[idxs], idxs])
    else:
        x_subs.append([x, np.arange(len(x))])

    # 逐子数据集离散化并对应替代原数据值.
    max_label = 0
    for i in range(len(x_subs)):
        x_, idxs_ = x_subs[i]
        q = max([1, len(x_) // 200])
        x_enc = pd.cut(x_, q, labels=False, duplicates='drop')
        x_enc = x_enc + max_label
        x[idxs_] = x_enc
        max_label = x_enc.max() + 1

    return x


    

# if __name__ == '__main__':
#     import sys
#     import os
    
#     BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
#     sys.path.append(BASE_DIR)

#     from src.setting import load_data_sets, plt

#     y_col = 'dry_gas'
#     X_df, y_df, X, y, x_cols = load_data_sets(y_col, normal_y=True)
#     x = X[:, 196]
    
#     plt.subplot(2, 1, 1)
#     plt.plot(x)

#     # 确定数据分割值.
#     self = DataBinning(x)
#     self.init_boxes(bins = 10)
#     self.merge_empty_boxes()
#     divide_values = self.get_divide_values()

#     # 对数据进行分割.
#     x_subs = []
#     if divide_values != []:
#         for i in range(len(divide_values)):
#             if i == 0:
#                 loc_r = divide_values[i][0]
#                 idxs = np.where(x <= loc_r)[0]
#                 x_subs.append([x[idxs], idxs])
#             else:
#                 loc_l, loc_r = divide_values[i - 1][1], divide_values[i][0]
#                 idxs = np.where((x <= loc_r) & (x >= loc_l))[0]
#                 x_subs.append([x[idxs], idxs])

#             if i == len(divide_values) - 1:
#                 loc_l = divide_values[i][1]
#                 idxs = np.where(x >= loc_l)[0]
#                 x_subs.append([x[idxs], idxs])
#     else:
#         x_subs.append([x, np.arange(len(x))])

#     # 逐子数据集离散化并对应替代原数据值.
#     max_label = 0
#     for i in range(len(x_subs)):
#         x_, idxs_ = x_subs[i]
#         q = max([1, len(x_) // 100])
#         x_enc = pd.qcut(x_, q, labels=False, duplicates='drop')
#         x_enc = x_enc + max_label
#         x[idxs_] = x_enc
#         max_label = x_enc.max() + 1
    
#     plt.subplot(2, 1, 2)
#     plt.plot(x)

    
    