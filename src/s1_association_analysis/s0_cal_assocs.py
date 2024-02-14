# -*- coding: utf-8 -*-
"""
Created on 2021/08/22 14:51:49

@File -> main.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据关联分析
"""

import pandas as pd
import warnings
import sys
import os

warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from src.setting import TARGETS, load_data_sets
from core.association import cal_dist_corr, cal_pearson_corr, cal_mi, cal_mic, cal_rmic, cal_spearman_corr


def cal_assoc_df(X, y, assoc_df, func, func_name):
    assocs_ = []
    for i in range(X.shape[1]):
        assocs_.append(func(X[:, i], y.flatten(), x_type='numeric'))
    assocs_ = pd.DataFrame(assocs_, columns=[func_name])
    assoc_df = pd.concat([assoc_df, assocs_], axis=1)
    return assoc_df


if __name__ == '__main__':
    for y_col in TARGETS:
        _, _, X, y, x_cols = load_data_sets(y_col, normal_y=True)

        funcs = [cal_dist_corr, cal_pearson_corr, cal_mi, cal_mic, cal_rmic, cal_spearman_corr]
        # funcs = [cal_mi]
        assoc_df = pd.DataFrame(x_cols, columns=['feature'])
        for func in funcs:
            func_name = func.__name__[4:]
            print('func: {}'.format(func_name))
            assoc_df = cal_assoc_df(X, y, assoc_df, func, func_name)
        
        assoc_df.to_csv(
            os.path.join(BASE_DIR, 'file/s1_assoc_analysis/assoc_df_{}.csv'.format(y_col)), 
            index=False)