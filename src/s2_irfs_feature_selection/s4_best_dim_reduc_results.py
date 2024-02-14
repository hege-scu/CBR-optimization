# -*- coding: utf-8 -*-
"""
Created on 2021/12/01 17:45:54

@File -> s3_merge_fimt_curves.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 获得各方法最优降维效果 
"""

import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from src.setting import IRFS_METHODS, TARGETS
from src.setting import plt


def locate_peak(series: np.ndarray, thres: float = 0.001):
    v_max = np.max(series)
    if np.abs(np.sort(series)[-2] - v_max) > thres:
        loc = np.argwhere(series == v_max).min()
    else:
        loc = np.argwhere(series >= (v_max - thres)).min()
    return loc


def cal_mape(y_true, y_pred):
    idxs = np.where(y_true != 0)
    y_true = y_true[idxs]
    y_pred = y_pred[idxs]
    return np.sum(np.abs((y_pred - y_true) / y_true)) / len(y_true)


if __name__ == '__main__':
    # method = 'CRMICM'
    # y_col = 'light_diesel'
    # model_name = 'rf'
    
    # r2_fimt_curve = pd.read_csv(
    #     '../../file/s2_incremental_test/fimt_accu_{}_{}_{}_{}.csv'.format(
    #         y_col, method, model_name, 'r2'))
    # mse_fimt_curve = pd.read_csv(
    #     '../../file/s2_incremental_test/fimt_accu_{}_{}_{}_{}.csv'.format(
    #         y_col, method, model_name, 'mse'))
    
    # n_features = r2_fimt_curve['n_features'].values
    # r2_values = r2_fimt_curve[method].values
    # mse_values = mse_fimt_curve[method].values
    # plt.plot(n_features, r2_values)
    # plt.grid(True)
    
    # max_loc = locate_peak(r2_values)
    # print('n_features: %d, r2: %.4f, mse: %.4f' % (
    #     n_features[max_loc], r2_values[max_loc], mse_values[max_loc]))
    
    model_name = 'rf'
    
    table = None
    for i, method in enumerate(IRFS_METHODS):
        fill_in_values = [method]
        for y_col in TARGETS:
            r2_fimt_curve = pd.read_csv(
                '../../file/s2_incremental_test/fimt_accu_{}_{}_{}_{}.csv'.format(
                    y_col, method, model_name, 'r2'))
            mse_fimt_curve = pd.read_csv(
                '../../file/s2_incremental_test/fimt_accu_{}_{}_{}_{}.csv'.format(
                    y_col, method, model_name, 'mse'))
            mape_fimt_curve = pd.read_csv(
                '../../file/s2_incremental_test/fimt_accu_{}_{}_{}_{}.csv'.format(
                    y_col, method, model_name, 'mape'))

            n_features = r2_fimt_curve['n_features'].values
            r2_values = r2_fimt_curve[method].values
            mse_values = mse_fimt_curve[method].values
            mape_values = mape_fimt_curve[method].values
            
            best_loc = locate_peak(r2_values)
            best_n = n_features[best_loc]
            best_r2 = r2_values[best_loc]
            best_mse = mse_values[best_loc]
            best_mape = mape_values[best_loc]
            
            fill_in_values += [best_n, best_r2, best_mse, best_mape]
        
        fill_in_values = pd.DataFrame([fill_in_values])
        
        if i == 0:
            table = fill_in_values
        else:
            table = pd.concat([table, fill_in_values], axis=0)

        
            