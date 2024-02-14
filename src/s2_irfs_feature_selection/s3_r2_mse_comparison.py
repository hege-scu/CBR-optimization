# -*- coding: utf-8 -*-
"""
Created on 2021/12/06 14:40:18

@File -> s3_r2_mse_comparison.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 
"""

import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from src.setting import IRFS_METHODS, TARGETS

IRFS_METHODS = ['CRMICM', 'JMIM', 'JMI', 'MRMR']

def get_metric_fimt_table(y_col, model_name):
    table = None
    for i, method in enumerate(IRFS_METHODS):
        r2_fimt_curve = pd.read_csv(
            os.path.join(BASE_DIR, 'file/s2_incremental_test/fimt_accu_{}_{}_{}_{}.csv'.format(
                y_col, method, model_name, 'r2')))
        # mse_fimt_curve = pd.read_csv(
        #     '../../file/s2_incremental_test/fimt_accu_{}_{}_{}_{}.csv'.format(
        #         y_col, method, model_name, 'mse'))

        n_features = r2_fimt_curve['n_features'].values
        r2_values = r2_fimt_curve[method].values
        # mse_values = mse_fimt_curve[method].values

        locs = np.argwhere(np.isin(n_features, [1, 31, 61, 91, 121])).flatten()
        
        table_ = [method] + r2_values[locs].tolist()
        table_ = pd.DataFrame([table_])

        if table is None:
            table = table_
        else:
            table = pd.concat([table, table_], axis=0)
    return table
    

def get_min_features_n_for_specific_metric(y_col, model_name, method, thres: float, metric: str):
    metric_fimt_curve = pd.read_csv(os.path.join(BASE_DIR, 'file/s2_incremental_test/fimt_accu_{}_{}_{}_{}.csv'.format(y_col, method, model_name, metric)))

    n_features = metric_fimt_curve['n_features'].values
    metric_values = metric_fimt_curve[method].values
    
    if metric == 'r2':
        locs = np.argwhere(metric_values >= thres)
        if len(locs) == 0:
            return 'NA'
        else:
            return n_features[locs.min()]
    elif metric in ['mse', 'mape']:
        locs = np.argwhere(metric_values <= thres)
        if len(locs) == 0:
            return 'NA'
        else:
            return n_features[locs.min()]
    else:
        raise ValueError


if __name__ == '__main__':
    model_name = 'rf'
    y_col = 'gasoline'

    # ---- 获得不同特征数下对应指标 ------------------------------------------------------------------
    
    # metric_table = get_metric_fimt_table(y_col, model_name)

    # ---- 获得达到指定指标值的特征数 ----------------------------------------------------------------

    metric = 'mape'
    thres = 0.07
    fn_lst = []
    for method in IRFS_METHODS:
        fn = get_min_features_n_for_specific_metric(y_col, model_name, method, thres, metric)
        fn_lst.append(fn)
    fn_df = pd.DataFrame([fn_lst])
    

    