# -*- coding: utf-8 -*-
"""
Created on 2021/08/22 17:03:38

@File -> s3_select_assoc_features.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 选择关联特征
"""

import pandas as pd
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from src.setting import TARGETS
from src.setting import load_data_sets
from core.association import cal_rmic

if __name__ == '__main__':
    thres = 0 # 0.15

    for y_col in TARGETS:
        _, _, X, y, x_cols = load_data_sets(y_col, normal_y=True)

        total_assoc_df = pd.DataFrame(x_cols, columns=['feature'])
        
        assocs_ = []
        for i in range(X.shape[1]):
            assocs_.append(cal_rmic(X[:, i], y.flatten(), x_type='numeric'))
        assocs_ = pd.DataFrame(assocs_, columns=['rmic'])
        total_assoc_df = pd.concat([total_assoc_df, assocs_], axis=1)

        assoc_df = total_assoc_df[total_assoc_df['rmic'] > thres]
        unassoc_df = total_assoc_df[total_assoc_df['rmic'] <= thres].sort_values(by='rmic', ascending=False)
        print(assoc_df.shape, unassoc_df.shape)
    
        assoc_df.to_csv(
            os.path.join(BASE_DIR, 'file/s1_assoc_analysis/assoc_features_{}.csv'.format(y_col)), 
            index=False
            )
        unassoc_df.to_csv(
            os.path.join(BASE_DIR, 'file/s1_assoc_analysis/unassoc_features_{}.csv'.format(y_col)), 
            index=False
            )