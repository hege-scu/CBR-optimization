# -*- coding: utf-8 -*-
"""
Created on 2021/08/22 15:40:29

@File -> s1_reverse_fimt.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 
"""

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMModel as LightGBM
# from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as mse, r2_score as r2
from collections import defaultdict
import pandas as pd
import numpy as np
import copy
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from src.setting import TARGETS
from src.setting import load_data_sets, train_test_split
from src.s1_association_analysis.setting import ASSOC_METHODS

def _init_model(name: str):
    if name == 'rf':
        model = RandomForestRegressor(
            n_estimators=100, 
            max_features='sqrt', 
            min_samples_leaf=3, 
            min_samples_split=10,
            n_jobs=2
            )
    if name == 'lightgbm':
        model = LightGBM(
            objective='regression',
            n_estimators=100,
            min_child_samples=50, 
            # min_samples_split=120
        )
    # if name == 'xgb':
    #     model = XGBRegressor(
    #         booster='gbtree',
    #         n_estimators=100,
    #         learning_rate=0.1,
    #         max_depth = 3,
    #     )
    
    return model


NAME = 'rf'
TEST_RATIO = 0.3
MODEL = _init_model(NAME)


def _cal_metric(y_true, y_pred, metric: str):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    if metric == 'r2':
        return r2(y_true, y_pred)
    if metric == 'mse':
        return mse(y_true, y_pred)
    if metric == 'mape':
        idxs = np.where(y_true != 0)
        y_true = y_true[idxs]
        y_pred = y_pred[idxs]
        return np.sum(np.abs((y_pred - y_true) / y_true)) / len(y_true)


def exec_repeat_test(r: int=2):
    r2_lst, mse_lst, mape_lst = [], [], []
    for i in range(r):
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_sub, y_sub, test_ratio=TEST_RATIO, seed=i)

        rgsr = copy.deepcopy(MODEL)
        rgsr.fit(X_train, Y_train.flatten())
        Y_test_pred = rgsr.predict(X_test)

        r2_value = r2(Y_test, Y_test_pred)
        mse_value = mse(Y_test, Y_test_pred)
        mape_value = _cal_metric(Y_test, Y_test_pred, metric='mape')

        r2_lst.append(r2_value)
        mse_lst.append(mse_value)
        mape_lst.append(mape_value)
    return r2_lst, mse_lst, mape_lst


if __name__ == '__main__':
    result_idx=2  # 控制这里的idx来选用f1或accu画图
    n_features = np.arange(1, 30 + 3, 3)

    for y_col in TARGETS:
        X_df, _, _, y, _ = load_data_sets(y_col, normal_y=True)
        assoc_df = pd.read_csv(os.path.join(BASE_DIR, 'file/s1_assoc_analysis/assoc_df_{}.csv'.format(y_col)))

        for method in ASSOC_METHODS:
            print('\ny_col: {}, \tassoc_method: {}'.format(y_col, method))
            assoc_df_ = assoc_df[['feature', method]].sort_values(by=method, ascending=False)

            results = defaultdict(list)
            for loc in n_features:
                features_sub = assoc_df_.iloc[-loc:]['feature'].tolist()

                # 数据样本.
                X_sub = copy.deepcopy(X_df)[features_sub].values
                y_sub = copy.deepcopy(y).reshape(-1, 1)
                
                r2_lst, mse_lst, mape_lst = exec_repeat_test()
                results[method].append([r2_lst, mse_lst, mape_lst])

            m_df = pd.DataFrame(n_features, columns=['n_features'])
            m_ = pd.DataFrame([np.mean(p[result_idx]) for p in results[method]], columns=[method])
            m_df = pd.concat([m_df, m_], axis=1)
            m_df.to_csv(
                os.path.join(BASE_DIR, 'file/s1_assoc_analysis/fimt_accu_{}_{}_{}_{}.csv'.format(NAME, y_col, method, result_idx)), 
                index=False)