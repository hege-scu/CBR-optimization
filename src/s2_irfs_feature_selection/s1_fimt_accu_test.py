# -*- coding: utf-8 -*-
"""
Created on 2021/11/25 16:42:11

@File -> s1_fimt_accu_test.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 特征增量准确度测试
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse, explained_variance_score as evs, r2_score as r2
from lightgbm import LGBMModel as LightGBM
from collections import defaultdict
import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from src.setting import load_data_sets, train_test_split
from src.s2_irfs_feature_selection.setting import N_FEATURES

TEST_RATIO = 0.3

def _init_model(name: str):
    if name == 'lightgbm':
        model = LightGBM(
            objective='regression',
            n_estimators=100,
            min_child_samples=50,
            boosting_type='goss',
            learning_rate=0.1,
            importance_type='split',
        )
    if name == 'rf':
        model = RandomForestRegressor(
            n_estimators=100, 
            max_features='sqrt', 
            min_samples_leaf=3, 
            min_samples_split=10,
            n_jobs = 3
            )
    # if name == 'xgb':
    #     model = XGBRegressor(
    #         booster='gbtree',
    #         n_estimators=100,
    #         learning_rate=0.1,
    #         max_depth = 3,
    #     )
    # if name == 'gbdt':
    #     model = GradientBoostingRegressor(
    #         n_estimators=100, 
    #         max_features='sqrt', 
    #         min_samples_leaf=50, 
    #         min_samples_split=150
    #     )
    # if name == 'knn':
    #     model = KNeighborsRegressor(
    #         n_neighbors=10,
    #         weights='distance',
    #         leaf_size=10,
    #         metric='euclidean'
    #     )
    # if name == 'svm':
    #     model = SVR(
    #         kernel='rbf',
    #         degree=4,
    #         C=1.0,
    #         epsilon=0.0,
    #     )
    # if name == 'mlp':
    #     model = MLPRegressor(
    #         hidden_layer_sizes=(6, 6),
    #         activation='relu',
    #         max_iter=10000,
    #         learning_rate_init=0.01,
    #         tol=0.0001,
    #         # random_state=1,
    #         alpha=0.1,
    #         batch_size=200,
    #         solver='lbfgs',
    #         warm_start=True,
    #         shuffle=False,
    #     )
    
    return model


def _cal_metric(y_true, y_pred, metric: str):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    if metric == 'r2':
        return r2(y_true, y_pred)
    if metric == 'evs':
        return evs(y_true, y_pred)
    if metric == 'mse':
        return mse(y_true, y_pred)
    if metric == 'mape':
        idxs = np.where(y_true != 0)
        y_true = y_true[idxs]
        y_pred = y_pred[idxs]
        return np.sum(np.abs((y_pred - y_true) / y_true)) / len(y_true)


def cal_fimt_values(y_col, methods, model_name, n_features, metric: str='r2'):
    """计算各特征选择方法的FIMT测试曲线值"""
    n_features_series = np.arange(1, n_features + 1, 1)  # NOTE: 这里需要修改

    for method in methods:
        print('\ny_col: {},\tmethod: {}'.format(y_col, method))

        results = defaultdict(list)
        X_df, _, _, y, _ = load_data_sets(y_col, normal_y=True, normal_x=True)

        # ---- 模型预测准备 -------------------------------------------------------------------------

        # 这里已经包括了所有关联特征和无关特征.
        features_df = pd.read_csv(os.path.join(BASE_DIR, 'file/s2_incremental_test/{}_{}.csv'.format(y_col, method)))
        features = features_df['x'].to_list()

        for i, loc in enumerate(n_features_series):
            features_sub = features[:loc]

            # 数据样本.
            X_sub = X_df[features_sub].values.copy()
            y_sub = y.reshape(-1, 1)

            # ---- 模型预测 -------------------------------------------------------------------------

            X_train, X_test, Y_train, Y_test = train_test_split(X_sub, y_sub, test_ratio=TEST_RATIO, seed=0)
            while True:
                model = _init_model(model_name)
                model.fit(X_train, Y_train.flatten())
                Y_train_pred = model.predict(X_train)
                Y_test_pred = model.predict(X_test)
                
                metric_train = _cal_metric(Y_train.flatten(), Y_train_pred.flatten(), metric)
                metric_test = _cal_metric(Y_test.flatten(), Y_test_pred.flatten(), metric)
                
                if (model_name == 'mlp') & (i >= 5):
                    if (metric == 'r2') & (np.abs(metric_test - results[method][-1]) < 0.2):
                        break
                else:
                    break

            # print('train %.4f, test %.4f' % (metric_train, metric_test))

            results[method].append(metric_test)

        m_df = pd.DataFrame(n_features_series, columns=['n_features'])
        m_ = pd.DataFrame(results[method], columns=[method])
        m_df = pd.concat([m_df, m_], axis=1)
        m_df.to_csv(
            os.path.join(BASE_DIR, 'file/s2_incremental_test/fimt_accu_{}_{}_{}_{}.csv'.format(
                y_col, method, model_name, metric)), index=False)


if __name__ == '__main__':
    from src.setting import TARGETS
    
    IRFS_METHODS = ['CRMICM', 'JMIM', 'JMI', 'MRMR'] # IRFS_METHODS = ['CRMICM'] #

    name = 'rf'
    metric = 'mse' # 改变不同的标准进行计算 r2, mape, mse

    for y_col in TARGETS[:]:
        cal_fimt_values(y_col, IRFS_METHODS, name, N_FEATURES, metric)