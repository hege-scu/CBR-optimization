# -*- coding: utf-8 -*-
"""
Created on 2021/11/25 15:21:08

@File -> s0_feature_selection.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 特征选择
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
import pandas as pd
import numpy as np
import warnings
import time
import json
import sys
import os

warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from src.setting import TARGETS, load_data_sets
from core.feature_selection.micm.crmicm import CondRMICMaximization
from core.feature_selection.micm.cmim_pyitlib import CondMIMaximization
# from core.feature_selection.micm.cmim_kraskov import CondMIMaximization
from core.feature_selection.mifs.mifs import MutualInformationFeatureSelector as MIFS
from core.association import cal_mi, cal_mic, cal_dist_corr, cal_pearson_corr, cal_spearman_corr
from src.s2_irfs_feature_selection.setting import N_FEATURES
from mod.data_process.numpy import discretize_arr, discretize_series

ASSOC_NAME_MAP = {
    'CRMICM': 'rmic',
    'CMIM': 'mi',
    'Pearson': 'pearson_corr',
    'Spearman': 'spearman_corr',
    'DistCorr': 'dist_corr',
    'MI': 'mi',
    'MIC': 'mic',
    'MDI': 'pearson_corr',
    'MDA': 'pearson_corr',
    'JMI': 'mi',
    'JMIM': 'mi',
    'MRMR': 'mi'}


def _get_feature_assocs_df(y_col, method):
    """获得该方法在对应目标上的全量特征关联系数表"""
    assoc_df = pd.read_csv(os.path.join(BASE_DIR, 'file/s1_assoc_analysis/assoc_df_%s.csv' % y_col))
    return assoc_df[['feature', ASSOC_NAME_MAP[method]]]


def _save_logs(logs, y_col, method):
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(os.path.join(BASE_DIR, 'file/logs/{}_{}.csv'.format(method, y_col)), index=False)


def exec_feature_selection(y_col, method, n_features) -> list:
    """执行特征选择"""
    
    # ---- step 1: 载入数据和对应关联系数 ------------------------------------------------------------
    
    _, _, X, y, x_cols = load_data_sets(y_col, normal_y=True, normal_x=True)
    feature_assocs = _get_feature_assocs_df(y_col, method)
    
    # ---- step 2: 特征选择 -------------------------------------------------------------------------

    # 时间记录.
    # with open('../../file/records/time_cost.json', 'r+') as f:
    #     tc = json.load(f)
    # time_costs = tc['feature_selection']  # type: dict
    # ts = time.time()

    if method == 'CRMICM':
        selector = CondRMICMaximization(X, y)
        logs = selector.exec_feature_selection(assoc_thres=0.0, iter_n=n_features)
        features_selected = [x_cols[p[1]] for p in logs]
        _save_logs(logs, y_col, method)

    if method == 'CMIM':
        selector = CondMIMaximization(X, y)
        logs = selector.exec_feature_selection(assoc_thres=0.1, iter_n=n_features)
        features_selected = [x_cols[p[1]] for p in logs]
        _save_logs(logs, y_col, method)

    if method in ['Pearson', 'Spearman', 'DistCorr', 'MI', 'MIC']:
        assoc_measure = {
            'Pearson': cal_pearson_corr, 
            'Spearman': cal_spearman_corr, 
            'DistCorr': cal_dist_corr, 
            'MI': cal_mi, 
            'MIC': cal_mic
            }

        accus = {}
        for i in range(len(x_cols)):
            accus[i] = assoc_measure[method](X[:, i], y.astype(np.float).flatten())
        accus_sorted = sorted(accus.items(), key=lambda d: d[1], reverse=True)
        features_selected = [x_cols[p[0]] for p in accus_sorted[:n_features]]

    if method in ['JMIM', 'JMI', 'MRMR']:
        N_lim = 500
        N = X.shape[0]
        idxs = np.random.permutation(range(N))[: N_lim]
        X, y = X[idxs, :], y[idxs, :]

        # 先离散化.
        # X = discretize_arr(X, n=80)
        y = discretize_series(y.flatten(), n=80)

        selector = MIFS(
            k=3,
            method=method,
            n_features=n_features,
            categorical=False,
            verbose=2,
            n_jobs=3)
        selector.fit(X, y.flatten())
        features_selected = [x_cols[p] for p in selector.ranking_]

    if method in ['MDI', 'MDA']:
        rgsr = RandomForestRegressor(
            n_estimators=10, 
            max_features='sqrt',
            max_depth=3,
            min_samples_leaf=50, 
            min_samples_split=150)
    
        if method == 'MDI':
            rgsr.fit(X, y.flatten())
            mdi_sorted = sorted(
                zip(map(lambda x: round(x, 4), rgsr.feature_importances_), x_cols), reverse=True)
            features_selected = [p[1] for p in mdi_sorted[:n_features]]

        if method == 'MDA':
            scores = defaultdict(list)
            ss = ShuffleSplit(n_splits=3, test_size=.3)

            for train_idx, test_idx in ss.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                rgsr.fit(X_train, y_train.flatten())
                accu = r2_score(y_test, rgsr.predict(X_test))

                for i in range(X.shape[1]):
                    X_t = X_test.copy()
                    np.random.shuffle(X_t[:, i])
                    shuff_accu = r2_score(y_test, rgsr.predict(X_t))
                    scores[x_cols[i]].append((accu - shuff_accu) / accu)

            mda_sorted = sorted(
                [(round(np.mean(score), 4), feat)for feat, score in scores.items()], reverse=True)
            features_selected = [p[1] for p in mda_sorted[:n_features]]
    
    # 更新时间记录.
    # te = time.time()
    # if method not in time_costs.keys():
    #     time_costs[method] = {}
    # time_costs[method].update({y_col: round(te - ts, 4)})
    # with open('../../file/records/time_cost.json', 'r+') as f:
    #     json.dump(tc, f)
    
    # ---- step 3: 获得总的特征排序表 ----------------------------------------------------------------
    
    features_unselected = set(x_cols).difference(set(features_selected))
    feature_assocs_ = feature_assocs[feature_assocs['feature'].isin(features_unselected)]
    feature_assocs_.sort_values(by=ASSOC_NAME_MAP[method], ascending=False, inplace=True)
    features_ranking = features_selected + feature_assocs_['feature'].to_list()

    print('  FINFISHED: y_col: %s, method: %s, fs num: %d' % (y_col, method, len(features_selected)))
    
    return features_ranking


def save_features(features_ranking: list, y_col, method):
    features_df = pd.DataFrame(features_ranking, columns=['x'])
    features_df.to_csv(
        os.path.join(BASE_DIR, 'file/s2_incremental_test/{}_{}.csv'.format(y_col, method)),
        index=False)


if __name__ == '__main__':
    IRFS_METHODS = ['CRMICM'] + ['JMIM', 'JMI', 'MRMR']
    # IRFS_METHODS = ['CRMICM', 'CMIM'] + ['Pearson', 'Spearman', 'DistCorr', 'MI', 'MIC', 'MDI', 'MDA']
    # IRFS_METHODS = ['JMIM', 'JMI', 'MRMR']

    for method in IRFS_METHODS[:]:
        for y_col in TARGETS[:]:
            print('method: %s, y_col: %s' % (method, y_col))
            features_ranking = exec_feature_selection(y_col, method, N_FEATURES)  # 需要根据s1关联分析确定 N_FEATURES值
            save_features(features_ranking, y_col, method)

