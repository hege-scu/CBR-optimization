# -*- coding: utf-8 -*-
"""
Created on 2021/08/20 10:44:34

@File -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 初始化
"""

__all__ = [
    'X_TYPES', 'ASSOC_METHODS',
    'cal_dist_corr', 'cal_pearson_corr', 'cal_spearman_corr', 'cal_mi', 'cal_mic', 'cal_rmic',
    'cal_time_delayed_assocs'
    ]

from pyitlib import discrete_random_variable as drv
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
import copy
import dcor

from .mi import PairwiseMI
from .mic import PairwiseMIC, PairwiseRMIC
from .univar_encoding import SuperCategorEncoding

X_TYPES = ['numeric', 'categoric']
ASSOC_METHODS = ['dist_corr', 'pearson_corr', 'spearman_corr', 'mi', 'mic', 'rmic']


# ---- 数据离散化 -----------------------------------------------------------------------------------

def discretize_series(x: np.ndarray, n: int = 100, method='qcut'):
    """对数据序列采用等频分箱"""
    q = int(len(x) // n)

    # ---- TODO: 确定分箱方式

    if method == 'qcut':
        x_enc = pd.qcut(x, q, labels=False, duplicates='drop').flatten()  # 等频分箱
    if method == 'cut':
        x_enc = pd.cut(x, q, labels=False, duplicates='drop').flatten()  # 等宽分箱

    # ----
    return x_enc


def discretize_arr(X: np.ndarray, n: int = 100, method: str = 'qcut'):
    """逐列离散化"""
    X = X.copy()
    for i in range(X.shape[1]):
        X[:, i] = discretize_series(X[:, i], n, method)
    return X.astype(int)


# ---- 计算关联度 -----------------------------------------------------------------------------------

# 这里所有函数统一写作func(x, y, x_type)的格式.

def cal_dist_corr(x, y, x_type: str='numeric'):
    """距离相关系数"""
    if x_type == 'categoric':
        x = _encode(x, y)
    return np.abs(dcor.distance_correlation(x, y))


def cal_pearson_corr(x, y, x_type: str='numeric'):
    """Pearson相关系数"""
    if x_type == 'categoric':
        x = _encode(x, y)
    return np.abs(pearsonr(x, y)[0])


def cal_spearman_corr(x, y, x_type: str='numeric'):
    """Pearson相关系数"""
    if x_type == 'categoric':
        x = _encode(x, y)
    return np.abs(spearmanr(x, y)[0])


def cal_mi(x, y, x_type: str='numeric'):
    if x_type == 'categoric':
        x = _encode(x, y)
    # return PairwiseMI(x, y).cal_assoc()

    x = discretize_series(x, n=80, method='qcut')
    y = discretize_series(y, n=80, method='qcut')
    return drv.information_mutual(x.astype(int), y.astype(int))
    

def cal_mic(x, y, x_type: str='numeric'):
    if x_type == 'categoric':
        x = _encode(x, y)
    return PairwiseMIC(x, y).cal_assoc()


def cal_rmic(x, y, x_type: str='numeric'):    #计算RMIC
    if x_type == 'categoric':
        x = _encode(x, y)
    return PairwiseRMIC(x, y).cal_assoc()


def _encode(x, y): 
    """如果x是类别型变量, 则对x进行编码
    注意: 这里选择有监督的编码,因此入参有y, 其他编码方式可以在univar_encoding里选择
    """
    super_enc = SuperCategorEncoding(x, y)
    x_enc = super_enc.mhg_encoding()
    return x_enc


# ---- 计算时滞关联 ----------------------------------------------------------------------------------

def cal_time_delayed_assocs(x, y, td_lags: list, x_type: str, method: str, **kwargs):
    """计算时滞关联"""
    assert method in ASSOC_METHODS
    if method == 'dist_corr':
        cal_assoc = cal_dist_corr
    if method == 'pearson_corr':
        cal_assoc = cal_pearson_corr
    if method == 'spearman_corr':
        cal_assoc = cal_spearman_corr
    if method == 'mi':
        cal_assoc = cal_mi
    if method == 'mic':
        cal_assoc = cal_mic
    if method == 'rmic':
        cal_assoc = cal_rmic
        
    td_assocs = []
    arr = np.vstack((x, y)).T
    for i, td_lag in enumerate(td_lags):
        print('%{:.2f}\r'.format(i / len(td_lags) * 100), end = '')
        
        # 注意x, y长度随td_lag变化.
        x_td, y_td = _gen_time_delayed_series(arr, td_lag, **kwargs)
        
        assoc = cal_assoc(x_td, y_td, x_type)
        td_assocs.append(assoc)
    
    return td_lags, td_assocs
        

def _gen_time_delayed_series(arr: np.ndarray, td_lag: int, max_len: int = 3000):
    """
    生成时间延迟序列, 按照td_lag的正负分别平移x或y
    :param arr: 样本数组, shape = (N, D = 2), 第一列为x, 第二列为y
    :param td_lag: 时间平移样本点数, 若td_lag > 0, 则x对应右方td_lag个样本点后的y;
                               若td_lag < 0, 则y对应右方td_lag个样本点后的x
    """
    arr = arr.copy()

    x_td = arr[:, 0]
    y_td = arr[:, 1]
    
    lag_remain = np.abs(td_lag) % arr.shape[0]  # 整除后的余数, 解决过远的周期平移的问题
    if lag_remain == 0:
        pass
    else:
        if td_lag > 0:
            y_td = y_td[lag_remain:]
            x_td = x_td[:-lag_remain]
        else:
            x_td = x_td[lag_remain:]
            y_td = y_td[:-lag_remain]

    # 如果数据量太大则进行采样计算.
    if len(x_td) > max_len:
        start_loc = np.random.randint(0, len(x_td) - max_len)
        x_td, y_td = x_td[start_loc: start_loc + max_len], y_td[start_loc: start_loc + max_len]
    
    return x_td, y_td
        
    