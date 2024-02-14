# -*- coding: utf-8 -*-
"""
Created on 2021/05/24 16:52:30

@File -> pyitlib_mie.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 直接使用pyitlib包计算信息熵
"""

from pyitlib import discrete_random_variable as drv
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np


def cal_mie(x: np.ndarray, y: np.ndarray, n_bins: int = 10):
    # 首先进行离散化.
    enc = KBinsDiscretizer(
        n_bins=n_bins, encode='ordinal', strategy='quantile')
    xs = enc.fit_transform(x.reshape(-1, 1)).astype(np.int)
    ys = enc.fit_transform(y.reshape(-1, 1)).astype(np.int)
    mie = drv.information_mutual(xs.flatten(), ys.flatten())
    return mie


def cal_rho(x: np.ndarray, y: np.ndarray, n_bins: int = 10):
    mie = cal_mie(x, y, n_bins)
    rho = np.sqrt(1 - np.power(2, -2 * mie))
    return rho
