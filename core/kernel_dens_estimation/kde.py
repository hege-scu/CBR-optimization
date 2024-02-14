# -*- coding: utf-8 -*-
"""
Created on 2021/11/24 20:27:18

@File -> kde.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 核密度函数估计
"""

from sklearn.neighbors._kde import KernelDensity
import pandas as pd
import numpy as np


def _normalize(x: np.ndarray):
    x_min, x_max = np.min(x), np.max(x)
    if x_min == x_max:
        return np.ones_like(x)
    else:
        return (x - x_min) / (x_max - x_min)


if __name__ == '__main__':
    import seaborn as sns
    import sys
    import os
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
    sys.path.append(BASE_DIR)
    
    from src.setting import load_data_sets, plt
    from core.association.mic import PairwiseRMIC
    
    _, _, X, y, x_cols = load_data_sets('dry_gas', normal_y=True)
    
    
    # ---- 一维核密度估计 ---------------------------------------------------------------------------
    
    xi = X[:, 217]
    xs = X[:, 70]
    y = y

    samples = np.vstack((xi, y.flatten(), xs)).T
    kde = KernelDensity(bandwidth=0.02).fit(samples)
    arr = kde.sample(10000)
    xi, y, xs = arr[:, 0], arr[:, 1], arr[:, 2]
    
    q = int(len(xs) // 1000)
    xs_encoded = pd.qcut(xs, q, labels=False, duplicates='drop').reshape(-1, 1)
    xs_labels = set(xs_encoded.flatten())
    
    arr = np.hstack((xi.reshape(-1, 1), y.reshape(-1, 1), xs_encoded))
    
    for label in xs_labels:
        arr_sub = arr[np.where(arr[:, 2] == label)[0], :-1]
        xi_sub, y_sub = arr_sub[:, 0], arr_sub[:, 1]
        xi_sub, y_sub = _normalize(xi_sub), _normalize(y_sub)

        plt.figure(label)
        plt.subplot(1, 2, 1)
        plt.scatter(xi_sub, y_sub, s=20, alpha = 0.1)
        # plt.legend([PairwiseRMIC(xi_sub, y_sub).cal_assoc()], loc='upper right')

        # a = np.vstack((xi_sub, y_sub)).T
        # kde = KernelDensity(
        #     algorithm='ball_tree',
        #     kernel='gaussian', 
        #     bandwidth=0.05, 
        #     leaf_size=50,
        #     ).fit(a)
        
        # base_ticks_ = np.arange(0.0, 1.1, 0.1)
        # mesh_a, mesh_b = np.meshgrid(base_ticks_, base_ticks_)
        # ticks = np.vstack((mesh_a.flatten(), mesh_b.flatten())).T
        # dens = np.exp(kde.score_samples(ticks))
        
        # samples = kde.sample(500)
        # x_a, x_b = samples[:, 0], samples[:, 1]
        # plt.subplot(1, 2, 2)
        # plt.scatter(x_a, x_b, s=20, alpha=0.1)
        plt.legend([
            '%d, %.4f' % (len(xi_sub), PairwiseRMIC(xi_sub, y_sub).cal_assoc())
            ], loc='upper right')
    
    