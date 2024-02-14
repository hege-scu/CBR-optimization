# -*- coding: utf-8 -*-
"""
Created on 2021/09/06 15:15:06

@File -> s2_rmic_assoc_coeff_plot.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 
"""

import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd
import numpy as np
import copy
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from src.setting import TARGETS
from src.setting import load_data_sets, plt
from src.s1_association_analysis.setting import ASSOC_METHODS

titles_map = {
    'gasoline': 'gasoline',
    'liquidtotal': 'liquid total',
    'coke': 'coke',
}

methods = ['dist_corr', 'pearson_corr', 'spearman_corr', 'mi','mic', 'rmic']
methods_alias = ['DistCorr', 'PearsonCorr', 'SpearmanCorr', 'MI', 'MIC', 'RMIC']

if __name__ == '__main__':
    result_idx=0  # 控制这里的idx来选用f1或accu画图

    cmap = plt.get_cmap('RdBu_r')
    cNorm  = colors.Normalize(vmin=1, vmax=len(ASSOC_METHODS))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    fig, axs = plt.subplots(1, 3, figsize=[10, 2.5])
    for c, y_col in enumerate(TARGETS):
        X_df, _, _, y, _ = load_data_sets(y_col, normal_y=True)
        assoc_df = pd.read_csv(os.path.join(BASE_DIR, 'file/s1_assoc_analysis/assoc_df_{}.csv'.format(y_col)))
        
        # 重新排序.
        assoc_df.sort_values(by='rmic', ascending=False)

        lw = 0.5

        for i, method in enumerate(methods):
            axs[c].scatter(
                assoc_df['rmic'].values,
                assoc_df[method].values, 
                # ls,
                marker = 'o',
                s = 10,
                label = method, 
                linewidth=lw,
                color=scalarMap.to_rgba(i + 1), 
                alpha = 0.8
                )

            axs[c].set_xticks([0, 0.5, 1.0])
            axs[c].set_xlim([0.0, 1.0])
            axs[c].set_ylim([0.0, 1.5])
            axs[c].set_yticks([0.0, 0.75, 1.5])
            axs[c].set_xlabel('RMIC value')
            axs[c].set_ylabel('coefficient value')
            axs[c].set_title(titles_map[y_col], fontsize=14)

        # break

        fig.tight_layout()
    fig.legend(methods_alias, bbox_to_anchor=(1.2, 0.9), ncol=1, frameon=False, fontsize=14)

    fig.savefig(os.path.join(BASE_DIR, 'img/assoc_FIMT_b.png'), bbox_inches='tight', dpi=600)
