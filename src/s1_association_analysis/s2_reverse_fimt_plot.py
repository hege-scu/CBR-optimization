# -*- coding: utf-8 -*-
"""
Created on 2021/08/22 16:13:20

@File -> s2_plot.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 画图
"""

from matplotlib.ticker import FuncFormatter
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd
import numpy as np
import copy
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from src.setting import TARGETS, plt
from src.s1_association_analysis.setting import ASSOC_METHODS

titles_map = {
    'gasoline': 'gasoline',
    'liquidtotal': 'liquid total',
    'coke': 'coke',
}

methods_alias = ['DistCorr', 'PearsonCorr', 'SpearmanCorr', 'MI', 'MIC', 'RMIC']


def to_percent(y, position):
    return str(int(100 * y)) + '%'  # 这里可以用round（）函数设置取几位小数

if __name__ == '__main__':
    cmap = plt.get_cmap('RdBu_r')
    cNorm  = colors.Normalize(vmin=1, vmax=len(ASSOC_METHODS))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    result_idx = 2  #设定不同的值 0, 1, 2，得到不同的准确度评价标准
    name = 'rf'
    fig, axs = plt.subplots(1, 3, figsize=[10, 2.5])
    for c, y_col in enumerate(TARGETS):
        for i, method in enumerate(ASSOC_METHODS):
            # 载入FIMT曲线.
            fimt = pd.read_csv(
                os.path.join(BASE_DIR, 'file/s1_assoc_analysis/fimt_accu_{}_{}_{}_{}.csv'.format(name, y_col, method, result_idx)))
            
            ls = '-'
            lw = 3.0
            
            axs[c].plot(
                fimt['n_features'], 
                fimt[method], 
                ls, 
                label = method, 
                linewidth=lw,
                color=scalarMap.to_rgba(i + 1), 
                alpha=0.8,
                zorder = i
                )
            
            axs[c].set_xticks([0, 10, 20, 30])
            axs[c].grid(True, axis='x', linewidth=0.3)
            axs[c].set_xlim([0.0, 30.0])

            if result_idx == 0:
                axs[c].set_ylim([-0.0, 1.0])
                axs[c].set_ylabel('$\\rm R^2$')
            if result_idx == 1:
                axs[c].set_ylabel('$\\rm MSE$')
                if y_col in ['gasoline']:
                    axs[c].set_ylim([-0.0, 0.12])
                if y_col in ['liquidtotal']:
                    axs[c].set_ylim([-0.0, 0.02])
                    axs[c].set_yticks([0.0, 0.01, 0.02])
                if y_col in ['coke']:
                    axs[c].set_ylim([-0.0, 0.02])
                    axs[c].set_yticks([0.0, 0.01, 0.02])
            if result_idx == 2:
                axs[c].set_ylabel('$\\rm MAPE$')
                if y_col in ['gasoline']:
                    axs[c].set_ylim([0.0, 2.0])
                    axs[c].set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
                if y_col in ['liquidtotal']:
                    axs[c].set_ylim([0.1, 0.4])
                    axs[c].set_yticks([0.1, 0.2, 0.3, 0.4])
                if y_col in ['coke']:
                    axs[c].set_ylim([0.0, 0.3])
                    axs[c].set_yticks([0.0, 0.1, 0.2, 0.3])
            
            axs[c].set_xlabel('$N_{\\rm F}$')
            axs[c].set_title(titles_map[y_col], fontsize=14)
            
        if result_idx == 2:
            axs[c].yaxis.set_major_formatter(FuncFormatter(to_percent))
            
        fig.tight_layout()
        
    fig.legend(methods_alias, bbox_to_anchor=(1.2, 0.9), ncol=1, frameon=False, fontsize=14)
    fig.savefig(os.path.join(BASE_DIR, 'img/assoc_FIMT_a_{}_{}.png'.format(name, result_idx)), bbox_inches='tight', dpi=600)