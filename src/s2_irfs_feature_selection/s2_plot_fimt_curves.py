# -*- coding: utf-8 -*-
"""
Created on 2021/11/25 17:11:01

@File -> s2_plot_fimt_curves.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 画出FIMT测试曲线
"""

from matplotlib.ticker import FuncFormatter
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from src.setting import IRFS_METHODS, TARGETS, plt
from src.s2_irfs_feature_selection.setting import N_FEATURES

# IRFS_METHODS = ['CRMICM', 'CMIM'] + ['Pearson', 'Spearman', 'DistCorr', 'MI', 'MIC', 'MDI', 'MDA']
#IRFS_METHODS = IRFS_METHODS[:]
IRFS_METHODS = ['CRMICM', 'JMIM', 'JMI', 'MRMR']

IRFS_METHODS_ALIAS = {
    'CRMICM': 'CRMICM', 'CMIM': 'CMIM', 'JMIM': 'JMIM', 'JMI': 'JMI', 'MRMR': 'MRMR', 
    'Pearson': 'PearsonCorr', 'Spearman': 'SpearmanCorr', 'DistCorr': 'DistCorr', 'MI': 'MI', 
    'MIC': 'MIC', 'MDI': 'MDI', 'MDA': 'MDA'
}

TITLES_MAP = {
    'gasoline': 'gasoline',
    'liquidtotal': 'liquid total',
    'coke': 'coke',
}


def to_percent(y, position):
    return str(int(100 * y)) + '%'  # 这里可以用round（）函数设置取几位小数

if __name__ == '__main__':
    model_name = 'rf'
    metric = 'mse'  # 改变不同的标准进行计算 r2, mape, mse

    print('model: %s, r2: %s' % (model_name, metric))
    
    # ---- 配色方案 ---------------------------------------------------------------------------------

    # cmap = plt.get_cmap('rainbow')
    cmap = plt.get_cmap('RdBu_r')
    cNorm  = colors.Normalize(vmin=0, vmax=len(IRFS_METHODS) + 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    # ---- 画图 ------------------------------------------------------------------------------------
    
    fig, axs = plt.subplots(2, 2, figsize=[10, 5])
    for c, y_col in enumerate(TARGETS):
        c = c // 2, c % 2
        for i, method in enumerate(IRFS_METHODS):
            # 载入FIMT曲线.
            fimt = pd.read_csv(
                os.path.join(BASE_DIR, 'file/s2_incremental_test/fimt_accu_{}_{}_{}_{}.csv'.format(
                    y_col, method, model_name, metric)))
            
            if method in ['CRMICM', 'CMIM']:
                lw = 2.5
            else:
                lw = 1.5
                
            if method in ['CMIM', 'JMI', 'Pearson', 'DistCorr', 'MIC', 'MDA', 'Ridge']:
                ls = '-.'
            else:
                ls = '-'
            
            axs[c].plot(
                fimt['n_features'], 
                fimt[method], 
                ls, 
                label = method,
                linewidth=lw,
                color=scalarMap.to_rgba(len(IRFS_METHODS) - i - 1), 
                alpha=0.8,
                zorder = -i
                )
            
        axs[c].set_xlim([0.0, N_FEATURES])

        if metric == 'r2':
            if model_name == 'rf':
                if y_col in ['gasoline']:
                    axs[c].set_ylim([0.97, 0.99])
                    axs[c].set_yticks([0.97, 0.98, 0.99])
                elif y_col in ['liquidtotal']:
                    axs[c].set_ylim([0.7, 1.0])
                    axs[c].set_yticks([0.97, 0.98, 0.99, 1.0])
                elif y_col in ['coke']:
                    axs[c].set_ylim([0.8, 0.96])
                    axs[c].set_yticks([0.8, 0.84, 0.88, 0.92, 0.96])
                else:
                    axs[c].set_ylim([0.8, 0.95])
                    axs[c].set_yticks([0.8, 0.85, 0.9, 0.95])
        elif metric == 'mse':
            if model_name == 'rf':
                if y_col in ['gasoline']:
                    axs[c].set_ylim([0.0, 0.003])
                    axs[c].set_yticks([0.0, 0.001, 0.002, 0.003])
                elif y_col in ['liquidtotal']:
                    axs[c].set_ylim([0.0, 0.003])
                    axs[c].set_yticks([0.0, 0.001, 0.002, 0.003])
                else:
                    axs[c].set_ylim([0.0, 0.004])
                    axs[c].set_yticks([0.0, 0.001, 0.002, 0.003, 0.004])
        elif metric == 'mape':
            if model_name == 'rf':
                if y_col in ['gasoline']:
                    axs[c].set_ylim([0.0, 0.3])
                    # axs[c].set_yticks([0.0, 0.001, 0.002, 0.003])
                elif y_col in ['liquidtotal']:
                    axs[c].set_ylim([0.00, 0.4])
                    # axs[c].set_yticks([0.97, 0.98, 0.99, 1.0])
                elif y_col in ['coke']:
                    axs[c].set_ylim([0.00, 0.3])
                    # axs[c].set_yticks([0.0, 0.001, 0.002, 0.003])

                
        axs[c].set_xticks([1, 31, 61, 91, 121])  # FIXME:
        axs[c].grid(True, axis='x', linewidth=0.3)
        axs[c].grid(True, axis='y', linewidth=0.3)
        axs[c].set_xlabel('$N_{\\rm F}$', fontsize=16)
        
        if metric == 'r2':
            axs[c].set_ylabel('$\\rm R^2$', fontsize=16)
        elif metric == 'mse':
            axs[c].set_ylabel('MSE', fontsize=16)
        elif metric == 'mape':
            axs[c].set_ylabel('MAPE', fontsize=16)
            axs[c].yaxis.set_major_formatter(FuncFormatter(to_percent))
            
        axs[c].set_title(TITLES_MAP[y_col], fontsize=16)

    fig.tight_layout()
    fig.legend(
        [IRFS_METHODS_ALIAS[p] for p in IRFS_METHODS], 
        bbox_to_anchor=(1.03, 0.0), 
        ncol=len(IRFS_METHODS) // 2, 
        frameon=False, 
        fontsize=14)
    fig.savefig(os.path.join(BASE_DIR, 'img/FIMT_{}_{}.png'.format(model_name, metric)), bbox_inches='tight', dpi=600)
    