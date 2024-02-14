# -*- coding: utf-8 -*-
"""
Template
  用于CBR优化效果有效性验证
  数据样本 10个分类，每个分类取5个数据点，总计50个；
  选取分馏工段，优化目标（汽油、总液收、焦炭）3个，2个距离度量方式；
  最后考察K近邻方式得到每个待优化点相邻的100个优化目标值分布与CBR优化方式得到的最优目标值进行比较。
"""
from matplotlib.ticker import FuncFormatter
import pandas as pd
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from src.setting import TARGETS, plt

titles_map = {
    'gasoline': 'Gasoline',
    'liquidtotal': 'Liquid total',
    'coke': 'Coke',
}


def to_percent(y, position):
    return str(int(100 * y)) + '%'  # 这里可以用round（）函数设置取几位小数

if __name__ == '__main__':
    fig, axs = plt.subplots(1, 3, figsize=[10, 2.5])

    method = 'EUCLID'  # ****注意，此处改变分别执行一次才能得到全部图 'EUCLID','ManH'
    sorts = ['nearest', 'best']

    for c, y_col in enumerate(TARGETS):  # 遍历优化目标
        for i, sort in enumerate(sorts):
            # 载入散点图.
            data = pd.read_csv(
                os.path.join(BASE_DIR, 'file/s3_optimization/optim_valid_{}_{}_{}.csv'.format(sort, y_col, method)))

            marker, color, alpha, s = '', '', 0, 0
            if sort == 'nearest':
                marker, color, alpha, s = 'o', 'b', 0.3, 1
            if sort == 'best':
                marker, color, alpha, s = '*', 'r', 1, 10

            axs[c].scatter(
                data['index'],
                data['target'],
                s = s,
                marker=marker,
                label=sort,
                linewidth=0.2,
                color=color,
                alpha=alpha,
                zorder=i  # zorder高的元素显示在zorder低低元素之上
            )

            axs[c].set_xticks([0, 10, 30, 50])
            axs[c].grid(True, axis='x', linewidth=0.3)
            axs[c].set_xlim([-0.1, 51.0])
            if y_col =='gasoline':
                axs[c].set_ylim([0.3, 0.6])
            if y_col =='liquidtotal':
                axs[c].set_ylim([0.6, 1.0])
            if y_col =='coke':
                axs[c].set_ylim([-0.1, 0.4])
            axs[c].set_ylabel('Yield')
            # axs[c].set_yticks([0.0, 0.1, 0.2, 0.3])
            axs[c].set_xlabel('$\\rm Samples$')
            axs[c].set_title(titles_map[y_col], fontsize=14)
            axs[c].yaxis.set_major_formatter(FuncFormatter(to_percent))

        fig.tight_layout()

    fig.legend(['nearest point'] + ['best point'], bbox_to_anchor=(1.15, 0.9), ncol=1, frameon=False, fontsize=14)
    fig.savefig(os.path.join(BASE_DIR, 'img/optim_valid_{}.png'.format(method)), bbox_inches='tight', dpi=600)

    print("CBR优化有效性验证图绘制成功！")