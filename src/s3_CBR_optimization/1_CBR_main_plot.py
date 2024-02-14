# -*- coding: utf-8 -*-
"""
Template
  用于CBR优化效果有效性验证
  数据样本 10个分类，每个分类取5个数据点，总计50个；
  选取分馏工段，优化目标（汽油、总液收、焦炭）3个，2个距离度量方式；
  最后考察K近邻方式得到每个待优化点相邻的100个优化目标值分布与CBR优化方式得到的最优目标值进行比较。
"""
import matplotlib.colors as colors
import matplotlib.cm as cmx
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

color_rgb = ['red', 'yellow', 'blue']


def to_percent(y, position):
    return str(int(100 * y)) + '%'  # 这里可以用round（）函数设置取几位小数


def factor_plot(plt, stages, method, factors, x0):
    for k, factor in enumerate(factors): #按工艺变量绘图
        fig, axs = plt.subplots(1, 3, figsize=[10, 2.5])
        for c, y_col in enumerate(TARGETS):  # 遍历优化目标
            for i, stage in enumerate(stages):
                # 载入曲线图.
                data = pd.read_csv(os.path.join(BASE_DIR,
                                 'file/s3_optimization/optimization_{}_{}_{}.csv'.format(stage, y_col, method)))

                index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 提供样本序号

                if i == 0:
                    axs[c].plot(
                        index,
                        x0[factor],
                        ls='-',
                        label='origin',
                        linewidth=2.0,
                        color='g',
                        alpha=0.8,
                        zorder=5  # zorder高的元素显示在zorder低低元素之上
                    )

                axs[c].plot(
                    index,
                    data[factor],
                    ls='-',
                    label=stage,
                    linewidth=2.0,
                    color= color_rgb[i],    # scalarMap.to_rgba(i+1),
                    alpha=0.8,
                    zorder=i  # zorder高的元素显示在zorder低低元素之上
                )

                axs[c].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                axs[c].grid(True, axis='x', linewidth=0.3)
                axs[c].set_xlim([1, 10])
                if factor == 'TIC-3204':  # 提升管原料预热温度
                    axs[c].set_ylim([180, 210])
                    axs[c].set_ylabel('Temperature, ℃')
                if factor == 'TIC-3201':  # 分馏塔塔顶温度
                    axs[c].set_ylim([100, 130])
                    axs[c].set_ylabel('Temperature, ℃')
                if factor == 'PIC-3101':  # 第一再生器压力
                    axs[c].set_ylim([0.1, 0.3])
                    axs[c].set_ylabel('Pressure, MPa')

                # axs[c].set_yticks([0.0, 0.1, 0.2, 0.3])
                axs[c].set_xlabel('$\\rm Samples$')
                axs[c].set_title(factor, fontsize=14)

            fig.tight_layout()

        fig.legend(['origin'] + stages, bbox_to_anchor=(1.1, 0.9), ncol=1, frameon=False, fontsize=14)
        fig.savefig(os.path.join(BASE_DIR, 'img/optim_analysis_{}_{}.png'.format(method, factor)), bbox_inches='tight', dpi=600)


def yield_plot(plt, stages, method, Y0):
    fig, axs = plt.subplots(1, 3, figsize=[10, 2.5])

    for c, y_col in enumerate(TARGETS):  # 遍历优化目标
        for i, stage in enumerate(stages):
            # 载入曲线图.
            data = pd.read_csv(
                os.path.join(BASE_DIR, 'file/s3_optimization/optimization_{}_{}_{}.csv'.format(stage, y_col, method)))

            index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 提供样本序号

            if i == 0:
                axs[c].plot(
                    index,
                    Y0[y_col],
                    ls='-',
                    label='origin',
                    linewidth=2.0,
                    color='g',
                    alpha=0.8,
                    zorder=5  # zorder高的元素显示在zorder低低元素之上
                )

            axs[c].plot(
                index,
                data[y_col],
                ls = '-',
                label=stage,
                linewidth=2.0,
                color=color_rgb[i],    # scalarMap.to_rgba(i+1),
                alpha=0.8,
                zorder= i  # zorder高的元素显示在zorder低低元素之上
            )

            axs[c].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            axs[c].grid(True, axis='x', linewidth=0.3)
            axs[c].set_xlim([1, 10.0])
            if y_col =='gasoline':
                axs[c].set_ylim([0.35, 0.6])
            if y_col =='liquidtotal':
                axs[c].set_ylim([0.75, 1.0])
            if y_col =='coke':
                axs[c].set_ylim([-0.05, 0.2])
            axs[c].set_ylabel('Yield')
            # axs[c].set_yticks([0.0, 0.1, 0.2, 0.3])
            axs[c].set_xlabel('$\\rm Samples$')
            axs[c].set_title(titles_map[y_col], fontsize=14)
            axs[c].yaxis.set_major_formatter(FuncFormatter(to_percent))

        fig.tight_layout()

    fig.legend(['origin'] + stages, bbox_to_anchor=(1.1, 0.9), ncol=1, frameon=False, fontsize=14)
    fig.savefig(os.path.join(BASE_DIR, 'img/optim_analysis_{}_{}.png'.format(method, 'yield')), bbox_inches='tight', dpi=600)


if __name__ == '__main__':
    method = 'ManH'  # ****注意，此处改变分别执行一次才能得到全部图 'EUCLID','ManH'
    stages = ['s0', 's1', 's2']
    # 关键工艺调控量，并结合特征权重最终选取的这三个进行工艺影响性分析
    factors = ['TIC-3204', 'TIC-3201', 'PIC-3101']  # 分别是提升管原料预热温度0.46，分馏塔塔顶温度0.395, 第一再生器压力0.496

    cmap = plt.get_cmap('RdBu_r')
    cNorm = colors.Normalize(vmin=1, vmax=len(stages))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    x0 = pd.read_csv(os.path.join(BASE_DIR, 'data/X0.csv'))
    Y0 = pd.read_csv(os.path.join(BASE_DIR, 'data/Y0.csv'))

    yield_plot(plt, stages, method, Y0)  # 绘制收率曲线
    factor_plot(plt, stages, method, factors, x0)  # 绘制影响因素曲线
    print("CBR优化有效性验证图绘制成功！")