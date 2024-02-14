# -*- coding: utf-8 -*-
"""
Template
"""

import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from src.setting import TARGETS
from src.setting import load_data, load_x0
from core.association import cal_rmic


def casebase_setup(x_df, x0_full_df, target='gasoline'):
    # 将读取的工艺数据库通过降维特征转为工艺案例库，并对应地处理待优化工况特征 。 x_df为带字段名的工艺数据库数据集, x0_full_df为待优化工况数据集
    # 由于特征降维与预测目标有关，故工艺案例库提取需要结合预测目标降维字段进行。

    global fields_df
    if target == 'gasoline':
        fields_df = pd.read_excel(os.path.join(BASE_DIR, 'src/s3_CBR_optimization/fileds/casebase_x_gasoline_CRMICM.xlsx'))

    if target == 'liquidtotal':
        fields_df = pd.read_excel(os.path.join(BASE_DIR, 'src/s3_CBR_optimization/fileds/casebase_x_liquidtotal_CRMICM.xlsx'))

    if target == 'coke':
        fields_df = pd.read_excel(os.path.join(BASE_DIR, 'src/s3_CBR_optimization/fileds/casebase_x_coke_CRMICM.xlsx'))

    fields = list(fields_df['index'])
    cols = fields[:]
    xx_df = x_df[cols]
    xx = xx_df.values

    x0_df = x0_full_df[cols]
    x0 = x0_df.values
    return xx, x0


def similarity(x, x0, method='ManH'):
    # x为案例库数据集，x0为待优化匹配记录
    sim = []  # sim = [datarows, colums_feed]

    if method == 'EUCLID':
        for i in range(x.shape[0]):  # 搜索匹配记录
            sim.append([])
            for j in range(x.shape[1]):
                sim[i].append(1 - ((x[i, j] - x0[j]) ** 2))

    if method == 'ManH':
        for i in range(x.shape[0]):  # 搜索匹配记录
            sim.append([])
            for j in range(x.shape[1]):
                maxj = np.max(x[:, j])
                minj = np.min(x[:, j])
                # sim[i][j] = (1 -( abs(x[i,j] - x0[j]) ) / (maxj - minj))
                sim[i].append(1 - (abs(x[i, j] - x0[j])) / (maxj - minj))  # 此处 maxj - minj = 1

    return sim


def simtotal_lims(sim, weight, count_lims):
    simm, weightt = sim[:, :count_lims], weight[:count_lims]
    simiarity_total = np.dot(simm, weightt)
    return simiarity_total


def simtotal_dcs(sim, weight, count_l, count_h):
    simm, weightt = sim[:, count_l + 1:count_h], weight[count_l + 1:count_h]
    simiarity_total = np.dot(simm, weightt)
    return simiarity_total


def simtotal_compute(sim, weight, variable_count, stage='s0', cal_mode='ManH'):
    global sim_total
    count0 = variable_count[0]  # lims变量数
    count1 = variable_count[0] + variable_count[1]  # lims、反再DCS变量数
    count2 = variable_count[0] + variable_count[1] + variable_count[2]  # lims、反再和分馏DCS变量数

    sim = np.array(sim)
    if stage == 's0':  # 优化全工段
        sim_total = simtotal_lims(sim, weight, count0 - 1)

    if stage == 's1':  # 优化分馏及吸收稳定工段
        sim_total1 = simtotal_lims(sim, weight, count0 - 1)  # lims
        sim_total2 = simtotal_dcs(sim, weight, count0 - 1, count1 - 1)  # dcs

        sim_total = 0.7 * sim_total1 + 0.3 * sim_total2  # 0.7和0.3为衡量lims和DCS变量匹配区分重要性的权重

    if stage == 's2':  # 优化吸收稳定工段
        sim_total1 = simtotal_lims(sim, weight, count0 - 1)  # lims
        sim_total2 = simtotal_dcs(sim, weight, count0 - 1, count2 - 1)  # dcs

        sim_total = 0.7 * sim_total1 + 0.3 * sim_total2  # 0.7和0.3为衡量lims和DCS变量匹配区分重要性的权重

    if cal_mode == 'EUCLID':
        sim_total = np.sqrt(sim_total)

    return sim_total


if __name__ == '__main__':

    stage_optimization = ['s0', 's1', 's2']  # S0表示反再进料匹配得全DCS变量；S1表示分馏前DCS和物料匹配得分馏和吸收稳定DCS变量；S2类推
    sim_distance = ['EUCLID', 'ManH']  # 分别表示基于欧式距离和曼哈顿距离的相似度计算
    variable_count = [9, 40, 20]  # 分别为降维选择后的 原料LIMS物性变量数、反再工段DCS变量数、分馏工段DCS变量数

    for stage in stage_optimization[:]:
        for y_col in TARGETS[:]:  # 遍历优化目标

            dataset, dataset_df, x_df, y_df, X, y, y_origin = load_data(y_col, normal_y=True)  # dataset为归一化前原始工艺数据（输入输出）
            x0_full_df, x0_full, x0_cols, y0 = load_x0()  # 读取待匹配物料及工况(多条记录)

            xx, x0 = casebase_setup(x_df, x0_full_df, y_col)  # 案例数据库及待优化工况数据集

            for method in sim_distance[:]:
                for i in range(x0.shape[0]):
                    # 将sim_total、weight、dataset_total、dataset_opt、y_best、dataset_best初始化
                    sim_total = []
                    weight = []
                    dataset_total = []
                    dataset_opt = []
                    y_best = []
                    dataset_best = []

                    xx0 = x0[i]  # 取某条待匹配记录
                    sim = similarity(xx, xx0, method)

                    for j in range(xx.shape[1]):  # 计算案例库数据变量对应的权重
                        weight.append(cal_rmic(xx[:, j], y.flatten(), x_type='numeric'))  # 计算权重或者提供权重信息

                    sim_total = simtotal_compute(sim, weight, variable_count, stage, method)

                    thres = sorted(sim_total)[int(len(sim_total) * 0.98)]  # 设定 相似度筛选阈值， 此处取记录集 相似度结果的98%位数值
                    sim_total = pd.DataFrame(sim_total, columns=['sim'])

                    dataset_total = pd.concat([dataset, sim_total], axis=1)
                    dataset_opt = dataset_total[dataset_total['sim'] > thres]  # 筛选相似度满足大于设定阈值的工况数据集

                    if y_col == 'coke':
                        y_best = min(dataset_opt[y_col].unique())
                        dataset_best = pd.DataFrame(dataset_opt[dataset_opt[y_col] <= y_best])  # 得到最优工况所在数据集
                    else:
                        y_best = max(dataset_opt[y_col].unique())
                        dataset_best = pd.DataFrame(dataset_opt[dataset_opt[y_col] >= y_best])  # 得到最优工况所在数据集

                    print('optimizaiton result: stage: %s, target: %s, similarity mode: %s, num: %d' % ( stage, y_col, method, i + 1))

                    if i == 0:  # 表示首次插入数据，需要带列名
                        # mode='a+'，追加写入数据
                        dataset_best.to_csv(os.path.join(BASE_DIR, 'file/s3_optimization/optimization_{}_{}_{}.csv'.
                                                         format(stage, y_col, method)), index=False, mode='a+', header=True)
                    else:
                        dataset_best.to_csv(os.path.join(BASE_DIR, 'file/s3_optimization/optimization_{}_{}_{}.csv'.
                                                         format(stage, y_col, method)), index=False, mode='a+', header=False)

    print("CBR优化执行成功！")
