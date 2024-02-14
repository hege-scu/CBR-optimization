# -*- coding: utf-8 -*-
"""
Template
  用于CBR优化效果有效性验证
  数据样本 10个分类，每个分类取5个数据点，总计50个；
  选取分馏工段，优化目标（汽油、总液收、焦炭）3个，2个距离度量方式；
  最后考察K近邻方式得到每个待优化点相邻的100个优化目标值分布与CBR优化方式得到的最优目标值进行比较。
"""

import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from src.setting import TARGETS, plt
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
    xx_map_df = x_df[cols].reset_index(drop=True)

    x0_df = x0_full_df[cols]
    x0 = x0_df.values
    x0_obs_df = x0_full_df[cols].reset_index(drop=True)
    x0_obs = x0_obs_df.values
    return xx, xx_df, x0, xx_map_df, x0_obs # xx, xx_df, x0为案例库及待优化工况的待匹配参数；xx_map, x0_obs为对应的匹配工艺参数结果


def similarity(x,x0,method='ManH'):
    # x为案例库数据集，x0为待优化匹配记录
    sim = []  # sim = [datarows, colums_feed]

    if method == 'EUCLID':
        for i in range(x.shape[0]):  # 搜索匹配记录
            sim.append([])
            for j in range(x.shape[1]):
                sim[i].append( 1 - ((x[i, j] - x0[j])**2))

    if method == 'ManH':
        for i in range(x.shape[0]):  # 搜索匹配记录
            sim.append([])
            for j in range(x.shape[1]):
                maxj = np.max(x[:, j])
                minj = np.min(x[:, j])
                # sim[i][j] = (1 -( abs(x[i,j] - x0[j]) ) / (maxj - minj))
                sim[i].append(1 - (abs(x[i, j] - x0[j])) / (maxj - minj))  # 此处 maxj - minj = 1

    return sim


def simtotal_lims(sim,weight,count_lims):
    simm, weightt = sim[:, :count_lims], weight[:count_lims]
    simiarity_total = np.dot(simm, weightt)
    return simiarity_total


def simtotal_dcs(sim,weight,count_l,count_h):
    simm, weightt = sim[:, count_l + 1:count_h], weight[count_l + 1:count_h]
    simiarity_total = np.dot(simm, weightt)
    return simiarity_total


def simtotal_compute(sim, weight, variable_count, stage='s0', cal_mode='ManH'):
    global sim_total
    count0 = variable_count[0] # lims变量数
    count1 = variable_count[0] + variable_count[1] # lims、反再DCS变量数
    count2 = variable_count[0] + variable_count[1] + variable_count[2] # lims、反再和分馏DCS变量数

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
    stage = 's1'  # 选取分馏工段
    sim_distance=['EUCLID','ManH']  # 分别表示基于欧式距离和曼哈顿距离的相似度计算
    variable_count = [9, 40, 20]    # 分别为降维选择后的 原料LIMS物性变量数、反再工段DCS变量数、分馏工段DCS变量数

    for y_col in TARGETS[:]:  # 遍历优化目标
        weight = []
        dataset, dataset_df, x_df, y_df, X, y, y_origin = load_data(y_col, normal_y=True) # dataset_df为 归一化后的原始数据集

        x0_full_df, x0_full, x0_cols, y0 = load_x0(True, '2')   # 读取待匹配物料及工况(多条记录),注意是验证用文件

        xx, xx_df, x0, xx_map_df, x0_obs = casebase_setup(x_df, x0_full_df, y_col)  # 案例数据库及待优化工况数据集

        for j in range(xx.shape[1]):  # 计算案例库数据变量对应的权重
            weight.append(cal_rmic(xx[:, j], y.flatten(), x_type='numeric'))  # 计算权重或者提供权重信息

        for method in sim_distance[:]:
            out_arr = np.zeros((1,2))
            out_best = pd.DataFrame(out_arr, columns= ['index','target'])

            for i in range(x0.shape[0]):
                xx0 = x0[i]  # 取某条待匹配记录
                sim = similarity(xx, xx0, method)

                sim_total = simtotal_compute(sim, weight, variable_count, stage, method)

                thres = sorted(sim_total)[int(len(sim_total) * 0.98)]  # 设定 相似度筛选阈值， 此处取记录集 相似度结果的98%位数值
                sim_total = pd.DataFrame(sim_total , columns=['sim'])

                dataset_norm = pd.concat([xx_map_df, y_df], axis=1)
                dataset_total = pd.concat([dataset_norm, sim_total], axis=1)
                dataset_opt = dataset_total[dataset_total['sim'] > thres]  # 筛选相似度满足大于设定阈值的工况数据集

                if y_col =='coke':
                    y_best = min(dataset_opt[y_col].unique())
                    dataset_best = pd.DataFrame(dataset_opt[dataset_opt[y_col] <= y_best])  # 得到最优工况所在数据集
                else:
                    y_best = max(dataset_opt[y_col].unique())
                    dataset_best = pd.DataFrame(dataset_opt[dataset_opt[y_col] >= y_best])  # 得到最优工况所在数据集

                dataset_best = dataset_best.drop(columns=[y_col, 'sim'])
                delta_score = y_best - y0[i]    # delta_score: 案例库搜索最优收率与测试样本收率之差, Y* - Y_obs
                theta_map = np.linalg.norm(dataset_best)  # theta_map: 就是案例库搜索得最优的工艺变量theta_*的范数值
                theta_obs = np.linalg.norm(x0_obs[i])       # theta_obs: 每个测试样本的工艺变量theta_obs的范数值

                #  dist: 测试实际工艺向量theta与案例库搜索最优工艺theta*之差的二范数值, ||theta_obs - theta*||
                dist = np.linalg.norm(x0_obs[i] - dataset_best)

                dist = pd.DataFrame([dist], columns=['dist'])
                delta_score = pd.DataFrame([delta_score], columns=['delta_score'])
                theta_map = pd.DataFrame([theta_map], columns=['theta_map'])
                theta_obs = pd.DataFrame([theta_obs], columns=['theta_obs'])
                out_best_i = pd.concat([dist, delta_score,theta_map,theta_obs], axis=1)

                if i ==0:
                    out_best = out_best_i
                else:
                    out_best = pd.concat([out_best, out_best_i])

                print('optimizaiton result: target: %s, similarity mode: %s, num: %d' % (y_col, method, i + 1))

            out_best.to_csv(os.path.join(BASE_DIR, 'file/s3_optimization/optim_valid_result_{}_{}.csv'.
                                    format(y_col, method)), index=False, mode='a+', header=True)

    print("CBR优化有效性验证执行成功！")