# -*- coding: utf-8 -*-
"""
Created on 2021/11/22 16:35:10

@File -> setting.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 项目设置
"""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../'))
sys.path.append(BASE_DIR)

from mod.config.config_loader import config_loader
from mod.data_process.numpy import normalize

PROJ_DIR, PROJ_CMAP = config_loader.proj_dir, config_loader.proj_cmap
plt = config_loader.proj_plt

# 载入项目变量配置.
ENC_CONFIG = config_loader.environ_config
MODEL_CONFIG = config_loader.model_config
TEST_PARAMS = config_loader.test_params

# ---- 定义环境变量 ---------------------------------------------------------------------------------

# ---- 定义模型参数 ---------------------------------------------------------------------------------

# 特征降维方法.
# IRFS_METHODS = ['CRMICM', 'CMIM', 'JMIM', 'JMI', 'MRMR', 'Pearson', 'Spearman', 'DistCorr', 'MI', \
#     'MIC', 'MDI', 'MDA', 'Lasso', 'Ridge']
IRFS_METHODS = ['CRMICM', 'CMIM', 'JMIM', 'JMI', 'MRMR', 'Pearson', 'Spearman', 'DistCorr', 'MI', \
    'MIC', 'MDI', 'MDA']
SSFS_METHODS = ['GA', 'Lasso', 'Ridge']
FE_METHODS = ['PCA', 'KPCA', 'LLE', 'PLS']

TARGETS = ['gasoline', 'liquidtotal', 'coke']

# ---- 定义通用函数 ---------------------------------------------------------------------------------

def _load_raw_data(case: str):
    X_df = pd.read_csv(os.path.join(PROJ_DIR, 'data/raw/{}/X.csv'.format(case)))
    Y_df = pd.read_csv(os.path.join(PROJ_DIR, 'data/raw/{}/Y.csv'.format(case)))
    # return X_df.iloc[750:][:], Y_df.iloc[750:][:]
    return X_df, Y_df


def load_data_sets(y_col: str, normal_y: bool = True, normal_x: bool = True):
    """noise: DESULFUR: 1e-2, FCC: 0.0, TE_CLFN: 0.0"""
    X_df, Y_df = _load_raw_data(case='FCC')
    X_df = X_df.reset_index(drop=True)
    y_df = Y_df[[y_col]].reset_index(drop=True)

    X = X_df.values
    y = y_df.values

    x_cols = X_df.columns.tolist()

    if normal_x:
        X = normalize(X)  # 数据正规化
        X_df = pd.DataFrame(X, columns=x_cols)  # 列名转化为list格式

    if normal_y:
        y = normalize(y)
        y_df = pd.DataFrame(y, columns=[y_col])
    return X_df, y_df, X, y, x_cols


def load_data(y_col: str, normal_y: bool = True, normal_x: bool = True, validation: bool = False):  # 专用于CBR模块的数据加载
    global X_df, Y_df, X_normal, y_normal
    if validation == False:
        X_df = pd.read_csv(os.path.join(PROJ_DIR, 'data/X.csv'))
        Y_df = pd.read_csv(os.path.join(PROJ_DIR, 'data/Y.csv'))
    if validation == True:  # 针对2_2基于范数 划分训练集
        # X_df = pd.read_csv(os.path.join(PROJ_DIR, 'data/X_train.csv'))
        # Y_df = pd.read_csv(os.path.join(PROJ_DIR, 'data/Y_train.csv'))
        X_df = pd.read_csv(os.path.join(PROJ_DIR, 'data/X.csv'))
        Y_df = pd.read_csv(os.path.join(PROJ_DIR, 'data/Y.csv'))

    X_df = X_df.reset_index(drop=True)
    y_df = Y_df[[y_col]].reset_index(drop=True)

    X = X_df.values
    y = y_df.values
    y_origin = y

    dataset = pd.concat([X_df, y_df], axis=1)  # 工艺数据库

    x_cols = X_df.columns.tolist()

    if normal_x:
        # x_scaler = MinMaxScaler()
        # X = x_scaler.fit_transform(X) # 归一化
        X = normalize(X)  # 数据正规化
        X_normal = pd.DataFrame(X, columns=x_cols)  # 列名转化为list格式

    if normal_y:
        y = normalize(y)
        y_normal = pd.DataFrame(y, columns=[y_col])
    daset_df = pd.concat([X_normal, y_normal], axis=1)  # 归一化的工艺数据库

    return dataset, daset_df, X_normal, y_normal, X, y, y_origin


def load_x0(normal_x: bool = True, validationmap: str ='0', y_col: str = 'gasoline'):
    global X0_df,y0_df, y0
    if validationmap =='0': # 用于代码文件 1
        X0_df = pd.read_csv(os.path.join(PROJ_DIR, 'data/X0.csv'))
    if validationmap =='1': # 用于代码文件2_1 和 3_2 若干测试案例验证
        X0_df = pd.read_csv(os.path.join(PROJ_DIR, 'data/X0_validation.csv'))
    if validationmap =='2': # 用于代码文件2_2基于范数图的测试集验证
        X0_df = pd.read_csv(os.path.join(PROJ_DIR, 'data/X0_test.csv'))
        y0_df = pd.read_csv(os.path.join(PROJ_DIR, 'data/Y0_test.csv'))
        y0_df = y0_df[[y_col]].reset_index(drop=True)
        y0 = normalize(y0_df.values)

    X0_df = X0_df.reset_index(drop=True)

    X0 = X0_df.values

    x0_cols = X0_df.columns.tolist()

    if normal_x:
        X0 = normalize(X0)
        X0_df = pd.DataFrame(X0, columns=x0_cols)  # 列名转化为list格式

    return X0_df, X0, x0_cols, y0


def train_test_split(X, y, seed: int = None, test_ratio=0.2):
    X, y = X.copy(), y.copy()
    assert X.shape[0] == y.shape[0]
    assert 0 <= test_ratio < 1

    if seed is not None:
        np.random.seed(seed)
        shuffled_indexes = np.random.permutation(range(len(X)))
    else:
        shuffled_indexes = np.random.permutation(range(len(X)))

    test_size = int(len(X) * test_ratio)
    train_index = shuffled_indexes[test_size:]
    test_index = shuffled_indexes[:test_size]
    return X[train_index], X[test_index], y[train_index], y[test_index]
