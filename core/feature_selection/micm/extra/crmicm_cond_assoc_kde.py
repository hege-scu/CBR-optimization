# -*- coding: utf-8 -*-
"""
Created on 2021/11/19 17:34:13

@File -> crmicm.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: CRMICM
"""

from sklearn.neighbors import KernelDensity
from collections import defaultdict
import pandas as pd
import numpy as np

# from ...association.mic import PairwiseRMIC
from ...association.mic import PairwiseRMIC
from ...data_encoding.univar_encoding import encode


# ---- 关联系数计算 ---------------------------------------------------------------------------------

def _random_sample(*x, N_limit: int):
    N = len(x[0])
    if N > N_limit:
        x_new = []
        idxs = np.random.permutation(range(N))[: N_limit]
        for _x in enumerate(x):
            x_new.append(_x.copy()[idxs])
        x_new = tuple(x_new)
    else:
        x_new = x
    return x_new


def cal_assoc(x, y):
    # NOTE: RMIC可能计算速度慢, 这里先进行随机采样.
    x, y = _random_sample(x, y, N_limit=3000)
    return PairwiseRMIC(x, y).cal_assoc()

# ---- 数据重采样 -----------------------------------------------------------------------------------

def _normalize(x: np.ndarray):
    x_min, x_max = np.min(x), np.max(x)
    if x_min == x_max:
        return np.ones_like(x)
    else:
        return (x - x_min) / (x_max - x_min)


def _kde_resample(xi, xs, y, bw: float=0.02, N: int=10000):
    # 重归一化到[0, 1]区间.
    xi, xs, y = _normalize(xi.flatten()), _normalize(xs.flatten()), _normalize(y.flatten())
    samples = np.vstack((xi, xs, y)).T
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(samples)
    kde_samples = kde.sample(N)
    return kde_samples[:, 0], kde_samples[:, 1], kde_samples[:, 2]


# ---- 条件关联系数计算 ------------------------------------------------------------------------------

class CondAssoc(object):
    """条件关联系数"""

    def __init__(self, xi: np.ndarray, xs: np.ndarray, y: np.ndarray):
        self.xi = xi.flatten()
        self.xs = xs.flatten()
        self.y = y.flatten()
        self.N = len(self.xi)
        
        if self.N < 2000:
            self.xi, self.xs, self.y = _kde_resample(self.xi, self.xs, self.y)
        
    @staticmethod
    def _binning_xs(xs):
        """在xs维数上进行离散化"""
        xs = xs.copy()

        # >>>
        # xs_encoded = encode(xs).reshape(-1, 1)
        # ---
        q = int(len(xs) // 1000)
        xs_encoded = pd.qcut(xs, q, labels=False, duplicates='drop').reshape(-1, 1)
        # <<<
        xs_labels = set(xs_encoded.flatten())
        return xs_encoded, xs_labels

    def cal_cond_assoc(self) -> float:
        """计算条件关联系数: Assoc(xi; y | xs)
        :param method: 求解方式, 'mi', 'mic'或'rmic'
        :param x_type: x的数值类型
        """
        xs_encoded, xs_labels = self._binning_xs(self.xs)
        arr = np.hstack((self.xi.reshape(-1, 1), self.y.reshape(-1, 1), xs_encoded))
        
        # FIXME: 删除xs样本数少的数据.
        # for label in xs_labels:
        #     samples_n = arr[arr[:, 2] == label].shape[0]
        #     if samples_n < 20:
        #         arr = arr[arr[:, 2] != label]
        # xs_labels = set(arr[:, 2])

        gain = 0.0
        for label in xs_labels:
            arr_sub = arr[np.where(arr[:, 2] == label)[0], :-1]
            prob = arr_sub.shape[0] / arr.shape[0]
            xi_sub, y_sub = arr_sub[:, 0], arr_sub[:, 1]
            assoc_sub = cal_assoc(xi_sub, y_sub) 
            gain += prob * assoc_sub
        return gain


def cal_cond_assoc(xi: np.ndarray, xs: np.ndarray, y: np.ndarray):
    return CondAssoc(xi, xs, y).cal_cond_assoc()


# ---- 特征选择 -------------------------------------------------------------------------------------

class CondRMICMaximization(object):
    """条件关联系数最大化"""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """初始化"""
        self._check_values(X, y)
        self.X = X.copy()
        self.y = y.flatten()
        self.N, self.D = X.shape
        self.assocs = self._init_assocs()

    def _check_values(self, X, y):
        try:
            assert X.shape[0] == len(y)
        except:
            raise ValueError('the lengths of X and y do not match')

    def _init_assocs(self) -> dict:
        """计算所有特征与目标的关联系数"""
        assocs = {}
        for i in range(self.D):
            assocs[i] = cal_assoc(self.X[:, i].flatten(), self.y)
        return assocs

    def _recog_redundant_markov_features(self, F: set, S: list, R: set, U: set, f: int, thres: float = 0.01) -> set:
        """每选中一个特征f, 通过图上的Markov链冗余条件删除给定候选特征集F中通过f与目标y相连的冗余特征, Assoc(y;R|f) = 0"""
        R_new = set()
        for fi in F.difference(set(S).union(R).union(U)):
            cassoc = cal_cond_assoc(self.X[:, fi], self.X[:, f], self.y)
            if cassoc < thres:
                R_new.add(fi)

        if len(R_new) > 0:
            print('|S| = {}, R_new for {}: {}'.format(len(S), f, R_new))
            
        return R_new

    def _init_redundant_set(self, F, S, U):
        """初始化冗余集"""
        R = set()
        R_dict = defaultdict(set)

        if S == []:
            pass
        else:
            R_ = set()
            for s in S:
                R_new = self._recog_redundant_markov_features(F, S, R, U, s)
                R_dict[s] = R_new
                R_ = R_.union(R_new)
            R = R_
        return R, R_dict

    def _update_redundant_set(self, f, F, S, R, U, R_dict):
        """使用条件概率判断独立性, 提前识别冗余, 减少后续计算量"""
        R_new = self._recog_redundant_markov_features(F, S, R, U, f)
        R = R.union(R_new)  # 总体冗余特征集合
        R_dict[f] = R_new  # 用于存储每个xi特征对应的在S中关于y的其他冗余特征
        return R, R_dict

    def exec_feature_selection(self, S: list=None, assoc_thres: float=0.0, iter_n: int=250):
        """执行特征选择
        :param S: 已被提前选中的特征集合
        :param assoc_thres: 用于判断关联和无关的阈值
        :param max_features_num: 最大特征数
        """
        F = set(self.assocs.keys())  # F: 总候选特征idx集(不变)
        S = [] if S is None else S.copy()  # S: 已选特征idx集
        U = set([k for k, v in self.assocs.items()if v <= assoc_thres])  # U: 无关特征集(不变)
        R, R_dict = self._init_redundant_set(F, S, U)
        
        # NOTE: 本项目使用logs.
        logs = []

        # 迭代进行特征选择.
        cassocs = defaultdict(dict)

        i = 0
        while True:
            if i == 0:
                min_cassocs = self.assocs.copy()
                for p in set(S).union(R).union(U):  # 去掉已选中的S、冗余的R和无关的U
                    min_cassocs.pop(p)
            else:
                R, R_dict = self._update_redundant_set(f, F, S, R, U, R_dict)
                if i == 1:
                    min_cassocs = {}
                    for fi in F.difference(set(S).union(R).union(U)):
                        xi = self.X[:, fi].flatten()
                        for fs in S:
                            xs = self.X[:, fs].flatten()
                            cassocs[fi][f] = cal_cond_assoc(xi, xs, self.y)
                        min_cassocs[fi] = min(cassocs[fi].values())
                else:
                    xf = self.X[:, f].flatten()  # 上一轮被选入特征
                    for fi in F.difference(set(S).union(R).union(U)):
                        xi = self.X[:, fi].flatten()
                        cassocs[fi][f] = cal_cond_assoc(xi, xf, self.y)
                        min_cassocs[fi] = min(min_cassocs[fi], cassocs[fi][f])

            if min_cassocs == {}:
                break
            else:
                criterion, f = max(zip(min_cassocs.values(), min_cassocs.keys()))
                logs.append([i, f, criterion, self.assocs[f]])
                print('i: {}, f: {}'.format(i, f))

            # 更新集合.
            min_cassocs.pop(f)
            S.append(f)

            i += 1

            if len(S) > iter_n:
                break
        
        # NOTE: 注意该项目里返回logs.
        # return S, R, U, R_dict
        return logs


CondRMICMaximization.__doc__ = """
代码中变量说明: 
    Y: 目标
    F: 全量候选特征集
    S: 已被选中特征集
    U: 与目标无关特征集
    R: 已识别出的冗余特征集

关联算法选择:
    目前测试RMIC的计算准确度和稳定性最高

参考文献:
    F. Fleuret: Fast Binary Feature Selection with Conditional Mutual Information, Journal of Machine Learning Research, 2004. 
"""