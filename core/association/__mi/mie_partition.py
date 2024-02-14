# -*- coding: utf-8 -*-
"""
Created on 2021/05/13 18:53:54

@File -> marginal_equiquant.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于Marginal Equiquantization方法的互信息计算
"""

__doc__ = """
    参考文献: 
    Georges A. Darbellay: Predictability: An Information-Theoretic Perspective, Signal Analysis \
        and Prediction, 1998.
"""

from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from typing import List
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../../../'))
sys.path.append(BASE_DIR)


# #### 预处理 #######################################################################################

def minmax_norm(arr: np.ndarray):
    D = arr.shape[1]
    scaler = MinMaxScaler()
    arr_norm = None
    for i in range(D):
        a = scaler.fit_transform(arr[:, i: i + 1])

        if arr_norm is None:
            arr_norm = a
        else:
            arr_norm = np.hstack((arr_norm, a))
    return arr_norm


# #### 数据样本离散化 ###############################################################################

class Cell(object):
    """边际等概率离散化中的单元格对象
    """

    def __init__(self, arr: np.ndarray) -> None:
        """初始化

        :param arr: 存储有x,y坐标的二维数组, shape = (N, D=2)
        """
        self.arr = arr.copy()
        self.N, self.D = arr.shape

        if self.D != 2:
            raise ValueError('the input dimension is not equal to 2')

    def _cal_area(self):
        area = 1.0
        for i in range(self.D):
            area *= self.bounds[i][1] - self.bounds[i][0]
        self.area = area

    def def_cell_bounds(self, bounds: List[tuple]):
        """用户定义cell的边界

        :param bounds: 边界值list, 如[(x_min, x_max), (y_min, y_max)]
        """
        self.bounds = bounds
        self._cal_area()

    def cal_proba_dens(self) -> float:
        """计算以样本数计的概率密度
        """
        if self.area == 0.0:
            return 0.0
        else:
            return self.N / self.area

    def _get_marginal_partition_thres(self) -> List[float]:
        """获取各维度上等边际概率(即等边际样本数)分箱的阈值
        """
        if self.N == 1:
            part_thres = list(self.arr.flatten())
        else:
            part_idx = self.N // 2  # 离散化位置idx
            part_thres = []
            for i in range(self.D):
                arr_srt = self.arr[np.argsort(self.arr[:, i]), :]  # 对应维度值升序排列
                if self.N % 2 == 0:  # 以均值划分
                    marginal_part_value = (
                        arr_srt[part_idx - 1, i] + arr_srt[part_idx, i]) / 2
                else:
                    marginal_part_value = (
                        arr_srt[part_idx - 1, i] + arr_srt[part_idx + 1, i]) / 2
                part_thres.append(marginal_part_value)
        return part_thres

    def get_marginal_partition_thres(self):
        self.part_thres = self._get_marginal_partition_thres()

    def exec_partition(self):
        """执行边际等概率离散化, 执行这一步的要求为self.N > 0
        """
        # 先在x方向上分为左右两部分.
        part_arr_l = self.arr[
            np.where((self.arr[:, 0] < self.part_thres[0]) &
                     (self.arr[:, 0] >= self.bounds[0][0]))
        ]
        part_arr_r = self.arr[
            np.where((self.arr[:, 0] >= self.part_thres[0])
                     & (self.arr[:, 0] <= self.bounds[0][1]))
        ]

        # 再在y方向上继续切分.
        part_arr_ul = part_arr_l[np.where(
            (part_arr_l[:, 1] >= self.part_thres[1]) & (part_arr_l[:, 1] <= self.bounds[1][1]))]
        part_arr_ll = part_arr_l[np.where(
            (part_arr_l[:, 1] < self.part_thres[1]) & (part_arr_l[:, 1] >= self.bounds[1][0]))]

        part_arr_ur = part_arr_r[np.where(
            (part_arr_r[:, 1] >= self.part_thres[1]) & (part_arr_r[:, 1] <= self.bounds[1][1]))]
        part_arr_lr = part_arr_r[np.where(
            (part_arr_r[:, 1] < self.part_thres[1]) & (part_arr_r[:, 1] >= self.bounds[1][0]))]

        cell_ul, cell_ur, cell_ll, cell_lr = Cell(part_arr_ul), Cell(part_arr_ur), \
            Cell(part_arr_ll), Cell(part_arr_lr)

        # 确定边界.
        (xl, xu), (yl, yu) = self.bounds
        x_thres, y_thres = self.part_thres
        cell_ul.def_cell_bounds([(xl, x_thres), (y_thres, yu)])
        cell_ur.def_cell_bounds([(x_thres, xu), (y_thres, yu)])
        cell_ll.def_cell_bounds([(xl, x_thres), (yl, y_thres)])
        cell_lr.def_cell_bounds([(x_thres, xu), (yl, y_thres)])
        return cell_ul, cell_ur, cell_ll, cell_lr

    def show(self, linewidth: float = 0.5):
        (xl, xu), (yl, yu) = self.bounds
        plt.plot([xl, xu], [yl, yl], '-', c='k', linewidth=linewidth)
        plt.plot([xu, xu], [yl, yu], '-', c='k', linewidth=linewidth)
        plt.plot([xu, xl], [yu, yu], '-', c='k', linewidth=linewidth)
        plt.plot([xl, xl], [yu, yl], '-', c='k', linewidth=linewidth)


# 递归离散化.
def _try_partition(cell: Cell, min_samples_split: int, p_eps: float):
    if cell.N < min_samples_split:
        return None, None, None, None  # TODO: 返回None可能会导致递归异常, 建议改为Cell([])
    else:
        proba_dens = cell.cal_proba_dens()

        # 尝试分裂一下, 并检查分裂效果.
        cell.get_marginal_partition_thres()
        cell_ul, cell_ur, cell_ll, cell_lr = cell.exec_partition()

        is_proba_dens_converged = True

        for c in [cell_ul, cell_ur, cell_ll, cell_lr]:
            if (np.abs(c.cal_proba_dens() - proba_dens) / proba_dens > p_eps):
                is_proba_dens_converged = False
                break

        if not is_proba_dens_converged:
            return cell_ul, cell_ur, cell_ll, cell_lr
        else:
            return None, None, None, None


def recursively_partition(cell: Cell, min_samples_split: int = 30, p_eps: float = 1e-3) -> tuple:
    """对一个cell进行递归离散化

    :param cell: 初始cell
    :param p_eps: 子cell概率与父cell相对偏差阈值, 如果所有都小于该值则终止离散化, defaults to 1e-3
    """
    leaf_cells = []

    def _partition(cell):
        part_ul, part_ur, part_ll, part_lr = _try_partition(
            cell, min_samples_split, p_eps)

        if part_ul is None:
            leaf_cells.append(cell)
        else:
            _partition(part_ul)
            _partition(part_ur)
            _partition(part_ll)
            _partition(part_lr)

    _partition(cell)
    return leaf_cells



# #### 互信息熵 #####################################################################################

class MutualInfoEntropy(object):

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x.reshape(-1, 1).copy()
        self.y = y.reshape(-1, 1).copy()
        self.arr = minmax_norm(np.hstack((self.x, self.y)))
        self.N = self.arr.shape[0]

    def _equiquantize(self, **kwargs):
        cell = Cell(self.arr)
        cell.def_cell_bounds([(0.0, 1.0), (0.0, 1.0)])
        leaf_cells = recursively_partition(cell, **kwargs)

        leaf_cells = [c for c in leaf_cells if c.N > 0]

        return leaf_cells

    def equiquantize(self, **kwargs):
        self.leaf_cells = self._equiquantize(**kwargs)

    def cal_mie(self):
        n_leafs = len(self.leaf_cells)

        mie = 0.0
        for i in range(n_leafs):
            cell = self.leaf_cells[i]  # type: Cell
            (xl, xu), (yl, yu) = cell.bounds

            Nxy = len(cell.arr)
            Nx = len(np.where((self.arr[:, 0] >= xl)
                              & (self.arr[:, 0] < xu))[0])
            Ny = len(np.where((self.arr[:, 1] >= yl)
                              & (self.arr[:, 1] < yu))[0])

            gain = Nxy * np.log2(Nxy / Nx / Ny)
            mie += gain

        mie = mie / self.N + np.log2(self.N)
        return mie


def cal_mie(x: np.ndarray, y: np.ndarray, **kwargs):
    mutual_info_entropy = MutualInfoEntropy(x, y)
    mutual_info_entropy.equiquantize(**kwargs)
    mie = mutual_info_entropy.cal_mie()
    return mie


def cal_rho(x: np.ndarray, y: np.ndarray, **kwargs):
    mie = cal_mie(x, y, **kwargs)
    rho = np.sqrt(1 - np.power(2, -2 * mie))
    return rho


# if __name__ == '__main__':
#     from src.settings import *

#     # ---- 载入数据 ---------------------------------------------------------------------------------

#     def load_data(func: str, radius: float) -> Tuple[np.ndarray]:
#         """载入数据
#         """
#         from core.dataset.data_generator import DataGenerator

#         N = 5000
#         data_gener = DataGenerator(N=N)
#         x, y, _, _ = data_gener.gen_data(func, normalize=True)

#         # 加入噪音.
#         from mod.data_process.add_noise import add_circlular_noise

#         x, y = add_circlular_noise(x, y, radius=radius)

#         return x, y

#     # ---- 生成数据 ---------------------------------------------------------------------------------

#     from core.dataset.data_generator import FUNC_NAMES

#     proj_plt.figure(figsize=[6, 6])
#     for func in FUNC_NAMES[:1]:
#         radius_lst = np.arange(0.1, 10.0, 0.1)
#         mie_lst = []
#         params = {'p_eps': 1e-3, 'min_samples_split': 1000}
#         for radius in radius_lst:
#             x, y = load_data(func, radius)
#             # mie = cal_mie(x, y, **params)
#             mie = cal_rho(x, y, **params)
#             mie_lst.append(mie)

#         proj_plt.scatter(radius_lst, mie_lst, s=6)
#         # proj_plt.pause(0.1)
