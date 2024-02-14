# -*- coding: utf-8 -*-
"""
Created on 2021/08/20 10:50:28

@File -> mi.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 
"""

from pyitlib import discrete_random_variable as drv
from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma, psi, digamma
from sklearn.neighbors import BallTree, KDTree
import numpy.linalg as la
from numpy import log
import pandas as pd
import numpy as np

from .__mi.mie_kraskov import mi as cal_kraskov

VAR_TYPES = ['numeric', 'categoric']


# def discretize_series(x: np.ndarray, n: int = 100, method='cut'):
#     """对数据序列采用等频分箱"""
#     q = int(len(x) // n)

#     # 分箱方式.
#     if method == 'qcut':
#         x_enc = pd.qcut(x, q, labels=False, duplicates='drop').flatten()  # 等频分箱
#     if method == 'cut':
#         x_enc = pd.cut(x, q, labels=False, duplicates='drop').flatten()  # 等宽分箱
        
#     return x_enc


class PairwiseMI(object):
    """MIC成对检验"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x.flatten()
        self.y = y.flatten()
        # self.x = discretize_series(x.flatten(), n=150)
        # self.y = discretize_series(y.flatten(), n=150)
        
    def cal_assoc(self):

        if len(self.x) < 1000:
            k = 3
        elif (len(self.x) < 3000) & (len(self.x) >= 1000):
            k = 2
        else:
            k = 1
        return cal_kraskov(self.x.reshape(-1, 1), self.y, k = k)

        # return drv.information_mutual(self.x.astype(int), self.y.astype(int))


# # ---- 连续离散变量之间互信息 -----------------------------------------------------------------------

# def _cal_mi_cd(x, y, k):
# 	"""
# 	Calculates the mututal information between a continuous vector x and a
# 	disrete class vector y.

# 	This implementation can calculate the MI between the joint distribution of
# 	one or more continuous variables (X[:, 1:3]) with a discrete variable (y).

# 	Thanks to Adam Pocock, the author of the FEAST package for the idea.

# 	Brian C. Ross, 2014, PLOS ONE
# 	Mutual Information between Discrete and Continuous Data Sets
# 	"""
	
# 	y = y.flatten()
# 	n = x.shape[0]
# 	classes = np.unique(y)
# 	knn = NearestNeighbors(n_neighbors = k)
# 	d2k = np.empty(n)
# 	Nx = []
# 	for yi in y:
# 		Nx.append(np.sum(y == yi))
	
# 	for c in classes:
# 		mask = np.where(y == c)[0]
# 		knn.fit(x[mask, :])
# 		d2k[mask] = knn.kneighbors()[0][:, -1]
	
# 	knn.fit(x)
# 	m = knn.radius_neighbors(radius = d2k, return_distance = False)
# 	m = [i.shape[0] for i in m]
	
# 	MI = psi(n) - np.mean(psi(Nx)) + psi(k) - np.mean(psi(m))
# 	return MI


# # ---- 连续连续变量之间互信息 -----------------------------------------------------------------------

# def cal_kraskov(x, y, z=None, k=3, base=2, alpha=0):
#     """ Mutual information of x and y (conditioned on z if z is not None)
#         x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
#         if x is a one-dimensional scalar and we have four samples
#     """
#     assert len(x) == len(y), "Arrays should have same length"
#     assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
#     x, y = np.asarray(x), np.asarray(y)
#     x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
#     x = add_noise(x)
#     y = add_noise(y)
#     points = [x, y]
#     if z is not None:
#         z = np.asarray(z)
#         z = z.reshape(z.shape[0], -1)
#         points.append(z)
#     points = np.hstack(points)
#     # Find nearest neighbors in joint space, p=inf means max-norm
#     tree = build_tree(points)
#     dvec = query_neighbors(tree, points, k)
#     if z is None:
#         a, b, c, d = avgdigamma(x, dvec), avgdigamma(
#             y, dvec), digamma(k), digamma(len(x))
#         if alpha > 0:
#             d += lnc_correction(tree, points, k, alpha)
#     else:
#         xz = np.c_[x, z]
#         yz = np.c_[y, z]
#         a, b, c, d = avgdigamma(xz, dvec), avgdigamma(
#             yz, dvec), avgdigamma(z, dvec), digamma(k)
#     return (-a - b + c + d) / log(base)


# def lnc_correction(tree, points, k, alpha):
#     e = 0
#     n_sample = points.shape[0]
#     for point in points:
#         knn = tree.query(point[None, :], k=k+1, return_distance=False)[0]
#         knn_points = points[knn]
#         knn_points = knn_points - knn_points[0]
#         covr = knn_points.T @ knn_points / k
#         _, v = la.eig(covr)
#         V_rect = np.log(np.abs(knn_points @ v).max(axis=0)).sum()
#         log_knn_dist = np.log(np.abs(knn_points).max(axis=0)).sum()
#         if V_rect < log_knn_dist + np.log(alpha):
#             e += (log_knn_dist - V_rect) / n_sample
#     return e


# def add_noise(x, intens=1e-12):
#     return x + intens * np.random.random_sample(x.shape)


# def avgdigamma(points, dvec):
#     tree = build_tree(points)
#     dvec = dvec - 1e-15
#     num_points = count_neighbors(tree, points, dvec)
#     return np.mean(digamma(num_points))


# def build_tree(points):
#     if points.shape[1] >= 20:
#         return BallTree(points, metric='chebyshev')
#     return KDTree(points, metric='chebyshev')


# def count_neighbors(tree, x, r):
#     return tree.query_radius(x, r, count_only=True)


# def query_neighbors(tree, x, k):
#     return tree.query(x, k=k + 1)[0][:, k]