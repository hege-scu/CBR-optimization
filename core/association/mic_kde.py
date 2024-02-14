# -*- coding: utf-8 -*-
"""
Created on 2021/11/24 20:51:02

@File -> mic_kde.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 使用核密度估计重采样计算RMIC
"""

from sklearn.neighbors._kde import KernelDensity
from minepy import MINE
import numpy as np

ALPHA = 0.6
C = 5
EPS = 1e-6


def _normalize(x: np.ndarray):
    x_min, x_max = np.min(x), np.max(x)
    if x_min == x_max:
        return np.ones_like(x)
    else:
        return (x - x_min) / (x_max - x_min)


def _kde_resample(x: np.ndarray, y: np.ndarray, bw: float=0.05, N: int=300):
    # 重归一化到[0, 1]区间.
    x, y = _normalize(x.flatten()), _normalize(y.flatten())
    samples = np.vstack((x, y)).T
    kde = KernelDensity(
        kernel='gaussian', 
        bandwidth=bw,
        ).fit(samples)
    kde_samples = kde.sample(N)
    return kde_samples[:, 0], kde_samples[:, 1]
    

class PairwiseMIC(object):
    """MIC成对检验"""

    def __init__(self, x: np.ndarray, y: np.ndarray, mic_params: dict = None):
        # self.x, self.y = x.flatten(), y.flatten()
        if len(x) < 1000:
            self.x, self.y = _kde_resample(x, y, N=1000)
        else:
            self.x, self.y = x.flatten(), y.flatten()
            
        if mic_params is None:
            mic_params = {'alpha': ALPHA, 'c': C}
        self.mic_params = mic_params

    def cal_assoc(self):
        mine = MINE(**self.mic_params)
        mine.compute_score(self.x, self.y)
        return mine.mic()


class PairwiseRMIC(object):
    """MIC成对检验"""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, mic_params: dict = None):
        # self.x, self.y = x.flatten(), y.flatten()
        if len(x) < 200:
            self.x, self.y = _kde_resample(x, y, N=200)
        else:
            self.x, self.y = x.flatten(), y.flatten()
            
        self.N = len(self.x)
        if mic_params is None:
            mic_params = {'alpha': ALPHA, 'c': C}
        self.mic_params = mic_params
    
    def cal_assoc(self, rebase: bool = True):
        if self.N < 100:
            r = 1
        else:
            r = 1
            
        m = []
        for _ in range(r):
            mine = MINE(**self.mic_params)
            mine.compute_score(
                self.x + EPS * np.random.random(self.N), 
                self.y + EPS * np.random.random(self.N)
                )
            m.append(mine.mic())
        mic = np.mean(m)
        
        if not rebase:
            return mic
        else:
            base_mic = self._cal_base_mic(len(self.x))
            
            # <<<<<<
            # if mic > base_mic:
            #     return (mic - base_mic) / (1.0 - base_mic)
            # else:
            #     return 0.0
            # ------
            return (mic - base_mic) / (1.0 - base_mic)
            # >>>>>>
    
    @staticmethod
    def _cal_base_mic(n):
        """经过排数值实验测试所得"""
        # return -0.057 * np.log(n) + 0.5134
        # return model.predict(np.array([[n]]))[0]
        return 0.974 * np.power(n, -0.29)