# -*- coding: utf-8 -*-
"""
Created on 2021/08/20 10:56:45

@File -> mic.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe:  RMIC、MIC等互信息核心算法
"""

from minepy import MINE
import numpy as np
import pickle
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

ALPHA = 0.6
C = 5
EPS = 1e-6

with open(os.path.join(BASE_DIR, 'core/association/rgsr.pickle'), 'rb') as f:
    model = pickle.load(f)


class PairwiseMIC(object):
    """MIC成对检验"""

    def __init__(self, x: np.ndarray, y: np.ndarray, mic_params: dict = None):
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
        self.x, self.y = x.flatten(), y.flatten()
        self.N = len(self.x)
        if mic_params is None:
            # mic_params = {'alpha': ALPHA, 'c': C, 'est': 'mic_e'}
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
            if mic > base_mic:
                return (mic - base_mic) / (1.0 - base_mic)
            else:
                return 0.0
            # ------
            # return (mic - base_mic) / (1.0 - base_mic)
            # >>>>>>
    
    @staticmethod
    def _cal_base_mic(n):
        """经过排数值实验测试所得"""
        # return -0.057 * np.log(n) + 0.5134
        # return model.predict(np.array([[n]]))[0]
        return 0.9893 * np.power(n, -0.292)