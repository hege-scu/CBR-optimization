# -*- coding: utf-8 -*-
"""
Created on 2021/1/25 1:03

@Project -> File: mi-time-delayed-chemical-process-prediction -> partial_derives.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 符号计算和偏微分方程
"""

from sympy import solve as symsolve
from sympy import symbols, diff
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../'))
sys.path.append(BASE_DIR)


def gen_x_syms(S_dim: int) -> list:
    """
    根据曲面维度生成所有变量符号x
    """
    x_syms = []
    for i in range(S_dim):
        x_syms.append(symbols('x_{}'.format(i)))
    return x_syms


def gen_sym_S(S, x_syms: list):
    """
    生成隐函数S的符号式
    * S(x_0, x_1, ..., x_n) = 0
    """
    S_sym = S(x_syms)
    return S_sym


def gen_sym_G(S_sym, x_syms: list) -> list:
    """
    求取隐函数S对应的显函数G符号式列表
    * x_n = G(x_0, x_1, ..., x_n-1)
    """
    G_syms_list = symsolve(S_sym, x_syms[-1])
    return G_syms_list


def cal_sym_func_value(F_sym, subs_dict: dict) -> float:
    """
    计算符号函数f的值
    :param f: sym func, 以symbol形式记录的函数
    :param subs_dict: dict, 符号函数中各变量和对应值字典, 例如{'x_0': 0.0, 'x_1': 1.0, ...}
    """
    v = float(F_sym.subs(subs_dict))
    return v


def cal_partial_derive_syms(F_sym, F_dim: int) -> list:
    """
    计算函数F的偏导数向量符号式
    """
    x_syms = gen_x_syms(F_dim)
    PD_syms = []
    for i in range(F_dim):
        PD_syms.append(diff(F_sym, x_syms[i]))
    return PD_syms


def extract_x_syms_in_func_sym(F_sym, F_dim: int) -> list:
    """
    提取函数符号F_sym中出现的所有自变量x的符号
    """
    x_sym_strs = []
    for i in range(F_dim):
        if symbols('x_{}'.format(i)) in F_sym.free_symbols:
            x_sym_strs.append('x_{}'.format(i))
    return x_sym_strs


class PartialDerives(object):
    """
    计算隐函数各维度偏导数的表达式和值
    """
    
    def __init__(self, func, f_dim):
        """
        初始化
        :param func: func, 函数必须为转换为隐函数形式输入, func(x) = 0
        :param f_dim: int, 函数里自变量的维数
        """
        self.F = func
        self.F_dim = f_dim
        
        # 生成变量和函数的偏导数符号.
        self.x_syms = gen_x_syms(self.F_dim)
        self.F_sym = gen_sym_S(self.F, self.x_syms)
        self.G_syms_list = gen_sym_G(self.F_sym, self.x_syms)
        self.G_sym = self.G_syms_list[0]  # TODO: 目前默认选择第一个解, 需要改为对所有解均进行计算
    
    def _cal_partial_derive_syms(self) -> list:
        """
        计算隐函数F对各x的偏导符号
        """
        PD_syms = cal_partial_derive_syms(self.F_sym, self.F_dim)
        return PD_syms
    
    @property
    def PD_syms(self):
        return self._cal_partial_derive_syms()
    
    def _extract_x_syms_in_PD(self) -> list:
        """
        提取PD_syms中各个符号表达式中所含自变量x符号
        """
        PD_have_x_syms = []
        for i in range(self.F_dim):
            PD_ = self.PD_syms[i]
            x_sym_strs_ = extract_x_syms_in_func_sym(PD_, self.F_dim)
            PD_have_x_syms.append(x_sym_strs_)
        return PD_have_x_syms
    
    def _extract_x_syms_in_G(self) -> list:
        G_has_x_syms = extract_x_syms_in_func_sym(self.G_sym, self.F_dim)
        return G_has_x_syms
    
    @property
    def PD_have_x_syms(self):
        return self._extract_x_syms_in_PD()
    
    @property
    def G_has_x_syms(self):
        return self._extract_x_syms_in_G()
    
    def cal_partial_derive_values(self, x: list):
        """
        计算偏导数
        * x不需要输入最后x_n的值, 可以通过G函数计算, 所以 x = [x_0, x_1, ..., x_n-1]
        """
        x = x.copy()
        assert len(x) == self.F_dim - 1
        
        # 计算G的值.
        subs_dict_ = {}
        for x_sym_str in self.G_has_x_syms:
            subs_dict_[x_sym_str] = x[int(x_sym_str.split('_')[1])]
        x_end = cal_sym_func_value(self.G_sym, subs_dict_)
        x.append(x_end)
        
        pd_values = []
        for i in range(self.F_dim):
            subs_dict_ = {}
            for x_sym_str in self.PD_have_x_syms[i]:
                subs_dict_[x_sym_str] = x[int(x_sym_str.split('_')[1])]
            pd_ = float(self.PD_syms[i].subs(subs_dict_))
            pd_values.append(pd_)
        
        return x, pd_values


if __name__ == '__main__':
    # %% 测试.
    def f(x: list):
        y = 0.5 * x[1] - x[0] ** 2
        return y
    
    
    f_dim = 2
    self = PartialDerives(f, f_dim)
    x, pd_values = self.cal_partial_derive_values([1])
