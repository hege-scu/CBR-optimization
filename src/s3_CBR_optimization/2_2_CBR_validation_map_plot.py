from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from src.setting import plt, PROJ_CMAP
blue, orange = PROJ_CMAP["blue"], PROJ_CMAP["orange"]

if __name__ == "__main__":
    # 载入三个收率的计算结果
    
    # 每个收率DF的字段意义:
    #   dist: 测试实际工艺向量theta与案例库搜索最优工艺theta*之差的二范数值, ||theta_obs - theta*||
    #   delta_score: 案例库搜索最优收率与测试样本收率之差, Y* - Y_obs
    #   theta_map: 就是案例库搜索得最优的工艺变量theta_*的范数值
    #   theta_obs: 每个测试样本的工艺变量theta_obs的范数值

    method = 'EUCLID'  # ****注意，此处改变分别执行一次才能得到全部图 'EUCLID','ManH'

    df_gas = pd.read_csv(os.path.join(BASE_DIR, 'file/s3_optimization/optim_valid_result_gasoline_{}.csv'.format(method)))
    df_liq = pd.read_csv(os.path.join(BASE_DIR, 'file/s3_optimization/optim_valid_result_liquidtotal_{}.csv'.format(method)))
    df_coke = pd.read_csv(os.path.join(BASE_DIR, 'file/s3_optimization/optim_valid_result_coke_{}.csv'.format(method)))
    
    df_lst = [df_gas, df_liq, df_coke]
    plt.figure(figsize=(12, 5))
    for i in range(3):
        df = df_lst[i]
        
        plt.subplot(1, 3, i + 1)

        # 1. 散点图可视化
        xy = df.values
        df["c"] = gaussian_kde(xy.T, bw_method="scott")(xy.T)
        df.sort_values(by="c", ascending=True, inplace=True)
        
        plt.scatter(
            df["dist"], df["delta_score"], marker="o", alpha=1, c=df["c"], cmap="jet", 
            s=df["c"] / 10, zorder=1)
        plt.colorbar()

        # 2. 线性回归
        # x, y = df[["dist"]].values, df["delta_score"].values
        # model = LinearRegression()
        # model.fit(x, y)
        #
        # a, b = round(model.coef_[0], 3), round(model.intercept_, 3)
        # r2 = r2_score(y, model.predict(x))
        # print(a, b, r2)
        #
        # std = np.std(model.predict(x))
        # std_z = 1.96
        # ci = std * std_z
        # x = np.linspace(-0.5, 1.5, 100).reshape(-1, 1)
        # plt.plot(x, model.predict(x) - ci, "--", c="k", linewidth=0.5)
        # plt.plot(x, model.predict(x), "-", c="k", linewidth=1.0, label="lr")
        # plt.plot(x, model.predict(x) + ci, "--", c="k", linewidth=0.5)

        if method == "EUCLID":
            alias = "Euclidean"
        else:
            alias = "ManH"
        
        plt.xlabel(alias + r" norm of $(\theta_{\rm obs} - \theta^*_{\rm MAP})$")
        plt.ylabel("$Y^{*} - Y_{\\rm obs}$")

        plt.xlim([0, 1.5])

        if i in [0, 1]:
            plt.ylim([-0.1, 0.8])
        elif i in [2]:
            plt.ylim([-0.8, 0.2])

        plt.grid(True, linewidth=0.2, zorder=-10)
        plt.tight_layout()

    plt.savefig(os.path.join(BASE_DIR, f'img/基于{method}距离的优化前后收率变化.png'), dpi=450)

    # 这张图是为了显示优化前后的工艺变量theta的变化. 因为theta可能为高维向量, 所以使用欧式范数来进行标记
    # 对比优化前后theta_obs和theta*之间欧氏范数变化, 来查看工艺变量的优化情况
    plt.figure(figsize=(10, 2))
    for i in range(3):
         plt.subplot(1, 3, i + 1)
        
         df = df_lst[i]
         x_map, x_obs = df["theta_map"].values, df["theta_obs"].values
        
         plt.scatter(x_map, np.ones_like(x_map), s=5, c="r", alpha=0.1, linewidth=0.3, zorder=5)
         plt.scatter(x_obs, np.zeros_like(x_obs), s=5, c="b", alpha=0.1, linewidth=0.3, zorder=5)
        
         for j, x in enumerate(x_obs):
             plt.plot([x, x_map[j]], [0, 1], "k", linewidth=0.1, alpha=0.2, zorder=-5)

         if method == "EUCLID":
             alias = "Euclidean"
         else:
             alias = "ManH"

         plt.xlabel(alias + r" norm of $\theta$")
         plt.yticks([0, 1], [r"$\theta_{\rm obs}$", r"$\theta_{\rm MAP}^*$"])
         plt.ylim([-0.5, 1.5])
    plt.tight_layout()

    plt.savefig(os.path.join(BASE_DIR, f'img/基于{method}距离的优化前后Theta范数变化.png'), dpi=450)

    print("基于范数的CBR优化有效性验证图绘制成功！")