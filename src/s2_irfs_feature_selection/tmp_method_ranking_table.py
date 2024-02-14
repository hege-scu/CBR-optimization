import pandas as pd
import numpy as np
import sys
import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from src.setting import IRFS_METHODS, TARGETS
from src.setting import plt

IRFS_METHODS = ['CRMICM', 'JMIM', 'JMI', 'MRMR']

def locate_peak(series: np.ndarray, thres: float = 0.002):
    v_max = np.max(series)
    if np.abs(np.sort(series)[-2] - v_max) > thres:
        loc = np.argwhere(series == v_max).min()
    else:
        loc = np.argwhere(series >= (v_max - thres)).min()
    return loc


def get_metric_fimt_table(y_col, model_name):
    table = None
    for i, method in enumerate(IRFS_METHODS):
        r2_fimt_curve = pd.read_csv(
            os.path.join(BASE_DIR, 'file/s2_incremental_test/fimt_accu_{}_{}_{}_{}.csv'.format(
                y_col, method, model_name, 'r2')))
        # mse_fimt_curve = pd.read_csv(
        #     '../../file/s2_incremental_test/fimt_accu_{}_{}_{}_{}.csv'.format(
        #         y_col, method, model_name, 'mse'))

        n_features = r2_fimt_curve['n_features'].values
        r2_values = r2_fimt_curve[method].values
        # mse_values = mse_fimt_curve[method].values

        locs = np.argwhere(np.isin(n_features, [1, 31, 61, 91, 121])).flatten()
        
        table_ = [method] + r2_values[locs].tolist()
        table_ = pd.DataFrame([table_])

        if table is None:
            table = table_
        else:
            table = pd.concat([table, table_], axis=0)
    return table
    

if __name__ == '__main__':
    model_name = 'rf'

    # 画出排名曲线.
    rk_df = pd.DataFrame(columns=IRFS_METHODS)
    for y_col in TARGETS:
        metric_table = get_metric_fimt_table(y_col, model_name)

        total_rank = []
        for label in range(1, 5):
            rank_list = []
            scores_ranked = metric_table.sort_values(by=label, ascending=False)[0].tolist()
            for method in IRFS_METHODS:
                rank_list.append(scores_ranked.index(method))
            total_rank.append(rank_list)
        
        total_rank = pd.DataFrame(total_rank, columns=IRFS_METHODS) + 1
        rk_df = pd.concat([rk_df, total_rank], axis=0)

    rk_df.to_csv(os.path.join(BASE_DIR, 'file/s2_incremental_test/rk_df_{}.csv'.format('r2')), index=False)

    # 寻找最适降维位置.        
    #         best_loc = locate_peak(r2_values)
    #         best_n = n_features[best_loc]
    #         best_r2 = r2_values[best_loc]
    #         best_mse = mse_values[best_loc]
            
    #         fill_in_values += [best_n, best_r2, best_mse]
        
    #     fill_in_values = pd.DataFrame([fill_in_values])
        
    #     if i == 0:
    #         table = fill_in_values
    #     else:
    #         table = pd.concat([table, fill_in_values], axis=0)