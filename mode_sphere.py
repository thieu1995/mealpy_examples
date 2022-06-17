#!/usr/bin/env python
# Created by "Thieu" at 16:33, 16/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pandas import DataFrame
from mealpy.swarm_based import WOA
from os import getcwd, path, makedirs
import time
import numpy as np
np.random.seed(12345)


def fitness_func(solution):
    return np.sum(solution**2)


if __name__ == "__main__":
    model_name = "WOA"
    N_TRIALS = 10
    LB = [-100, ] * 50
    UB = [100, ] * 50
    epoch = 100
    pop_size = 50
    mode_names = ["single", "swarm", "thread", "process"]

    PATH_ERROR = "history/error/" + model_name + "/"
    PATH_BEST_FIT = "history/best_fit/"
    check_dir1 = f"{getcwd()}/{PATH_ERROR}"
    check_dir2 = f"{getcwd()}/{PATH_BEST_FIT}"
    if not path.exists(check_dir1): makedirs(check_dir1)
    if not path.exists(check_dir2): makedirs(check_dir2)

    ## Run model
    best_fit_full = {}
    best_fit_columns = []
    list_total_time = []

    for mode_name in mode_names:
        error_full = {}
        error_columns = []
        best_fit_list = []

        for id_trial in range(1, N_TRIALS + 1):
            time_start = time.perf_counter()
            problem = {
                "fit_func": fitness_func,
                "lb": LB,
                "ub": UB,
                "minmax": "min",
                "log_to": None,
            }
            model = WOA.BaseWOA(problem, epoch, pop_size)
            _, best_fitness = model.solve(mode=mode_name)
            time_end = time.perf_counter() - time_start

            temp = f"trial_{id_trial}"
            error_full[temp] = model.history.list_global_best_fit
            error_columns.append(temp)
            best_fit_list.append(best_fitness)

            list_total_time.append([mode_name, id_trial, time_end])

        df = DataFrame(error_full, columns=error_columns)

        df.to_csv(f"{PATH_ERROR}{len(LB)}D_{model_name}_{mode_name}_sphere_error.csv", header=True, index=False)
        best_fit_full[mode_name] = best_fit_list
        best_fit_columns.append(mode_name)

    df = DataFrame(best_fit_full, columns=best_fit_columns)
    df.to_csv(f"{PATH_BEST_FIT}/{len(LB)}D_{model_name}_sphere_best_fit.csv", header=True, index=False)

    df_time = DataFrame(np.array(list_total_time), columns=["mode", "trial", "total_time"])
    df_time.to_csv(f"{PATH_BEST_FIT}/{len(LB)}D_{model_name}_sphere_total_time.csv", header=True, index=False)
