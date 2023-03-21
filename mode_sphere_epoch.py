#!/usr/bin/env python
# Created by "Thieu" at 16:33, 16/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pandas import DataFrame
from mealpy.swarm_based import WOA
from pathlib import Path
import time
import numpy as np
np.random.seed(12345)


def fitness_func(solution):
    return np.sum(solution**2)


if __name__ == "__main__":
    model_name = "WOA"
    N_TRIALS = 10
    problem = {
        "fit_func": fitness_func,
        "lb": [-100, ] * 50,
        "ub": [100, ] * 50,
        "minmax": "min",
        "log_to": None,
    }
    epoch = 500
    pop_size = 50
    mode_names = ["single", "swarm", "thread", "process"]

    PATH_ERROR = f"history/error/{model_name}/"
    PATH_BEST_FIT = "history/best_fit/"
    Path(PATH_ERROR).mkdir(parents=True, exist_ok=True)
    Path(PATH_BEST_FIT).mkdir(parents=True, exist_ok=True)

    ## Run model
    best_fit_full = {}
    list_total_time = []

    for mode_name in mode_names:
        error_full = {}
        best_fit_list = []

        for id_trial in range(1, N_TRIALS + 1):
            time_start = time.perf_counter()

            model = WOA.OriginalWOA(epoch, pop_size)
            _, best_fitness = model.solve(problem, mode=mode_name)
            time_end = time.perf_counter() - time_start

            temp = f"trial_{id_trial}"
            error_full[temp] = model.history.list_global_best_fit
            best_fit_list.append(best_fitness)

            list_total_time.append([mode_name, id_trial, time_end])

        df = DataFrame(error_full)

        df.to_csv(f"{PATH_ERROR}{model_name}_{mode_name}_sphere_error.csv", header=True, index=False)
        best_fit_full[mode_name] = best_fit_list

    df = DataFrame(best_fit_full)
    df.to_csv(f"{PATH_BEST_FIT}/{model_name}_sphere_best_fit.csv", header=True, index=False)

    df_time = DataFrame(np.array(list_total_time), columns=["mode", "trial", "total_time"])
    df_time.to_csv(f"{PATH_BEST_FIT}/{model_name}_sphere_total_time.csv", header=True, index=False)
