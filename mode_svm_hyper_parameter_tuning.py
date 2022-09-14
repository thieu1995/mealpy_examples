#!/usr/bin/env python
# Created by "Thieu" at 17:26, 16/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

## Link: https://vitalflux.com/classification-model-svm-classifier-python-example/

import time
from pathlib import Path
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.classification_svc import ClassificationSVC
from mealpy.swarm_based import WOA
from src.utils import data_util


if __name__ == "__main__":
    # LABEL ENCODER
    list_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel_encoder = LabelEncoder()
    kernel_encoder.fit(list_kernels)

    data = data_util.generate_data_classification_data(test_ratio=0.25)
    data["KERNEL_ENCODER"] = kernel_encoder

    # x1. C: float [0.1 to 10000.0]
    # x2. Kernel: [‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’]

    LB = [0.1, 0.]
    UB = [10000.0, 3.99]
    problem = ClassificationSVC(lb=LB, ub=UB, minmax="max", data=data, save_population=False, log_to=None)

    model_name = "WOA"
    N_TRIALS = 10
    epoch = 100
    pop_size = 20
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
        df.to_csv(f"{PATH_ERROR}{model_name}_{mode_name}_svc_paras_tuning_error.csv", header=True, index=False)
        best_fit_full[mode_name] = best_fit_list

    df = DataFrame(best_fit_full)
    df.to_csv(f"{PATH_BEST_FIT}/{model_name}_svc_paras_tuning_best_fit.csv", header=True, index=False)

    df_time = DataFrame(np.array(list_total_time), columns=["mode", "trial", "total_time"])
    df_time.to_csv(f"{PATH_BEST_FIT}/{model_name}_svc_paras_tuning_total_time.csv", header=True, index=False)
