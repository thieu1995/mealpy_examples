#!/usr/bin/env python
# Created by "Thieu" at 10:50, 17/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from src.timeseries_mlp import TimeSeriesMLP
from src.utils import data_util
from mealpy.swarm_based import WOA
import time
import numpy as np

np.random.seed(12345)


if __name__ == "__main__":

    list_optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    list_network_weight_initials = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    list_activations = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

    # LABEL ENCODER
    opt_encoder = LabelEncoder()
    opt_encoder.fit(list_optimizers)  # domain range ==> 7 values

    nwi_encoder = LabelEncoder()
    nwi_encoder.fit(list_network_weight_initials)

    act_encoder = LabelEncoder()
    act_encoder.fit(list_activations)

    data = data_util.generate_time_series_data(train_ratio=0.75)
    data["OPT_ENCODER"] = opt_encoder
    data["NWI_ENCODER"] = nwi_encoder
    data["ACT_ENCODER"] = act_encoder

    model_name = "WOA"
    N_TRIALS = 10
    LB = [1, 5, 0, 0.01, 0, 0, 5]
    UB = [3.99, 20.99, 6.99, 0.5, 7.99, 7.99, 50]
    epoch = 10
    pop_size = 20
    mode_names = ["single", "swarm", "thread", "process"]

    problem = TimeSeriesMLP(lb=LB, ub=UB, minmax="min", data=data, save_population=False, log_to="console")

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
        df.to_csv(f"{PATH_ERROR}{model_name}_{mode_name}_mlp_paras_tuning_error.csv", header=True, index=False)
        best_fit_full[mode_name] = best_fit_list

    df = DataFrame(best_fit_full)
    df.to_csv(f"{PATH_BEST_FIT}/{model_name}_mlp_paras_tuning_best_fit.csv", header=True, index=False)

    df_time = DataFrame(np.array(list_total_time), columns=["mode", "trial", "total_time"])
    df_time.to_csv(f"{PATH_BEST_FIT}/{model_name}_mlp_paras_tuning_total_time.csv", header=True, index=False)
