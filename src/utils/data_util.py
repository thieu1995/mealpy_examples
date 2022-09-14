#!/usr/bin/env python
# Created by "Thieu" at 18:00, 13/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# univariate mlp example
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    ## https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def generate_time_series_data(train_ratio=0.75):
    ## Make dataset
    dataset = pd.read_csv("data/monthly-sunspots.csv", usecols=["Sunspots"])
    scaler = MinMaxScaler()
    scaled_seq = scaler.fit_transform(dataset.values).flatten()

    # choose a number of time steps
    n_steps = 3
    # split into samples            60% - training
    x_train_point = int(len(scaled_seq) * train_ratio)
    X_train, y_train = split_sequence(scaled_seq[:x_train_point], n_steps)
    X_test, y_test = split_sequence(scaled_seq[x_train_point:], n_steps)

    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test, "n_steps": n_steps}


def generate_data_classification_data(test_ratio=0.3):
    # Load the data set; In this example, the breast cancer dataset is loaded.
    bc = datasets.load_breast_cancer()
    X = bc.data
    y = bc.target

    # Create training and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=1, stratify=y)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return {"X_train": X_train_std, "y_train": y_train, "X_test": X_test_std, "y_test": y_test, "scaler": sc}
