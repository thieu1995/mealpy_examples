#!/usr/bin/env python
# Created by "Thieu" at 10:11, 17/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# univariate mlp example
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import optimizers
import statsmodels.datasets.co2 as co2


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


# LABEL ENCODER
OPT_ENCODER = LabelEncoder()
OPT_ENCODER.fit(['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'])  # domain range ==> 7 values

WOI_ENCODER = LabelEncoder()
WOI_ENCODER.fit(['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'])

ACT_ENCODER = LabelEncoder()
ACT_ENCODER.fit(['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'])


def decode_solution(solution):
    batch_size = 2 ** int(solution[0])
    epoch = 10 * int(solution[1])
    opt_integer = int(solution[2])
    opt = OPT_ENCODER.inverse_transform([opt_integer])[0]
    learning_rate = solution[3]
    network_weight_initial_integer = int(solution[4])
    network_weight_initial = WOI_ENCODER.inverse_transform([network_weight_initial_integer])[0]
    act_integer = int(solution[5])
    activation = ACT_ENCODER.inverse_transform([act_integer])[0]
    n_hidden_units = int(solution[6])
    return {
        "batch_size": batch_size,
        "epoch": epoch,
        "opt": opt,
        "learning_rate": learning_rate,
        "network_weight_initial": network_weight_initial,
        "activation": activation,
        "n_hidden_units": n_hidden_units,
    }


def generate_loss_value(structure, data):
    # define model
    model = Sequential()
    model.add(Dense(structure["n_hidden_units"], activation=structure["activation"],
                    input_dim=data["n_steps"], kernel_initializer=structure["network_weight_initial"]))
    model.add(Dense(1))

    # Compile model
    optimizer = getattr(optimizers, structure["opt"])(learning_rate=structure["learning_rate"])
    model.compile(optimizer=optimizer, loss='mse')

    # fit model
    model.fit(data["X_train"], data["y_train"], epochs=structure["epoch"], batch_size=structure["batch_size"], verbose=0)

    # We take the loss value of validation set as a fitness value for selecting the best model
    # demonstrate prediction
    yhat = model(data["X_test"])
    return mean_squared_error(data["y_test"], yhat)


def generate_dataset():
    ## Make dataset
    dataset = co2.load(as_pandas=True).data
    dataset = dataset.fillna(dataset.interpolate())
    scaler = MinMaxScaler()
    scaled_seq = scaler.fit_transform(dataset.values).flatten()

    # choose a number of time steps
    n_steps = 3
    # split into samples            60% - training
    x_train_point = int(len(scaled_seq) * 0.75)
    X_train, y_train = split_sequence(scaled_seq[:x_train_point], n_steps)
    X_test, y_test = split_sequence(scaled_seq[x_train_point:], n_steps)

    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test, "n_steps": n_steps}



