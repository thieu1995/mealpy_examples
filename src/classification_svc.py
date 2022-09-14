#!/usr/bin/env python
# Created by "Thieu" at 10:24, 14/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

## https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

# Assumption that we are trying to optimize the hyper-parameter of SVC model
# 1. C
# 2. Kernel


# Rules:
# x1. C: float [0.1 to 10000.0]
# x2. Kernel: [‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’]

## kernel = 'precomputed', will show the error with dataset, better to not use it.


# Solution = vector of float number = [ x1, x2 ]

# x1: is float number
# x2: need LabelEncoder to convert string into integer number, need to use int function


# univariate SVC example
from sklearn.svm import SVC
from permetrics.classification import ClassificationMetric
from mealpy.utils.problem import Problem


class ClassificationSVC(Problem):
    def __init__(self, lb, ub, minmax, data=None, name="Support Vector Classification", **kwargs):
        super().__init__(lb, ub, minmax, data=data, **kwargs)  ## data is needed because when initialize the Problem class, we need to check the output of fitness
        self.data = data
        self.name = name

    def decode_solution(self, solution):
        # C = solution[0]
        #
        # kernel_integer = int(solution[1])
        # kernel = KERNEL_ENCODER.inverse_transform([kernel_integer])[0]
        # 0 - 0.99 ==> 0 index ==> should be linear (for example)
        # 1 - 1.99 ==> 1 index ==> should be poly

        C = solution[0]
        kernel_integer = int(solution[1])
        kernel = self.data["KERNEL_ENCODER"].inverse_transform([kernel_integer])[0]
        return {
            "C": C,
            "kernel": kernel,
        }

    def generate_trained_model(self, structure):
        # print('Trying to generate trained model...')
        model = SVC(C=structure["C"], kernel=structure["kernel"])
        model.fit(self.data["X_train"], self.data["y_train"])
        # print("Return model")
        return model

    def generate_loss_value(self, structure):
        model = self.generate_trained_model(structure)

        # We take the loss value of validation set as a fitness value for selecting the best model demonstrate prediction
        y_pred = model.predict(self.data["X_test"])

        evaluator = ClassificationMetric(self.data["y_test"], y_pred, decimal=6)
        loss = evaluator.accuracy_score(average="macro")
        return loss

    def fit_func(self, solution):
        structure = self.decode_solution(solution)
        fitness = self.generate_loss_value(structure)
        return fitness
