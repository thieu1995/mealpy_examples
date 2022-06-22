#!/usr/bin/env python
# Created by "Thieu" at 09:57, 12/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.swarm_based import GWO
from mealpy.evolutionary_based import FPA
from src.timeseries_util import decode_solution, generate_dataset, generate_loss_value


def fitness_function(solution, data):
    structure = decode_solution(solution, data)
    fitness = generate_loss_value(structure, data)
    return fitness


## Make this variable as global, because it won't change during the optimization process
DATA = generate_dataset()

LB = [1, 7, 0, 0.01, 0, 0, 5]
UB = [3.99, 10.99, 6.99, 0.5, 7.99, 7.99, 50]
problem = {
    "fit_func": fitness_function,
    "lb": LB,
    "ub": UB,
    "minmax": "min",
    "data": DATA
}

# model = GWO.BaseGWO(problem, epoch=5, pop_size=20)
model = FPA.BaseFPA(problem, epoch=50, pop_size=20)
model.solve()

print(f"Best solution: {model.solution[0]}")
structure = decode_solution(model.solution[0])

print(f"Batch-size: {structure['batch_size']}, Epoch: {structure['epoch']}, "
      f"Opt: {structure['opt']}, Learning-rate: {structure['learning_rate']}")
print(f"NWI: {structure['network_weight_initial']}, "
      f"Activation: {structure['activation']}, n-hidden: {structure['n_hidden_units']}")
