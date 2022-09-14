#!/usr/bin/env python
# Created by "Thieu" at 17:44, 14/06/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from mealpy.swarm_based import WOA
import numpy as np
np.random.seed(12345)

def fitness_func(solution):
    return np.sum(solution**2)

problem_dict = {
    "fit_func": fitness_func,
    "lb": [-10, ] * 5,
    "ub": [10, ] * 5,
    "minmax": "min",
    "save_population": True,
    "log_to": "file",
    "log_file": "history/so/result.log"
}

model = WOA.OriginalWOA(epoch=100, pop_size=50)
best_position, best_fitness_value = model.solve(problem_dict)

print(best_position)
print(best_fitness_value)

model.history.save_global_best_fitness_chart(filename="history/so/gbfc")
model.history.save_local_best_fitness_chart(filename="history/so/lbfc")
model.history.save_diversity_chart(filename="history/so/dc", algorithm_name='WOA')
model.history.save_global_objectives_chart(filename="history/so/goc")
model.history.save_local_objectives_chart(filename="history/so/loc")
model.history.save_exploration_exploitation_chart(filename="history/so/eec")
model.history.save_runtime_chart(filename="history/so/rc")
model.history.save_trajectory_chart(filename="history/so/tc")

