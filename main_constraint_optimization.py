#!/usr/bin/env python
# Created by "Thieu" at 18:24, 14/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.evolutionary_based import DE
import numpy as np
np.random.seed(12345)

## Link: https://en.wikipedia.org/wiki/Test_functions_for_optimization
def fitness_function(pos):
    def fxy(pos):
        return (1-pos[0])**2 + 100*(pos[1] - pos[0]**2)**2
    def g1(pos):
        return (pos[0]-1)**3 - pos[1] + 1
    def g2(pos):
        return pos[0] + pos[1] - 2

    def penalty(value):
        if 0 < value < 1:
            return value
        elif value >= 1:
            return value**2
        else:
            return 0

    fitness = fxy(pos) + penalty(g1(pos)) + penalty(g2(pos))
    return fitness

problem = {
    "fit_func": fitness_function,
    "lb": [-1.5, -0.5],
    "ub": [1.5, 2.5],
    "minmax": "min",
    "save_population": True,
}

## Run the algorithm
model = DE.BaseDE(epoch=100, pop_size=50)
best_position, best_fitness_value = model.solve(problem)
print(f"Best solution: {best_position}, Best target: {best_fitness_value}")

## You can access all of available figures via object "history" like this:
model.history.save_global_objectives_chart(filename="history/co/goc")
model.history.save_local_objectives_chart(filename="history/co/loc")
model.history.save_global_best_fitness_chart(filename="history/co/gbfc")
model.history.save_local_best_fitness_chart(filename="history/co/lbfc")
model.history.save_runtime_chart(filename="history/co/rtc")
model.history.save_exploration_exploitation_chart(filename="history/co/eec")
model.history.save_diversity_chart(filename="history/co/dc", algorithm_name='DE')
model.history.save_trajectory_chart(list_agent_idx=[3, 7, 10, 15], filename="history/co/tc")
