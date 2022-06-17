#!/usr/bin/env python
# Created by "Thieu" at 17:44, 14/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.human_based import TLO
import numpy as np
np.random.seed(12345)

## Link: https://en.wikipedia.org/wiki/Test_functions_for_optimization
def fitness_function(pos):
    def f1(pos):
        return -10*(np.exp(-0.2*np.sqrt(pos[0]**2+pos[1]**2)) + np.exp(-0.2*np.sqrt(pos[1]**2+pos[2]**2)))
    def f2(pos):
        return np.sum([np.abs(x)**0.8 + 5*np.sin(x**3) for x in pos])
    obj1, obj2 = f1(pos), f2(pos)
    return [obj1, obj2]

problem = {
    "fit_func": fitness_function,
    "lb": [-5, -5, -5],
    "ub": [5, 5, 5],
    "minmax": "min",
    "obj_weights": [0.5, 0.5]
}

## Run the algorithm
model = TLO.BaseTLO(problem, epoch=100, pop_size=50)
model.solve()
print(f"Best solution: {model.solution[0]},\nBest target: {model.solution[1]}")

## You can access all of available figures via object "history" like this:
model.history.save_global_objectives_chart(filename="mo/goc")
model.history.save_local_objectives_chart(filename="mo/loc")
model.history.save_global_best_fitness_chart(filename="mo/gbfc")
model.history.save_local_best_fitness_chart(filename="mo/lbfc")
model.history.save_runtime_chart(filename="mo/rtc")
model.history.save_exploration_exploitation_chart(filename="mo/eec")
model.history.save_diversity_chart(filename="mo/dc", algorithm_name='TLO')
model.history.save_trajectory_chart(list_agent_idx=[6, 13, 24], filename="mo/tc")
