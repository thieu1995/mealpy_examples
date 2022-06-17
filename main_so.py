

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
    "log_to": "file",
    "log_file": "result.log"
}

model = WOA.BaseWOA(problem_dict, epoch=100, pop_size=50)
best_position, best_fitness_value = model.solve()

print(best_position)
print(best_fitness_value)

model.history.save_global_best_fitness_chart()
model.history.save_local_best_fitness_chart()
model.history.save_diversity_chart(algorithm_name='WOA')
model.history.save_global_objectives_chart()
model.history.save_local_objectives_chart()
model.history.save_exploration_exploitation_chart()
model.history.save_runtime_chart()
model.history.save_trajectory_chart()

