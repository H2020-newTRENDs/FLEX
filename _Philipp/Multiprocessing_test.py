import multiprocessing

import pyomo.environ as pyo
import numpy as np

from pathos.multiprocessing import ProcessingPool
from multiprocessing import Pool


def create_model():
    m = pyo.AbstractModel()
    m.t = pyo.Set()

    m.x = pyo.Param(m.t, mutable=True)

    m.y = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.z = pyo.Var(m.t, within=pyo.NonNegativeReals)

    def constraint(m, t):
        return m.y[t] == 2 * m.z[t]

    m.constraint = pyo.Constraint(m.t, rule=constraint)

    def objective(m):
        return sum(m.x[t] * m.y[t] - m.x[t] * m.z[t] for t in m.t)

    m.Objective = pyo.Objective(rule=objective)

    return m


def create_instance():
    pyomo_dict = {None: {"t": {None: np.arange(1, 4)},
                         "x": {1: 2, 2: 3, 3: 1}}}

    model = create_model()
    instance = model.create_instance(data=pyomo_dict)
    return instance

def update_instance(pyomo_dictionary, instance):
    # pyomo_dictionary, instance = total_list[0], total_list[1]
    # update instance
    for t in range(1, 4):
        instance.x[t] = pyomo_dictionary[None]["x"][t]

    # solve instance
    Opt = pyo.SolverFactory("gurobi")
    result = Opt.solve(instance, tee=True)
    print("update instance done")
    return result



def multi_proces():
    pyomo_dict1 = {None: {"t": {None: np.arange(1, 4)},
                          "x": {1: 2, 2: 3, 3: 1}}}

    pyomo_dict2 = {None: {"t": {None: np.arange(1, 4)},
                          "x": {1: 1, 2: 1, 3: 1}}}

    pyomo_dict3 = {None: {"t": {None: np.arange(1, 4)},
                          "x": {1: 3, 2: 3, 3: 3}}}

    instance_empty = create_instance()
    input_parameter_list = [pyomo_dict1, pyomo_dict2, pyomo_dict3]
    instance_list = [instance_empty] * len(input_parameter_list)
    fette_liste = [input_parameter_list, instance_list]

    # test_liste = [input_parameter_list[0], instance_list[0]]
    # update_instance(test_liste)
    # pool_multi = Pool(processes=2)
    # result = pool_multi.map(update_instance, fette_liste)

    pool_pathos = ProcessingPool(2)
    # Pool_parallel = ParallelPool(nodes=number_of_cpus)  # provide number of cores
    ergebnis = pool_pathos.map(update_instance, input_parameter_list, instance_list)
    pool_pathos.close()



if __name__=="__main__":
    multi_proces()

