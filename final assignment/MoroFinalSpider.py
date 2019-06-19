# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:02:02 2019

@author: georg
"""

SEED = 42 
import numpy.random
import random
numpy.random.seed(SEED)
random.seed(SEED)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functools
import time


from ema_workbench import (Model, RealParameter, Policy, Constant, Scenario, Constraint, 
                           ScalarOutcome, MultiprocessingEvaluator, ema_logging, perform_experiments)
from ema_workbench.util import ema_logging, save_results, load_results
from ema_workbench.em_framework.optimization import (EpsilonProgress, HyperVolume, ArchiveLogger)
from problem_formulation import get_model_for_problem_formulation
from ema_workbench.em_framework import sample_uncertainties
from ema_workbench.em_framework.evaluators import BaseEvaluator

%matplotlib inline  
# With this backend, the output of plotting commands is displayed inline within frontends like the Jupyter notebook, 
# directly below the code cell that produced it. The resulting plots will then also be stored in the notebook document.

ema_logging.log_to_stderr(ema_logging.INFO)
dike_model, planning_steps = get_model_for_problem_formulation(2) # assign problem_formulation_id

for levers in dike_model.levers:
   print(levers)
   

# percentile90 = functools.partial(np.percentile, q=90)
# def var_mean(data):
#     return ((np.percentile(data,q=75)-np.percentile(data,q=25))*np.sum(data)/(data.shape[0]))

def mean_rob(data):
    return np.sum(data)/(data.shape[0]*1e9)

def threshold(direction, threshold, data):
    if direction == SMALLER:
        return np.sum(data<=threshold)/data.shape[0]
    else:
        return np.sum(data>=threshold)/data.shape[0]

MAXIMIZE = ScalarOutcome.MAXIMIZE
MINIMIZE = ScalarOutcome.MINIMIZE 

SMALLER = "SMALLER"

death_funcs = functools.partial(threshold, SMALLER, 0.00001)
damage_funcs = functools.partial(threshold, SMALLER, 100000)

robustness_functions = [ScalarOutcome('Robustness metric Damage', kind=MAXIMIZE,
                             variable_name='Expected Annual Damage', function = damage_funcs),
                        ScalarOutcome('Robustness metric Dike Costs', kind=MINIMIZE,
                             variable_name='Dike Investment Costs', function = mean_rob),
                        ScalarOutcome('Robustness metric RfR Costs', kind=MINIMIZE,
                             variable_name='RfR Investment Costs', function = mean_rob),
                        ScalarOutcome('Robustness metric Evacuation Costs', kind=MINIMIZE,
                             variable_name='Evacuation Costs', function = mean_rob),
                        ScalarOutcome('Robustness metric Deaths', kind=MAXIMIZE,
                             variable_name='Expected Number of Deaths', function = death_funcs)]
                        


start = time.time()

n_scenarios = 2
nfe = 6000
scenarios = sample_uncertainties(dike_model, n_scenarios)
epsilons = [0.01, 0.01, 0.01, 0.01, 0.01]

BaseEvaluator.reporting_frequency = 15

# is the order of the hypervolume according to the order of the outcomes? 
convergence_metrics = [HyperVolume(minimum=[0,0,0,0,0], maximum=[1.1, 3 , 3, 3, 1.1]),
                       EpsilonProgress()]
 
constraint = [Constraint("Death constraint", outcome_names="Robustness metric Deaths", function=lambda x:max(0, 0.75-x))]

with MultiprocessingEvaluator(dike_model) as evaluator:
    archive, convergence = evaluator.robust_optimize(robustness_functions, scenarios,
                            nfe=nfe, convergence=convergence_metrics, epsilons=epsilons, constraints=constraint, logging_freq = 1, convergence_freq = 10)

archive.to_csv(f"Outputs/robust_optimization/Archive{n_scenarios}scen{nfe}nfe{epsilons}")
convergence.to_csv(f"Outputs/robust_optimization/Convergence{n_scenarios}scen{nfe}nfe{epsilons}")

# run your code
end = time.time()
elapsed = end - start



archive = pd.read_csv(f"Outputs/robust_optimization/Archive{n_scenarios}scen{nfe}nfe{epsilons}", index_col=0)
convergence = pd.read_csv(f"Outputs/robust_optimization/Convergence{n_scenarios}scen{nfe}nfe{epsilons}", index_col=0)

# check the found solutions
levers = archive.loc[:, [l.name for l in dike_model.levers]]
levers.T

policies_to_evaluate = [Policy('policy'+str(i), **policy.to_dict()) 
                        for i, policy in pd.DataFrame.from_dict(levers).iterrows()]
# policies_to_evaluate


fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, figsize=(8,4))
ax1.plot(convergence.nfe, convergence.epsilon_progress)
ax1.set_ylabel('$\epsilon$-progress')
ax2.plot(convergence.nfe, convergence.hypervolume)
ax2.set_ylabel('hypervolume')

ax1.set_xlabel('number of function evaluations')
ax2.set_xlabel('number of function evaluations')
plt.show()