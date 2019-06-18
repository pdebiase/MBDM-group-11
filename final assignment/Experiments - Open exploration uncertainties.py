import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ema_workbench import (MultiprocessingEvaluator,
                           Scenario, Constraint, Policy,
                           ScalarOutcome)
from ema_workbench.em_framework.optimization import EpsilonProgress, HyperVolume
from ema_workbench.em_framework.evaluators import perform_experiments, optimize
from ema_workbench.util import ema_logging, save_results, load_results
from ema_workbench.analysis import (pairs_plotting, prim, 
                                    feature_scoring, parcoords,
                                    dimensional_stacking, plotting)

from problem_formulation import get_model_for_problem_formulation

sns.set_style('white')

ema_logging.log_to_stderr(ema_logging.INFO)


'''PROBLEM FORMULATION
   ===================

   For different list of outcomes:
    0 = 2-objective PF
    1 = 3-objective PF
    2 = 5-objective PF
    3 = Disaggregate over locations
    4 = Disaggregate over time
    5 = Fully disaggregated
'''
problem_formulation_id = 2 # assign problem_formulation_id
num_planning_steps = 2     # assign number of planning steps
dike_model, planning_steps = get_model_for_problem_formulation(problem_formulation_id, 
                                                               num_planning_steps)

'''PERFORM EXPERIMENTS : RANDOM POLICIES
   =====================================
   
   For open exploration over uncertainties, 
   perform expriments for ten sampled polices
   over 1,000 sampled uncertainties
'''

with MultiprocessingEvaluator(dike_model) as evaluator:
    results = evaluator.perform_experiments(scenarios= 1000, policies = 10) # using lhs sampling

save_results(results, './results/open_exploration_uncertainties_10000runs.tar.gz')

results = load_results('./results/open_exploration_uncertainties_10000runs.tar.gz')

experiments, outcomes = results

'''PERFORM EXPRIMENTS : ZERO POLICY
   ================================

   Perform experiments for zero policy 
   using the same sampled scenarios

'''

# use the same 1,000 sampled scenarios, run for zero policy
sampled_scenarios = experiments.loc[:, [u.name for u in dike_model.uncertainties]]
sampled_scenarios.drop_duplicates(inplace=True)
ref_scenarios = [Scenario(i, **scenario.to_dict())
                 for i, scenario in pd.DataFrame.from_dict(sampled_scenarios).iterrows()] # sampled scenarios

policy0 = {'DikeIncrease': 0, 'DaysToThreat': 0, 'RfR': 0}
ref_policy = {}
for key in dike_model.levers:
    _, s = key.name.split('_')
    if ' ' in s:
        s, _ = s.split(' ')
    ref_policy.update({key.name: policy0[s]})                

ref_policy0 = Policy('baseline', **ref_policy) # policy0

with MultiprocessingEvaluator(dike_model) as evaluator:
    results = evaluator.perform_experiments(scenarios= ref_scenarios, policies = ref_policy0) # using lhs sampling

save_results(results, './results/open_exploration_uncertainties_policy0.tar.gz')