import math
import warnings
import simple_colors
import display
import value_iteration
import metareasoner as MR
from salp_mdp import SalpEnvironment, SalpAgent
from timeit import default_timer as timer

Env = SalpEnvironment("grids/salp/illustration.txt")
agent = SalpAgent(Env)
context_sim = 3 # options - 1 or 2 or 3 or 4
context_ordering = {1: [[0, 1], [0, 1]] , 2: [[1, 0], [1, 0]], 3: [[0, 1], [1, 0]], 4: [[0, 1], [1, 0]]}
savenames = {1: 'Task > NSE Mitigation', 
             2: 'NSE Mitigation > Task', 
             3: 'Contextial Approach without Conflict Resolution', 
             4: 'Contextial Approach with Conflict Resolution'}


# agent.V, agent.Pi = value_iteration.value_iteration(agent, Env.Reward_for_obj[0])
Env.contextual_orderings = context_ordering[context_sim]
agent, Pi_G = value_iteration.contextual_lexicographic_value_iteration(agent)
conflict = MR.conflict_checker(agent.Pi_G, agent)
if conflict: print(simple_colors.red('Conflict: ' + str(conflict), ['bold']) )
else: print(simple_colors.green('Conflict: ' + str(conflict), ['bold']))
display.animate_policy(agent, Pi_G, stochastic_transition=True)#, savenames[context_sim]) 
# MR.conflict_resolver(agent.Pi_G, agent)