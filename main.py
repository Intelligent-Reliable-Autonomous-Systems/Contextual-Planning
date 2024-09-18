import simple_colors
import display
import value_iteration
import oracle_policy
import metareasoner as MR
from salp_mdp import SalpEnvironment, SalpAgent
from timeit import default_timer as timer

context_sim = 1 # options - [0, 1, 2, 3, 4, 5]
for context_sim in range(5):
    savenames = {0: 'Task only',
                1: 'Task > NSE Mitigation', 
                2: 'NSE Mitigation > Task', 
                3: 'Contextial Approach without Conflict Resolution', 
                4: 'Contextial Approach with Conflict Resolution',
                5: 'Oracle'}

    print(simple_colors.cyan('Context Simulation: ' + savenames[context_sim], ['bold', 'underlined']))
    Env = SalpEnvironment("grids/salp/illustration_eddy.txt", context_sim)
    agent = SalpAgent(Env)
    if context_sim == 5:
        Pi_G = oracle_policy.contextual_lexicographic_value_iteration_oracle(agent)
    else:
        agent, Pi_G = value_iteration.contextual_lexicographic_value_iteration(agent)
    agent.Pi_G = Pi_G
    conflict = MR.conflict_checker(Pi_G, agent)
    if conflict: print(simple_colors.red('Conflict Detected!', ['bold']) )
    else: print(simple_colors.green('No Conflicts', ['bold']))
    if context_sim == 4:
        Pi_G = MR.conflict_resolver(Pi_G, agent)
        conflict = MR.conflict_checker(Pi_G, agent)
        if conflict: print(simple_colors.red('Conflict Detected!', ['bold']) )
        else: print(simple_colors.green('No Conflicts', ['bold']))
    # print(Pi_G)
    display.animate_policy(agent, Pi_G, stochastic_transition=False)#, savenames[context_sim]) 
    wait = input("Press Enter to continue...")