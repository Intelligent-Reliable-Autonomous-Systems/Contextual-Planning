import simple_colors
import display
import value_iteration
import oracle_policy
import metareasoner as MR
from timeit import default_timer as timer

domain = 'taxi' # options - ['salp', 'warehouse', 'taxi']
context_sim = 1 # options - [0, 1, 2, 3, 4, 5, 6]

for context_sim in range(6):
    if domain == 'salp':
        from salp_mdp import SalpEnvironment, SalpAgent
        Env = SalpEnvironment("grids/salp/illustration_eddy.txt", context_sim)
        agent = SalpAgent(Env)
    elif domain == 'warehouse':
        from warehouse_mdp import WarehouseEnvironment, WarehouseAgent
        Env = WarehouseEnvironment("grids/warehouse/illustration.txt", context_sim)
        agent = WarehouseAgent(Env)
    elif domain == 'taxi':
        from taxi_mdp import TaxiEnvironment, TaxiAgent
        Env = TaxiEnvironment("grids/taxi/illustration.txt", context_sim)
        agent = TaxiAgent(Env)
    else:
        print(simple_colors.red('Invalid domain name!', ['bold']))
        break
    savenames = {0: 'Task only',
                1: 'Task > NSE Mitigation', 
                2: 'NSE Mitigation > Task', 
                3: 'Scalarization',
                4: 'Contextual Approach without Conflict Resolution', 
                5: 'Contextual Approach with Conflict Resolution',
                6: 'Oracle'}

    print(simple_colors.cyan('Context Simulation: ' + savenames[context_sim], ['bold', 'underlined']))
    if context_sim == 6:
        Pi_G = oracle_policy.contextual_lexicographic_value_iteration_oracle(agent)
    elif context_sim == 3:
        agent, Pi_G = value_iteration.contextual_scalarized_value_iteration(agent)
    else:
        agent, Pi_G = value_iteration.contextual_lexicographic_value_iteration(agent)
    agent.Pi_G = Pi_G
    conflict = MR.conflict_checker(Pi_G, agent)
    if conflict: print(simple_colors.red('Conflict Detected!', ['bold']) )
    else: print(simple_colors.green('No Conflicts', ['bold']))
    if context_sim == 5:
        Pi_G = MR.conflict_resolver(Pi_G, agent)
        conflict = MR.conflict_checker(Pi_G, agent)
        if conflict: print(simple_colors.red('Conflict Detected!', ['bold']) )
        else: print(simple_colors.green('No Conflicts', ['bold']))
    if domain == 'salp':
        display.animate_policy_salp(agent, Pi_G, savenames[context_sim], stochastic_transition=False)#, savenames[context_sim]) 
    elif domain == 'warehouse':
        display.animate_policy_warehouse(agent, Pi_G, savenames[context_sim], stochastic_transition=False)#, savenames[context_sim]) 
    elif domain == 'taxi':
        display.animate_policy_taxi(agent, Pi_G, savenames[context_sim], stochastic_transition=False)#, savenames[context_sim]) 
    wait = input("Press Enter to continue...")