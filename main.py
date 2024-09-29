import simple_colors
import global_policy
import display

domain = 'salp' # options - ['salp', 'warehouse', 'taxi']
trials = 1000
savenames = {   0: 'Task only',
                1: 'Single Preference (Meta-ordering)', 
                2: 'Scalarization Single Preference (Meta-ordering)',
                3: 'Scalarization Contextual Preferences',
                4: 'Contextual Scalarization DNN', 
                5: 'Contextual Approach without Conflict Resolution',
                6: 'Contextual Approach with Conflict Resolution'}
sim_results = {}  # averaged over trials for each sim above 0-6
for context_sim in range(7):
    detected_counter = 0
    solved_counter = 0
    not_solved_counter = 0
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
    
    print(simple_colors.cyan('Context Simulation: ' + savenames[context_sim], ['bold', 'underlined']))
    print(agent)
    agent, Pi_G = global_policy.get_global_policy(agent, context_sim)
    R1_stats, R2_stats, R3_stats, reached_goal_percentage = global_policy.get_multiple_rollout_states(agent, Pi_G, context_sim, trials)
    sim_results[context_sim] = [savenames[context_sim], R1_stats, R2_stats, R3_stats, reached_goal_percentage]
display.report_sim_results(sim_results, trials)