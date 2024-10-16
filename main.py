import simple_colors
import global_policy
import numpy as np
import display
import warnings
warnings.filterwarnings('ignore')

for domain in ['salp']:#, 'warehouse', 'taxi']:
    trials = 5 #100
    savenames = {   0: 'Task only',
                    1: 'Single Preference (Meta-ordering)', 
                    2: 'Scalarization Single Preference (Meta-ordering)',
                    3: 'LMDP (Wray et al. 2015)',
                    4: 'Yang et al. (2019)', 
                    5: 'Contextual Approach without Conflict Resolution',
                    6: 'Contextual Approach with Conflict Resolution'}
    sim_results = {}  # averaged over trials for each sim above 0-6
    R1_means_over_grids = [[] for _ in range(7)]
    R2_means_over_grids = [[] for _ in range(7)]
    R3_means_over_grids = [[] for _ in range(7)]
    reached_goal_percentage_over_grids = [[] for _ in range(7)]
    conflict_percentage_over_grids = [[] for _ in range(7)]
    percentile_stats = {i:{'mean': [], 'std': []} for i in range(len(savenames.keys()))}
    for context_sim in range(3,7):
        trajectories_o1, trajectories_o2, trajectories_o3 = [], [], []
        for grid_num in range(1):
            if domain == 'salp':
                from salp_mdp import SalpEnvironment, SalpAgent
                # Env = SalpEnvironment("grids/salp/illustration"+str(grid_num)+"_15x15.txt", context_sim)
                Env = SalpEnvironment("grids/salp/illustration_6x6.txt", context_sim)
                agent = SalpAgent(Env)
            elif domain == 'warehouse':
                from warehouse_mdp import WarehouseEnvironment, WarehouseAgent
                Env = WarehouseEnvironment("grids/warehouse/illustration"+str(grid_num)+"_15x15.txt", context_sim)
                agent = WarehouseAgent(Env)
            elif domain == 'taxi':
                from taxi_mdp import TaxiEnvironment, TaxiAgent
                Env = TaxiEnvironment("grids/taxi/illustration"+str(grid_num)+"_15x15.txt", context_sim)
                agent = TaxiAgent(Env)
            else:
                print(simple_colors.red('Invalid domain name!', ['bold']))
                break
            
            print(simple_colors.cyan('Context Simulation: ' + savenames[context_sim], ['bold', 'underlined']))
            agent, Pi_G = global_policy.get_global_policy(agent, context_sim)
            R1_stats, R2_stats, R3_stats, reached_goal_percentage, conflict_percentage, percentile_trajectories_for_objs = global_policy.get_multiple_rollout_states(agent, Pi_G, context_sim, trials)
            sim_results[context_sim] = [savenames[context_sim], R1_stats, R2_stats, R3_stats, reached_goal_percentage]
            R1_means_over_grids[context_sim].append(R1_stats[0])
            R2_means_over_grids[context_sim].append(R2_stats[0])
            R3_means_over_grids[context_sim].append(R3_stats[0])
            trajectories_o1.append(percentile_trajectories_for_objs[0])
            trajectories_o2.append(percentile_trajectories_for_objs[1])
            trajectories_o3.append(percentile_trajectories_for_objs[2])
            reached_goal_percentage_over_grids[context_sim].append(reached_goal_percentage)
            conflict_percentage_over_grids[context_sim].append(conflict_percentage)
            # display.report_sim_results(sim_results, trials, grid_num)
        # save in percentile_data/domain/context_sim/o1.txt,o2.txt,o3.txt as integer values
        
        # np.savetxt('percentile_data/'+domain+'/'+str(context_sim)+'/o1.txt', trajectories_o1, fmt='%d')
        # np.savetxt('percentile_data/'+domain+'/'+str(context_sim)+'/o2.txt', trajectories_o2, fmt='%d')
        # np.savetxt('percentile_data/'+domain+'/'+str(context_sim)+'/o3.txt', trajectories_o3, fmt='%d')
    print(simple_colors.green('Simulation completed!', ['bold', 'underlined']))
    display.report_sim_results_over_grids_and_trails([savenames, R1_means_over_grids, R2_means_over_grids, R3_means_over_grids, reached_goal_percentage_over_grids,conflict_percentage_over_grids], trials)    
    wait = input("Press Enter to continue...")
    