import simple_colors
import global_policy

def run_all_sims(domain, sim_name, trials):
    sim_results = {}  # averaged over trials for each sim above 0-7
    
    R1_means_over_grids = [[] for _ in range(8)]  # objective o1 reward for all sims in sim_name
    R2_means_over_grids = [[] for _ in range(8)]  # objective o2 reward for all sims in sim_name
    R3_means_over_grids = [[] for _ in range(8)]  # objective o3 reward for all sims in sim_name
    
    reached_goal_percentage_over_grids = [[] for _ in range(8)]
    conflict_percentage_over_grids = [[] for _ in range(8)]
    percentile_stats = {i:{'mean': [], 'std': []} for i in range(len(sim_name.keys()))}
    for context_sim in list(sim_name.keys()):
        trajectories_o1, trajectories_o2, trajectories_o3 = [], [], []
        for grid_num in range(5):
            if domain == 'salp':
                from salp_mdp import SalpEnvironment, SalpAgent
                Env = SalpEnvironment("grids/salp/illustration"+str(grid_num)+"_15x15.txt", context_sim)
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
            
            print(simple_colors.cyan('Context Simulation: ' + sim_name[context_sim], ['bold', 'underlined']))
            agent, Pi_G = global_policy.get_global_policy(agent, context_sim)
            agent.get_trajectory()
            Traj = agent.trajectory
            R1_stats, R2_stats, R3_stats, reached_goal_percentage, conflict_percentage, percentile_trajectories_for_objs = global_policy.get_multiple_rollout_states(agent, Pi_G, context_sim, trials)
            sim_results[context_sim] = [sim_name[context_sim], R1_stats, R2_stats, R3_stats, reached_goal_percentage]
            R1_means_over_grids[context_sim].append(R1_stats[0])
            R2_means_over_grids[context_sim].append(R2_stats[0])
            R3_means_over_grids[context_sim].append(R3_stats[0])
            trajectories_o1.append(percentile_trajectories_for_objs[0])
            trajectories_o2.append(percentile_trajectories_for_objs[1])
            trajectories_o3.append(percentile_trajectories_for_objs[2])
            reached_goal_percentage_over_grids[context_sim].append(reached_goal_percentage)
            conflict_percentage_over_grids[context_sim].append(conflict_percentage)

    return [sim_results, 
            R1_means_over_grids, R2_means_over_grids, R3_means_over_grids, 
            reached_goal_percentage_over_grids, conflict_percentage_over_grids]

    