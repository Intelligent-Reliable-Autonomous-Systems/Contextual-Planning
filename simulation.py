import simple_colors
import global_policy

def run_all_sims(domain, sim_names, trials):
    sim_results = {}  # averaged over trials for each sim above 0-7
    
    R1_means = [[] for _ in range(8)]  # objective o1 reward for all sims in sim_names
    R2_means = [[] for _ in range(8)]  # objective o2 reward for all sims in sim_names
    R3_means = [[] for _ in range(8)]  # objective o3 reward for all sims in sim_names
    
    reached_goal_percentage_over_grids = [[] for _ in range(8)]
    conflict_percentage_over_grids = [[] for _ in range(8)]
    for context_sim in list(sim_names.keys()):
        for grid_num in range(5):  # 5 grids for each domain
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
            if grid_num == 0:
                print(simple_colors.blue('\nSIM: ' + sim_names[context_sim], ['bold', 'underlined']))
                print(simple_colors.yellow('['+domain+' env ' + str(grid_num)+ ']', ['bold', 'underlined']))
            else:
                print(simple_colors.yellow('['+domain+' env ' + str(grid_num)+ ']', ['bold', 'underlined']))
            agent, Pi_G = global_policy.get_global_policy(agent, context_sim)
            agent.get_trajectory()
            R1_stats, R2_stats, R3_stats, reached_goal_percentage, conflict_percentage = global_policy.get_multiple_rollout_states(agent, Pi_G, context_sim, trials)
            R1_means[context_sim].append(R1_stats[0])  # [R1_grid1, R1_grid2, R1_grid3, R1_grid4, R1_grid5]
            R2_means[context_sim].append(R2_stats[0])  # [R2_grid1, R2_grid2, R2_grid3, R2_grid4, R2_grid5]
            R3_means[context_sim].append(R3_stats[0])  # [R3_grid1, R3_grid2, R3_grid3, R3_grid4, R3_grid5]
            reached_goal_percentage_over_grids[context_sim].append(reached_goal_percentage)
            conflict_percentage_over_grids[context_sim].append(conflict_percentage)
        sim_results[context_sim] = [R1_means[context_sim], R2_means[context_sim], R3_means[context_sim], reached_goal_percentage_over_grids[context_sim], conflict_percentage_over_grids[context_sim]]
    return sim_results

    