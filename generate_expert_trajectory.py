import simple_colors
import global_policy
import warnings
warnings.filterwarnings('ignore')

for domain  in ['salp']:#, 'warehouse', 'taxi']:
    savenames = {   0: 'Task only',
                    1: 'Single Preference (Meta-ordering)', 
                    2: 'Scalarization Single Preference (Meta-ordering)',
                    3: 'Scalarization Contextual Preferences',
                    4: 'Yang et al. (2019)', 
                    5: 'Contextual Approach without Conflict Resolution',
                    6: 'Contextual Approach with Conflict Resolution'}
    sim_results = {}  # averaged over trials for each sim above 0-6
    R1_means_over_grids = [[] for _ in range(7)]
    R2_means_over_grids = [[] for _ in range(7)]
    R3_means_over_grids = [[] for _ in range(7)]
    reached_goal_percentage_over_grids = [[] for _ in range(7)]
    for context_sim in range(7):
        if context_sim != 6:
            continue
        if domain == 'salp':
            from salp_mdp import SalpEnvironment, SalpAgent
            Env = SalpEnvironment("grids/salp/illustration_6x6.txt", context_sim)
            agent = SalpAgent(Env)
        elif domain == 'warehouse':
            from warehouse_mdp import WarehouseEnvironment, WarehouseAgent
            Env = WarehouseEnvironment("grids/warehouse/illustration0_15x15.txt", context_sim)
            agent = WarehouseAgent(Env)
        elif domain == 'taxi':
            from taxi_mdp import TaxiEnvironment, TaxiAgent
            Env = TaxiEnvironment("grids/taxi/illustration0_15x15.txt", context_sim)
            agent = TaxiAgent(Env)
        else:
            print(simple_colors.red('Invalid domain name!', ['bold']))
            break
        
        print(simple_colors.cyan('Context Simulation: ' + savenames[context_sim], ['bold', 'underlined']))
        agent, Pi_G = global_policy.get_global_policy(agent, context_sim)
        Tau = []
        counter = 0
        while counter < 10:
            agent.reset()
            trajectory = agent.get_trajectory(Pi_G)
            if trajectory[-1][0] == agent.s_goal:
                counter += 1
                print('trajectory ',counter, 'added')
                for data in trajectory:
                    Tau.append(data)
        with open('expert_trajectories.txt', 'w') as file:
            file.write("State\tAction\tR1\tR2\tR3\tContext\n")  # Writing the header
            for data in Tau:
                file.write("{} {} {} {} {} {}\n".format(data[0], data[1], data[2], data[3], data[4], data[5]))
