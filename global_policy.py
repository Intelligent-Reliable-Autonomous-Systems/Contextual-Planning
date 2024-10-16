import numpy as np
import value_iteration
import simple_colors
import metareasoner as MR

savenames = {   0: 'Task only',
                1: 'Single Preference (Meta-ordering)', 
                2: 'Scalarization Single Preference (Meta-ordering)',
                3: 'Wrays et al. (2015)',
                4: 'Contextual Scalarization DNN', 
                5: 'Contextual Approach without Conflict Resolution',
                6: 'Contextual Approach with Conflict Resolution'}

def get_global_policy(agent, context_sim):
    '''Get the global policy for the agent for a given context simulation
    1. Get the global policy for the agent for the given context simulation
    2. Check for conflicts in the global policy
    3. If there are conflicts, resolve them
    4. Return the global policy
    '''
    if context_sim == 0:
        agent, Pi_G = value_iteration.value_iteration(agent, agent.Grid.R_obs[0])
    elif context_sim == 1:
        agent, Pi_G = value_iteration.lexicographic_value_iteration(agent, agent.Grid.f_w(agent.Grid.OMEGA[0])) 
    if context_sim == 2:
        agent, Pi_G = value_iteration.scalarized_value_iteration(agent,agent.Grid.f_w(agent.Grid.OMEGA[0]))
    elif context_sim == 3:
        agent, Pi_G = value_iteration.LMDP(agent)
    elif context_sim == 4:
        agent, Pi_G = agent.get_contextual_scalarized_dnn_policy()
    elif context_sim == 5:
        agent, Pi_G = value_iteration.contextual_lexicographic_value_iteration(agent)
    elif context_sim == 6:
        agent, Pi_G = value_iteration.contextual_lexicographic_value_iteration(agent)
    agent.Pi_G = Pi_G
    conflict = MR.conflict_checker(Pi_G, agent)
    if conflict:
        print(simple_colors.red('Conflict Detected!', ['bold']) )
    else: 
        print(simple_colors.green('No Conflicts', ['bold']))
    if context_sim == 6 and conflict:
        Pi_G = MR.conflict_resolver(Pi_G, agent)
        conflict = MR.conflict_checker(Pi_G, agent)       
        if conflict: 
            print(simple_colors.red('Conflict count not be resolved!', ['bold']) )
        else: 
            print(simple_colors.green('Conflict resolved!', ['bold']))
    agent.Pi_G = Pi_G
    return agent, Pi_G

def get_rollout(agent, Pi_G, context_sim, display_trajecotry=False):
    '''Get the rollout for the agent for a given context simulation
    returns the rewards in all objectives and whether the agent reached the goal'''
    agent.reset()
    reached_goal = False
    agent.follow_policy_rollout(Pi_G)
    conflict = MR.conflict_checker(Pi_G, agent)
    if display_trajecotry:
        print(simple_colors.yellow('Trajectory for '+savenames[context_sim]+':', ['bold']))
        for i in range(len(agent.trajectory)):
            print(simple_colors.yellow(str(agent.trajectory[i])))
    if agent.s == agent.s_goal:
        reached_goal = True
        r1, r2, r3 = agent.r_1, agent.r_2, agent.r_3
        # print(simple_colors.green(savenames[context_sim]+' reached the goal!', ['bold']))
    else:
        reached_goal = False
        r1, r2, r3 = 0, 0, 0
        # print(simple_colors.red(savenames[context_sim]+' did not reach the goal!', ['bold']))
    return r1, r2, r3, reached_goal, conflict

def get_multiple_rollout_states(agent, Pi_G, context_sim, trials, display_trajectory=False):
    '''Get the rollout stats for the agent for a given context simulation for multiple trials
    returns the mean and standard deviation of the rewards in all objectives as a list of 2 and the number of times the agent reached the goal'''
    R1 = []
    R2 = []
    R3 = []
    max_values = {'o_1': 76, 'o_2': 100, 'o_3': 100}
    percentile_trajectories_for_objs = {i:[] for i in [0,1,2]} # to record the number of times the agent reached a certain percentile of the maximum value as an objective based dict percentile_traj[o1] = [...]
    reached_goal_counter = 0
    conflict_counter = 0
    for rollout in range(trials):
        r1, r2, r3, reached_goal, conflict = get_rollout(agent, Pi_G, context_sim, display_trajectory)
        if reached_goal:
            reached_goal_counter += 1
            conflict_counter += int(conflict)
            R1.append(r1)
            R2.append(r2)
            R3.append(r3)
        else:
            conflict_counter += int(conflict)            
            R1.append(0)
            R2.append(0)
            R3.append(0)
    # computing mean and standard deviation over all rollouts
    R1_stats = [round(sum(R1)/trials,2), round(np.std(R1),1)]
    R2_stats = [round(sum(R2)/trials,2), round(np.std(R2),1)]
    R3_stats = [round(sum(R3)/trials,2), round(np.std(R3),1)]
    for i in range(0, 101,10):
        percentile_trajectories_for_objs[0].append(sum([int(R1[j] > i*max_values['o_1']/100) for j in range(len(R1))]))
        percentile_trajectories_for_objs[1].append(sum([int(R2[j] > i*max_values['o_2']/100) for j in range(len(R2))]))
        percentile_trajectories_for_objs[2].append(sum([int(R3[j] > i*max_values['o_3']/100) for j in range(len(R3))]))
        
    return R1_stats, R2_stats, R3_stats, round(reached_goal_counter/ trials * 100, 1), round(conflict_counter/ trials * 100, 1), percentile_trajectories_for_objs

def visualize_global_policy(domain, agent, Pi_G, context_sim):
    '''Visualize the global policy for the agent for a given context simulation
    1. Visualize the global policy for the agent for the given context simulation
    '''
    import display
    savenames = {0: 'Task only',
                1: 'Task > NSE Mitigation', 
                2: 'NSE Mitigation > Task', 
                3: 'Scalarization Single Preference',
                4: 'Scalarization Contextual Preferences',
                5: 'Contextual Scalarization DNN', 
                6: 'Contextual Approach without Conflict Resolution',
                7: 'Contextual Approach with Conflict Resolution'}
    if domain == 'salp':
        display.animate_policy_salp(agent, Pi_G, savenames[context_sim], stochastic_transition=False)#, savenames[context_sim]) 
    elif domain == 'warehouse':
        display.animate_policy_warehouse(agent, Pi_G, savenames[context_sim], stochastic_transition=False)#, savenames[context_sim]) 
    elif domain == 'taxi':
        display.animate_policy_taxi(agent, Pi_G, savenames[context_sim], stochastic_transition=False)#, savenames[context_sim]) 
    wait = input("Press Enter to continue...")
