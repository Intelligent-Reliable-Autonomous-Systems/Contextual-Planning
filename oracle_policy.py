import copy
import math
import simple_colors

def value_iteration_last_obj(agent, Reward, A):
    '''
    Returns the value function and policy for the given agent and reward function R(s,a).
    '''
    S = agent.S
    A = agent.A
    V = {s: 0 for s in S}
    Pi = {s: None for s in S}
    gamma = 0.99
    residual = {s: 0 for s in S}
    Q = {s: {a: 0 for a in A[s]} for s in S}
    iter = 0
    while True:
        V_prev = copy.deepcopy(V)
        for s in S:
            if s == agent.s_goal:
                V[s] = Reward(s, 'Noop')
                Pi[s] = 'Noop'
                residual[s] = abs(V[s] - V_prev[s])
                continue
            for a in A[s]:
                T = agent.get_transition_prob(s, a)
                Q[s][a] = Reward(s, a) + gamma * sum([T[s_prime] * V[s_prime] for s_prime in list(T.keys())])
            V[s] = max(Q[s].values())
            Pi[s] = max(Q[s], key=Q[s].get)
            residual[s] = abs(V[s] - V_prev[s])
        if max(residual.values()) < 1e-6 or iter > 1000:
            # print(simple_colors.blue('Value Iteration converged in {} iterations.\n'.format(iter)))
            break
        iter += 1
    return V, Pi

def Q_value_iteration(agent, Reward):
    '''
    Returns the Q value function
    '''
    S = agent.S
    A = agent.A
    V = {s: 0 for s in S}
    Pi = {s: None for s in S}
    gamma = 0.99
    residual = {s: 0 for s in S}
    Q = {s: {a: 0 for a in A[s]} for s in S}
    iter = 0
    while True:
        V_prev = copy.deepcopy(V)
        for s in S:
            if s == agent.s_goal:
                V[s] = Reward(s, 'Noop')
                Q[s]['Noop'] = V[s]
                Pi[s] = 'Noop'
                residual[s] = abs(V[s] - V_prev[s])
                continue
            for a in A[s]:
                T = agent.get_transition_prob(s, a)
                Q[s][a] = Reward(s, a) + gamma * sum([T[s_prime] * V[s_prime] for s_prime in list(T.keys())])
            V[s] = max(Q[s].values())
            Pi[s] = max(Q[s], key=Q[s].get)
            residual[s] = abs(V[s] - V_prev[s])
        if max(residual.values()) < 1e-6 or iter > 1000:
            # print(simple_colors.blue('Value Iteration converged in {} iterations.\n'))
            break
        iter += 1
    return Q


def action_set_value_iteration(agent, Reward, A):
    '''
    returns the agent with a pruned action set based on all optimal policy space of for provided Reward function R(s,a).
    '''
    S = agent.S
    V = {s: 0 for s in S}
    Pi = {s: None for s in S}
    gamma = 0.99
    residual = {s: 0 for s in S}
    Q = {s: {a: 0 for a in A[s]} for s in S}
    iter = 1
    while True:
        V_prev = copy.deepcopy(V)
        for s in S:
            if s == agent.s_goal:
                V[s] = Reward(s, 'Noop')
                Q[s]['Noop'] = V[s]
                Pi[s] = 'Noop'
                residual[s] = abs(V[s] - V_prev[s])
                continue
            for a in A[s]:
                T = agent.get_transition_prob(s, a)
                Q[s][a] = Reward(s, a) + gamma * sum([T[s_prime] * V[s_prime] for s_prime in list(T.keys())])
            V[s] = max(Q[s].values())
            Pi[s] = max(Q[s], key=Q[s].get)
            residual[s] = abs(V[s] - V_prev[s])
        if max(residual.values()) < 1e-6 or iter > 1000:
            # print(simple_colors.green('Action Set Value Iteration converged in {} iterations.\n'.format(iter)))
            break
        iter += 1
    # Prune the action set
    A_pruned = {s: [] for s in S}
    for s in S:
        A_pruned[s] = [a for a in A[s] if (Q[s][a]) >= (V[s])]
    return A_pruned

def lexicographic_value_iteration_oracle(agent, ordering, A_pruned):
    '''
    params:
    agent: agent object
    ordering: list of objectives in the order of priority
    '''
    for obj in ordering[:-1]:
        obj_next = ordering[ordering.index(obj)+1]
        print('\t\tComputing policy for '+ simple_colors.yellow('objective' + ': ' 
                                                            + str(obj) + ' (' 
                                                            + agent.Grid.objective_names[obj] + ')'))
        Reward = agent.Grid.f_R(obj)
        A_pruned = worst_action_prune(agent, obj, obj_next, A_pruned)
        print('Pruned worst action Action Set in oracle mode')
        A_pruned = action_set_value_iteration(agent, Reward, A_pruned)
        
    
    # For the last objective, do value iteration
    Reward = agent.Grid.f_R(ordering[-1])
    print('\t\tComputing policy for '+ simple_colors.yellow('objective' + ': ' 
                                                        + str(ordering[-1]) + ' (' 
                                                        + agent.Grid.objective_names[ordering[-1]] + ')'))
    V, Pi = value_iteration_last_obj(agent, Reward, A_pruned)
    return Pi
    
def contextual_lexicographic_value_iteration_oracle(agent):
    '''
    params:
    agent: agent object
    ordering: list of objectives in the order of priority
    context_map: dictionary mapping each state to a context
    '''
    Grid = agent.Grid
    context_map = Grid.context_map
    Contexts = Grid.Contexts
    for context in Contexts:
        agent.A = copy.deepcopy(agent.A_initial)
        A_initial = copy.deepcopy(agent.A_initial)
        ordering = Grid.f_w(context)
        print('Computing policy for ' + simple_colors.red('context') + ': ' + simple_colors.red(str(context) + ' (' + agent.Grid.context_names[context] + ')'))
        print('\tOrdering: ', ordering)
        Pi = lexicographic_value_iteration_oracle(agent, ordering, A_initial)
        agent.PI[context] = Pi  # Store the policy for each context
    
    agent.A = A_initial
    # synthesize the global policy Pi_G
    Pi_G = {}
    for s in agent.S:
        context = context_map[s]
        Pi_G[s] = agent.PI[context][s]
    agent.Pi_G = Pi_G
    return Pi_G

def worst_action_prune(agent, obj, obj_next, A_pruned):
    '''
    params:
    agent: agent object
    context: context to be considered
    returns:
    context policy for oracle
    '''
    # implementation logic: if the order in [1, 2, 3] corresponding to o1 \succ o2 \succ o3 then 
    # for o_1: prune all actions that are unacceptable for o_2 and then from this set of pruned 
    # actions, remove all actions that are unacceptable for o_3. At any point during this process, 
    # if a state has no action available for execution, reinstate the action with the least impact 
    # for o_2 or o_3, depending on when in the process it is detected. basically compuing independent
    # Q values for each objective independently and sequentially removing the lowest Q value action corresponding
    # to the next objective in the ordering. if only one objective is present, then it is the same as
    # value iteration.
    Q_value_for_obj_next = {}
    Reward = agent.Grid.f_R(obj_next)
    Q_value_for_obj_next = Q_value_iteration(agent, Reward)
    # for idx, obj in enumerate(ordering[:-1]):
    print('Pruning actions for objective ' + str(obj)+ ' based on next objective ' + str(obj_next))
    for s in agent.S:
        # all worst action for obj_next
        Q_min = min(Q_value_for_obj_next[s].values())
        worst_action_list_for_next_obj = [a for a in Q_value_for_obj_next[s] if Q_value_for_obj_next[s][a] == Q_min]
        # remove all actions that are unacceptable for obj_next
        A_pruned[s] = [a for a in A_pruned[s] if a not in worst_action_list_for_next_obj]
        if A_pruned[s] == []:
            A_pruned[s] = worst_action_list_for_next_obj
            # print('Reinstating action for state: ', s)
    return A_pruned
