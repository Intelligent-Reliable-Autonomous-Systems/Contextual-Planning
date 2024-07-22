import copy
import simple_colors

def value_iteration(agent, Reward):
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

def action_set_value_iteration(agent, Reward):
    '''
    returns the agent with a pruned action set based on all optimal policy space of for provided Reward function R(s,a).
    '''
    S = agent.S
    A = agent.A
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
        A_pruned[s] = [a for a in A[s] if (Q[s][a]) >= (V[s] - 2)]
        # print('State: ', s)
        # print('Q[s]: ', Q[s])
        # print('V[s]: ', V[s])
        # print('Pruned action set: ', A_pruned[s])
        # print('\n\n')
    agent.A = copy.deepcopy(A_pruned)
    return agent

def lexicographic_value_iteration(agent, ordering):
    '''
    params:
    agent: agent object
    ordering: list of objectives in the order of priority
    '''
    for obj in ordering[:-1]:
        print('\t\tComputing policy for '+ simple_colors.yellow('objective' + ': ' 
                                                            + str(obj) + ' (' 
                                                            + agent.Grid.objective_names[obj] + ')'))
        Reward = agent.Grid.f_R(obj)
        agent = action_set_value_iteration(agent, Reward)
    
    # For the last objective, do value iteration
    Reward = agent.Grid.f_R(ordering[-1])
    print('\t\tComputing policy for '+ simple_colors.yellow('objective' + ': ' 
                                                        + str(ordering[-1]) + ' (' 
                                                        + agent.Grid.objective_names[ordering[-1]] + ')'))
    V, Pi = value_iteration(agent, Reward)
    return agent, Pi
    
def contextual_lexicographic_value_iteration(agent):
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
        ordering = Grid.f_w(context)
        print('Computing policy for ' + simple_colors.red('context') + ': ' + simple_colors.red(str(context) + ' (' + agent.Grid.context_names[context] + ')'))
        print('\tOrdering: ', ordering)
        agent, Pi = lexicographic_value_iteration(agent, ordering)
        agent.PI[context] = Pi  # Store the policy for each context
    
    agent.A = copy.deepcopy(agent.A_initial)
    # synthesize the global policy Pi_G
    Pi_G = {}
    for s in agent.S:
        context = context_map[s]
        Pi_G[s] = agent.PI[context][s]
    agent.Pi_G = Pi_G
    return agent, Pi_G

def labeled_RTDP(agent, Pi_G, Reward):
    '''
    params:
    agent: agent object
    policy: policy Pi_G
    Reward: reward function as dict R[s]
    
    Perform labeled Real Time Dynamic Programming to return value function
    '''
    # is_solved = {s: False for s in agent.S}
    # V = {s: 0 for s in agent.S}
    # gamma = 1
    # epsilon = 1e-6
    # delta = 0
    # s0 = agent.s0
    # while not is_solved[s0]:
    #     s = s0
    #     while not is_solved[s]:
    #         V_prev = copy.deepcopy(V)
    #         a = Pi_G[s]
    #         T = agent.get_transition_prob(s, a)
    #         print("Reward[s]: ", Reward[s])
    #         print("The other terms: ",gamma * sum([T[s_prime] * V_prev[s_prime] for s_prime in list(T.keys())]))
    #         V[s] = Reward[s] + gamma * sum([T[s_prime] * V[s_prime] for s_prime in list(T.keys())])
    #         print("delta: ", delta)
    #         print("abs(V[s] - V_prev[s]): ", abs(V[s] - V_prev[s]))
    #         delta = abs(V[s] - V_prev[s])
    #         is_solved[s] = delta < epsilon
    #         # sample a state based on the transition probabilities
    #         print(str(s)+", "+a+" --> "+str(agent.sample_state(s, a)))
    #         if s == agent.s_goal:
    #             break
    #         s = agent.sample_state(s, a)
    # return V

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
            # print('s: ', s)
            if s == agent.s_goal:
                V[s] = Reward[s]
                Q[s]['Noop'] = V[s]
                Pi[s] = 'Noop'
                residual[s] = abs(V[s] - V_prev[s])
                continue
            a = Pi_G[s]
            T = agent.get_transition_prob2(s, a)
            # print('T: ', T)
            Q[s][a] = Reward[s] + gamma * sum([T[s_prime] * V[s_prime] for s_prime in list(T.keys())])
            # print('Q[s]: ', Q[s])
            V[s] = max(Q[s].values())
            # print('V[s]: ', V[s])
            # print('V_prev[s]: ', V_prev[s])
            Pi[s] = max(Q[s], key=Q[s].get)
            residual[s] = abs(V[s] - V_prev[s])
            # print('residual[s]: ', residual[s])
            # print('\n\n')
        if max(residual.values()) < 1e-6 or iter > 1000:
            # print(simple_colors.blue('Value Iteration converged in {} iterations.\n'.format(iter)))
            break
        iter += 1
    return V
            