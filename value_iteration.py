import copy
import math
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
    return agent, Pi

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
        A_pruned[s] = [a for a in A[s] if round(Q[s][a]) >= round(V[s])-1]
    agent.A = copy.deepcopy(A_pruned)
    return agent

def lexicographic_value_iteration(agent, ordering, context= None):
    '''
    params:
    agent: agent object
    ordering: list of objectives in the order of priority
    '''
    for obj in ordering[:-1]:
        print('\t\tComputing policy for '+ simple_colors.yellow('objective' + ': ' + str(obj) + ' (' + agent.Grid.objective_names[obj] + ')'))
        if context is not None:
            Reward = agent.Grid.f_R(obj, context)
        else:
            Reward = agent.Grid.R_obs[obj]
        agent = action_set_value_iteration(agent, Reward)
    
    # For the last objective, do value iteration
    if context is not None:
        Reward = agent.Grid.f_R(ordering[-1], context)
    else:
        Reward = agent.Grid.R_obs[ordering[-1]]
    print('\t\tComputing policy for '+ simple_colors.yellow('objective' + ': ' + str(ordering[-1]) + ' (' + agent.Grid.objective_names[ordering[-1]] + ')'))
    V, Pi = value_iteration(agent, Reward)
    return agent, Pi
    
def scalarized_value_iteration(agent, ordering):
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
                V[s] = sum([agent.scalarization_weights[i] * agent.Grid.R_obs[ordering[i]](s, 'Noop') for i in range(len(ordering))])
                Pi[s] = 'Noop'
                residual[s] = abs(V[s] - V_prev[s])
                continue
            for a in A[s]:
                T = agent.get_transition_prob(s, a)
                scalarized_reward = sum([agent.scalarization_weights[i] * agent.Grid.R_obs[ordering[i]](s, a) for i in range(len(ordering))])
                Q[s][a] = scalarized_reward + gamma * sum([T[s_prime] * V[s_prime] for s_prime in list(T.keys())])
            V[s] = max(Q[s].values())
            Pi[s] = max(Q[s], key=Q[s].get)
            residual[s] = abs(V[s] - V_prev[s])
        if max(residual.values()) < 1e-6 or iter > 1000:
            # print(simple_colors.blue('Value Iteration converged in {} iterations.\n'.format(iter)))
            break
        iter += 1
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
        agent, Pi = lexicographic_value_iteration(agent, ordering, context)
        agent.PI[context] = Pi  # Store the policy for each context
    
    agent.A = copy.deepcopy(agent.A_initial)
    # synthesize the global policy Pi_G
    Pi_G = {}
    for s in agent.S:
        context = context_map[s]
        Pi_G[s] = agent.PI[context][s]
    agent.Pi_G = Pi_G
    return agent, Pi_G

    
def contextual_scalarized_value_iteration(agent):
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
        print('Computing ' + simple_colors.yellow('scalarized') + ' policy for ' + simple_colors.red('context') + ': ' + simple_colors.red(str(context) + ' (' + agent.Grid.context_names[context] + ')'))
        print('\tOrdering: ', ordering)
        agent, Pi = scalarized_value_iteration(agent, ordering)
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
    S = agent.S
    A = copy.deepcopy(Pi_G)
    V = {s: 0 for s in S}
    V[agent.s_goal] = Reward[agent.s_goal]
    gamma = 0.99
    S_unsolved = [s for s in S if s != agent.s_goal]
    S_solved = [agent.s_goal]
    Delta = {s: 0 for s in S}
    delta = 0
    epsilon = 1e-6
    iter = 0
    while True:
        V_prev = copy.deepcopy(V)
        S_check = []
        delta = 0
        for s in S_unsolved:
            V[s] = Reward[s]
            T = agent.get_transition_prob(s, A[s])
            s_succesors = list(T.keys())
            for s_prime in s_succesors:
                V[s] += gamma * T[s_prime] * V[s_prime]
                if s_prime in S_solved and s not in S_check:
                    S_check.append(s)
            delta = max(delta, abs(V[s] - V_prev[s]))
            Delta[s] = abs(V[s] - V_prev[s])
        for s in S_check:
            if Delta[s] < epsilon:
                S_solved.append(s)
                S_unsolved.remove(s)
        if delta < epsilon or iter > 1000:
            print(simple_colors.blue('L-RTDP converged in {} iterations.\n'.format(iter)))
            break
        iter += 1
    return V
            
