import copy
import simple_colors
from infer_context import get_context_from_inference_for_grid
from value_iteration import labeled_RTDP, lexicographic_value_iteration

def get_context_map(S, infer_context_flag, domain_name, filename='none'):
    '''Returns a dictionary mapping each state in S to a context in Contexts.'''
    context_map = {}
    if infer_context_flag:
        context_map = get_context_from_inference_for_grid(domain_name, filename)
        return context_map
    if domain_name == 'salp':
        for s in S:
            # s = (s[0]: x, s[1]: y, s[2]: sample_with_agent, s[3]: coral_flag, s[4]: eddy, s[5]: done_flag)
            if s[2] == 'P' and s[3] is True:
                context_map[s] = 1  # Context 2 - Prioritize coral NSE escape
            elif s[4] is True:
                context_map[s] = 2  # Context 3 - Prioritize Eddy escape
            else:
                context_map[s] = 0  # Context 1 - Prioritize Task
    elif domain_name == 'warehouse':
        for s in S:
            # s = (s[0]: x, s[1]: y, s[2]: package_status, s[3]: slippery_tile, s[4]: narrow_corridor)
            if s[2] == 'P' and s[3] is True:
                context_map[s] = 1  # Context 2 - Prioritize slippery tile NSE escape
            elif s[4] is True:
                context_map[s] = 2  # Context 3 - Prioritize moving out of narrow corridor
            else:
                context_map[s] = 0  # Context 1 - Prioritize package delivery Task
    elif domain_name == 'taxi':
        for s in S:
            # s: (s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type)
            if s[4] == 'A':
                context_map[s] = 1  # Context 2 - Prioritize autonomy-enabled road
            elif s[2] == 'P' and s[3] is True:
                context_map[s] = 2  # Context 3 - Prioritize pothole NSE escape when passenger is present
            else:
                context_map[s] = 0  # Context 1 - Prioritize passenger pickup/dropoff Task
    return context_map

def conflict_checker(Pi, agent):
    '''Returns a boolean indicating whether there is a conflict in Pi between contextual objectives.'''
    Reachability_Reward = {s: 0 for s in agent.S}
    Reachability_Reward[agent.s_goal] = 1.0
    V = labeled_RTDP(agent, Pi, Reachability_Reward)
    if any([V[s] == 0.0 for s in agent.S]):
        print(simple_colors.red('Conflict States: ', ['bold']))
        for s in agent.S:
            if V[s] == 0.0:
                print(simple_colors.red('\t'+str(s), ['bold'])) 
        return True
    return False

def conflict_checker_new(Pi, agent):
    '''Returns a boolean indicating whether there is a conflict in Pi between contextual objectives.'''
    Reachability_Reward = {s: 0 for s in agent.S}
    Reachability_Reward[agent.s_goal] = 1.0
    V_r = labeled_RTDP(agent, Pi, Reachability_Reward)
    if any([V_r[s] == 0.0 for s in agent.S]):
        print(simple_colors.red('Conflict States: ', ['bold']))
        for s in agent.S:
            if V_r[s] == 0.0:
                print(simple_colors.red('\t'+str(s), ['bold'])) 
        return True, V_r
    return False, V_r

def conflict_resolver(Pi, agent):
    '''Returns a dictionary mapping each objective to a priority level.'''
    print(simple_colors.cyan('Invoking Conflict Resolver', ['bold']))
    A = copy.deepcopy(agent.A_initial)
    PI = copy.deepcopy(agent.PI)  # dictionary mapping each context to it's policy over entire state space
    Grid = agent.Grid
    state2context_map = Grid.state2context_map  # dictionary mapping each state to a context
    context2state_map = Grid.context2state_map  # dictionary mapping each context to a list of states
    
    Contexts = Grid.Contexts  # list of contexts
    OMEGA = Grid.OMEGA  # meta-ordering over "contexts" as a list in order of decreasing priority: [2,3,1] = 2 > 3 > 1 
    
    # now we generate sub-lists of priority orderings over contexts in order of resolving: [2,3,1] -> [[1],[3,1],[2,3,1]]
    context_update_orderings = [OMEGA[i:] for i in range(len(OMEGA)-1,-1,-1)]
    print('Omega: ', OMEGA)
    print('Context Update Orderings: ', context_update_orderings)
    for context_update_ordering in context_update_orderings:
        print('Context Update Ordering: ', context_update_ordering)
        contexts_with_fixed_actions = [c for c in Contexts if c not in context_update_ordering] 
        print('Contexts with Fixed Actions: ', contexts_with_fixed_actions)
        # fix the actions for the contexts that are not being updated
        for context in contexts_with_fixed_actions:
            for s in context2state_map[context]:
                a = Pi[s]
                A[s] = [a]
        agent.A = A
        # now we update the actions for the contexts that are being updated
        for context in context_update_ordering:
            _, PI[context] =  lexicographic_value_iteration(agent, agent.Grid.f_w(context))
            for s in context2state_map[context]:
                a = PI[context][s]
                A[s] = [a]
                Pi[s] = a
            agent.A = A
        # check if Pi still has conflict
        if conflict_checker(Pi, agent):
            print(simple_colors.red('Conflict not resolved yet!', ['bold']))
            agent.A = copy.deepcopy(agent.A_initial)
            continue
        else:
            print("conflict has been resolved after a lexicographic update of contexts: ", context_update_ordering)
            agent.A = copy.deepcopy(agent.A_initial)
            return Pi
    print(simple_colors.red('Conflict could not be resolved!', ['bold']))

def conflict_resolver_new(Pi, agent):
    '''Returns a dictionary mapping each objective to a priority level.'''
    conflict, V_r = conflict_checker_new(Pi, agent)
    if not conflict:
        print(simple_colors.cyan('No conflicts to be resolved', ['bold']))
        return Pi
    print(simple_colors.cyan('Invoking (new) conflict resolver', ['bold']))
    A = copy.deepcopy(agent.A_initial)
    PI = copy.deepcopy(agent.PI)  # dictionary mapping each context to it's policy over entire state space
    Grid = agent.Grid
    state2context_map = Grid.state2context_map  # dictionary mapping each state to a context
    context2state_map = Grid.context2state_map  # dictionary mapping each context to a list of states
    
    Contexts = Grid.Contexts  # list of contexts
    OMEGA = Grid.OMEGA  # meta-ordering over "contexts" as a list in order of decreasing priority: [2,3,1] = 2 > 3 > 1 
    
    # now we generate sub-lists of priority orderings over contexts in order of resolving: [2,3,1] -> [[1],[3,1],[2,3,1]]
    print('Omega: ', OMEGA)
    # print('Context update ordering for problem states: ', context_update_orderings)
    S_unreachable = [s for s in agent.S if V_r[s] == 0.0]  # s_u : states where conflict exists that cannot reach goal or any s_r states
    S2C_u = {s: state2context_map[s] for s in S_unreachable}
    C_u = list(set(list(S2C_u.values())))
    C_u_ordered = [c_u for c_u in reversed(OMEGA) if c_u in C_u]  # ordered lowest to highest priority
    context_update_orderings_all = [OMEGA[i:] for i in range(len(OMEGA)-1,-1,-1)]
    context_update_orderings = [context_update_orderings_all[i] for i in range(len(context_update_orderings_all)) if C_u_ordered[0] in context_update_orderings_all[i]]
    print('Context Update Orderings: ', context_update_orderings)
    for context_update_ordering in context_update_orderings:
        print('Context Update Ordering: ', context_update_ordering)
        contexts_with_fixed_actions = [c for c in Contexts if c not in context_update_ordering] 
        print('Contexts with Fixed Actions: ', contexts_with_fixed_actions)
        # fix the actions for the contexts that are not being updated
        for context in contexts_with_fixed_actions:
            for s in context2state_map[context]:
                a = Pi[s]
                A[s] = [a]
        agent.A = A
        # now we update the actions for the contexts that are being updated
        for context in context_update_ordering:
            _, PI[context] =  lexicographic_value_iteration(agent, agent.Grid.f_w(context))
            for s in context2state_map[context]:
                a = PI[context][s]
                A[s] = [a]
                Pi[s] = a
            agent.A = A
        # check if Pi still has conflict
        if conflict_checker(Pi, agent):
            print(simple_colors.red('Conflict not resolved yet!', ['bold']))
            agent.A = copy.deepcopy(agent.A_initial)
            continue
        else:
            print("conflict has been resolved after a lexicographic update of contexts: ", context_update_ordering)
            agent.A = copy.deepcopy(agent.A_initial)
            return Pi
    print(simple_colors.red('Conflict could not be resolved!', ['bold']))