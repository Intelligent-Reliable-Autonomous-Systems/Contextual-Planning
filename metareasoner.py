import copy
import simple_colors
from value_iteration import labeled_RTDP, lexicographic_value_iteration

def get_context_map(S, Contexts, domain_name):
    '''Returns a dictionary mapping each state in S to a context in Contexts.'''
    context_map = {}
    if domain_name == 'salp':
        for s in S:
            # context_map[s] = None  # ICP by Yash
            # s = (s[0]: x, s[1]: y, s[2]: sample_with_agent, s[3]: coral_flag, s[4]: done_flag)
            if s[2] != 'X' and s[3] is True:
                context_map[s] = Contexts[1] # Context 1 - NSE mitigation is priority
            else:
                context_map[s] = Contexts[0] # Context 2 - Task is priority
    return context_map

def conflict_checker(Pi, agent):
    '''Returns a boolean indicating whether there is a conflict in Pi between contextual objectives.'''
    Reachability_Reward = {s: 0 for s in agent.S}
    Reachability_Reward[agent.s_goal] = 1.0
    V = labeled_RTDP(agent, Pi, Reachability_Reward)
    # print('Value Function: ', V)
    if any([V[s] == 0 for s in agent.S]):
        return True
    return False

def conflict_resolver(Pi, agent):
    '''Returns a dictionary mapping each objective to a priority level.'''
    print(simple_colors.cyan('Invoking Conflict Resolver', ['bold']))
    S = agent.S
    A = copy.deepcopy(agent.A_initial)
    PI = copy.deepcopy(agent.PI)  # dictionary mapping each context to it's policy over entire state space
    Grid = agent.Grid
    state2context_map = Grid.state2context_map  # dictionary mapping each state to a context
    context2state_map = Grid.context2state_map  # dictionary mapping each context to a list of states
    
    Contexts = Grid.Contexts  # list of contexts
    OMEGA = Grid.OMEGA  # meta-ordering over "contexts" as a list in order of decreasing priority: [2,3,1] = 2 > 3 > 1 
    
    # now we generate sub-lists of priority orderings over contexts in order of resolving: [2,3,1] -> [[1],[3,1],[2,3,1]]
    context_update_orderings = [OMEGA[i:] for i in range(len(OMEGA)-1,-1,-1)]
    
    for context_update_ordering in context_update_orderings:
        # print('Context Update Ordering: ', context_update_ordering)
        contexts_with_fixed_actions = [c for c in Contexts if c not in context_update_ordering] 
        # print('Contexts with Fixed Actions: ', contexts_with_fixed_actions)
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
            agent.A = copy.deepcopy(agent.A_initial)
            continue
        else:
            print("conflict has been resolved anfter a lexicographic update of contexts: ", context_update_ordering)
            agent.A = copy.deepcopy(agent.A_initial)
            return Pi



# TODO:
# 1. Value Iterations (done)
# 2. main.py
# 3. layouts folder (done)
# 4. Conflict Checker --> works with VI, now do L-RTDP
# 5. Conflict Resolver --> Implemented
# 6. Test with manual policies with conflicts
# 7. A heatmap metric to show a conflict visually

