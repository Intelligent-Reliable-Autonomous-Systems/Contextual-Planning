import numpy as np, os
import value_iteration
from collections import Counter
from pathlib import Path
import warnings

def P_r_given_s_a_c(env, r_fn, s, a, c):
    # Returns P(r|s, a, c)
    obj_order = env.f_w(c)
    reordered_r_fn_from_mdp = [env.f_R(o, c) for o in obj_order]

    r_from_mdp = [reordered_r_fn_from_mdp[0](s, a), reordered_r_fn_from_mdp[1](s, a), reordered_r_fn_from_mdp[2](s, a)]
    r = [r_fn[0](s, a), r_fn[1](s, a), r_fn[2](s, a)]
    if r==r_from_mdp:
        return 1
    else:
        return 0


def P_a_given_s_c(env, all_context_pi, expert_actions, a, s, c, use_expert_traj):
    # Returns P(a | s, c)
    if use_expert_traj==True: # if there's expert traj. for the curent problem instance
        if s in expert_actions.keys():
            if c in expert_actions[s]['possible_contexts']:
                action_taken = expert_actions[s]['action']
                action_counts = Counter(action_taken)
                for item, count in action_counts.items():
                    if item==a:
                        action_prob = action_counts[a] / len(action_taken)
                        return action_prob
                    else: return 0
            else: return 0
        else:
            action_taken = all_context_pi[c][s]
            if action_taken==a:
                action_prob = 1
                return action_prob
            else:
                return 0

    elif use_expert_traj==False: # if not expert traj. for current instance
        action_taken = all_context_pi[c][s]
        if action_taken==a:
            action_prob = 1
            return action_prob
        else:
            return 0

def P_c(env, expert_actions, c, s, a, method):
    # Returns P(c), the prior over context c
    if method=='uniform':
        return 1 / len(env.Contexts)
    elif method=='informed':
        contexts = env.Contexts
        if s in expert_actions.keys():
            if a in expert_actions[s]['action']:
                possible_contexts = expert_actions[s]['possible_contexts']
                context_counts = Counter(possible_contexts)
                if c in possible_contexts:
                    return context_counts[c] / len(possible_contexts)
                else:
                    return 0

        else:
            prob = 1/len(contexts)
            return prob
    return 1/len(contexts)


# Sum over all actions 'a' and contexts 'c'
def P_c_given_s_r(s, env, all_context_pi, expert_actions, possible_actions, all_contexts, prior, use_expert_traj):
    # Numerator for P(c | s, r)
    posterior = np.zeros(len(all_contexts))  # To store P(c | s, r) for each context 'c'

    for idx_c, c in enumerate(all_contexts):
        # Calculate the numerator sum
        obj_order = env.f_w(c)
        reordered_r_fn = [env.f_R(o, c) for o in obj_order]
        numerator_sum = 0
        for a in possible_actions:
            numerator_term = P_r_given_s_a_c(env, reordered_r_fn, s, a, c) * P_a_given_s_c(env, all_context_pi, expert_actions, a, s, c, use_expert_traj) * P_c(env, expert_actions, c, s, a, method=prior)
            numerator_sum += numerator_term
        posterior[idx_c] = numerator_sum

    # Denominator for normalization
    denominator = 0
    for c_prime in all_contexts:
        den_obj_order = env.f_w(c_prime)
        den_reordered_r_fn = [env.f_R(o, c_prime) for o in den_obj_order]
        inner_sum_c_prime = 0
        for a in possible_actions:
            denominator_term = P_r_given_s_a_c(env, den_reordered_r_fn, s, a, c_prime) * P_a_given_s_c(env, all_context_pi, expert_actions, a, s, c_prime, use_expert_traj) * P_c(env, expert_actions, c_prime, s, a, method=prior)
            inner_sum_c_prime += denominator_term
        denominator += inner_sum_c_prime

    # Normalize posterior for each context
    for idx_c in range(len(posterior)):
        posterior[idx_c] /= denominator

    return posterior


def get_expert_actions(data, domain):
    expert_actions = {}
    for idx, row in data.iterrows():
        if domain=='salp':
            col_names = ['x', 'y', 'status', 'coral', 'eddy']
        elif domain=='warehouse':
            col_names = ['x', 'y', 'status', 'slip', 'narrow']
        elif domain=='taxi':
            col_names = ['x', 'y', 'status', 'pothole', 'road']
        x = row[col_names[0]]
        y = row[col_names[1]]
        status = row[col_names[2]]
        feature2 = row[col_names[3]] # could be coral, slip or pothole
        feature3 = row[col_names[4]] # could be eddy, narrow or road
        if domain=='taxi':
            feature3 = 'R' if feature3==True else 'A'
        action = row['action']
        state = (x, y, status, feature2, feature3)
        possible_contexts = row['Possible_Contexts']
        if state not in expert_actions.keys():
            expert_actions[state] = {
                                        'possible_contexts': [],
                                        'action': []
                                    }
        expert_actions[state]['possible_contexts'].extend(possible_contexts)
        expert_actions[state]['action'].append(action)

    return expert_actions



def get_context(env, agent, tr_data, te_data, domain, te_inference_filename, use_expert_traj):
    final_contexts = []
    te_data_copy = te_data.copy()
    int_to_action_mapping = {0: "Noop", 1: "pick", 2: "drop", 3: "U", 4: "D", 5: "L", 6: "R"}
    agent, _ = value_iteration.contextual_lexicographic_value_iteration(agent)
    if use_expert_traj==True:
        expert_actions = get_expert_actions(tr_data, domain)
        prior = 'informed'
    else:
        expert_actions = None
        prior = 'uniform'
    all_context_pi = agent.PI

    for index, row in te_data.iterrows():
        if domain=='salp':
            col_names = ['x', 'y', 'status', 'coral', 'eddy']
        elif domain=='warehouse':
            col_names = ['x', 'y', 'status', 'slip', 'narrow']
        elif domain=='taxi':
            col_names = ['x', 'y', 'status', 'pothole', 'road']
        x = row[col_names[0]]
        y = row[col_names[1]]
        status = row[col_names[2]]
        feature2 = row[col_names[3]] # could be coral, slip or pothole
        feature3 = row[col_names[4]] # could be eddy, narrow or road
        te_state = (x, y, status, feature2, feature3)
        possible_actions = agent.A_initial[te_state]
        if (x, y) == env.goal_location and status=='D':
            possible_actions.append('Noop')
        all_contexts = env.Contexts

        # Compute the posterior for each context given the state
        posterior_context_given_s_r = P_c_given_s_r(te_state, env, all_context_pi, expert_actions, possible_actions, all_contexts, prior, use_expert_traj)
        posterior_context_given_s_r = np.array(posterior_context_given_s_r)
        max_prob = np.max(posterior_context_given_s_r)
        max_prob_contexts = np.where(posterior_context_given_s_r==max_prob)[0].tolist()
        if len(max_prob_contexts)==1:
            final_contexts.append(max_prob_contexts[0])
        else:
            inf_context = None
            for o in env.OMEGA:
                if o in max_prob_contexts:
                    inf_context=o
                    break
            final_contexts.append(inf_context)

    te_data_copy['inferred_context'] = final_contexts

    te_inference_filename = Path(te_inference_filename)
    te_inference_filename.parent.mkdir(parents=True, exist_ok=True)
    te_data_copy.to_csv(te_inference_filename, index=False)
    return te_data_copy