import re, csv, os, sys
import pandas as pd
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import value_iteration
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
import random
from pathlib import Path

def get_observed_data(filename, domain='salp'):
    data = []
    pattern = re.compile("[^0-9-]")

    with open(filename, 'r') as f1:
        reader = csv.reader(f1, delimiter=',')
        for row_idx, row in enumerate(reader):
            if row_idx != 0:
                x_coord, y_coord = int(pattern.sub("", row[0])), int(row[1])
                feature_vals = row[2:]
                split_last_entry = feature_vals[-1].split()
                feature_vals = feature_vals[:-1] + split_last_entry

                if domain=='salp':
                    feature_to_int_mapping = {" 'X'": 0, " 'P'": 1, " 'D'": 2}
                    action_mapping = {"Noop": 0, "pick": 1, "drop": 2, "U": 3, "D": 4, "L": 5, "R": 6}
                    sample_status = feature_to_int_mapping[feature_vals[0]]
                    coral_status = 0 if feature_vals[1]==' False' else 1
                    eddy_status = 0 if feature_vals[2]=='False)' else 1
                    action = action_mapping[feature_vals[3]]
                    r1 = int(feature_vals[4])
                    r2 = int(feature_vals[5])
                    r3 = int(feature_vals[6])
                    gt_context = int(feature_vals[7])
                    # mapped_context = get_context_map_from_clmdp(state, action)
                    data.append([x_coord, y_coord, sample_status, coral_status, eddy_status, action, r1, r2, r3, gt_context])

                    column_names = ['x', 'y', 'status', 'coral', 'eddy', 'action','R1', 'R2', 'R3', 'GT_Context']
                elif domain=='warehouse':
                    feature_to_int_mapping = {" 'X'": 0, " 'P'": 1, " 'D'": 2}
                    action_mapping = {"Noop": 0, "pick": 1, "drop": 2, "U": 3, "D": 4, "L": 5, "R": 6}
                    status = feature_to_int_mapping[feature_vals[0]]
                    slip = 0 if feature_vals[1]==' False' else 1
                    narrow = 0 if feature_vals[2]=='False)' else 1
                    action = action_mapping[feature_vals[3]]
                    r1 = int(feature_vals[4])
                    r2 = int(feature_vals[5])
                    r3 = int(feature_vals[6])
                    gt_context = int(feature_vals[7])
                    # mapped_context = get_context_map_from_clmdp(state, action)
                    data.append([x_coord, y_coord, status, slip, narrow, action, r1, r2, r3, gt_context])
                    column_names = ['x', 'y', 'status', 'slip', 'narrow', 'action','R1', 'R2', 'R3', 'GT_Context']
                elif domain=='taxi':
                    feature_to_int_mapping = {" 'X'": 0, " 'P'": 1, " 'D'": 2}
                    action_mapping = {"Noop": 0, "pick": 1, "drop": 2, "U": 3, "D": 4, "L": 5, "R": 6}
                    status = feature_to_int_mapping[feature_vals[0]]
                    pothole = 0 if feature_vals[1]==' False' else 1
                    road = 0 if feature_vals[2]=="'A')" else 1
                    action = action_mapping[feature_vals[3]]
                    r1 = int(feature_vals[4])
                    r2 = int(feature_vals[5])
                    r3 = int(feature_vals[6])
                    gt_context = int(feature_vals[7])
                    # mapped_context = get_context_map_from_clmdp(state, action)
                    data.append([x_coord, y_coord, status, pothole, road, action, r1, r2, r3, gt_context])
                    column_names = ['x', 'y', 'status', 'pothole', 'road', 'action','R1', 'R2', 'R3', 'GT_Context']
    data_df = pd.DataFrame(data, columns = column_names)
    return data, data_df

def get_data(filename, env, agent, domain, is_train=False):
    test_data = []
    if not is_train:
        state_space = env.S
        for s in state_space:
            gt_state_context = env.state2context_map[s]
            test_data.append([*s, gt_state_context])
        if domain=='salp':
            # <x, y, X/P/D, True/False, True/False>
            column_names = ['x', 'y', 'status', 'coral', 'eddy', 'GT_Context']
        elif domain=='warehouse':
            # <x, y, X/P/D, True/False, True/False>
            column_names = ['x', 'y', 'status', 'slip', 'narrow', 'GT_Context']
        elif domain=='taxi':
            # <x, y, X/P/D, True/False, R/A>
            column_names = ['x', 'y', 'status', 'pothole', 'road', 'GT_Context']
        test_data_df = pd.DataFrame(test_data, columns = column_names)
        return test_data_df

    else:
        _, observed_data = get_observed_data(filename, domain)
        contexts = env.Contexts
        agent, _ = value_iteration.contextual_lexicographic_value_iteration(agent)
        all_ordering_pi = {}

        for c in contexts:
            obj_ordering = env.f_w(c)
            all_ordering_pi[(c, tuple(obj_ordering))] = agent.PI[c]

        data_with_mapped_context_df = update_tr_data(observed_data, all_ordering_pi, env, agent, domain)
        return data_with_mapped_context_df

def get_pi_obj_order_r(env_agent, obj_order, reward_order):
    _, pi_for_r = value_iteration.lexicographic_value_iteration(env_agent, obj_order, R=reward_order)
    return pi_for_r



def update_tr_data(data, all_ordering_pi, env, agent, domain):
    possible_contexts = []
    int_to_action_mapping = {0: "Noop", 1: "pick", 2: "drop", 3: "U", 4: "D", 5: "L", 6: "R"}
    contexts = {}
    if domain=='salp':
        for index, row in data.iterrows():
            x = row['x']
            y = row['y']
            status = 'X' if row['status'] == 0 else 'P' if row['status'] == 1 else 'D'
            coral = bool(row['coral'])
            eddy = bool(row['eddy'])
            action = int_to_action_mapping[int(row['action'])]
            R1 = row['R1']
            R2 = row['R2']
            R3 = row['R3']
            GT_Context = int(row['GT_Context'])
            state = (x, y, status, coral, eddy)

            for k, v in all_ordering_pi.items():  # key: (context, tuple(obj_order)), values: {(state): action}
                context = k[0]
                pi = v # pi => {(state): action}
                for pi_state, pi_action in pi.items():
                    if state==pi_state and action==pi_action:
                        if (state, action) not in contexts:
                            contexts[(state, action)] = []
                        contexts[(state, action)].append(context)
                        break

            possible_contexts.append([
                x, y, status, coral, eddy, action, R1, R2, R3, GT_Context, contexts.get((state, action), [])])

        column_names = ['x', 'y', 'status', 'coral', 'eddy', 'action', 'R1', 'R2', 'R3', 'GT_Context', 'Possible_Contexts']
    elif domain=='warehouse':
        for index, row in data.iterrows():
            x = row['x']
            y = row['y']
            status = 'X' if row['status'] == 0 else 'P' if row['status'] == 1 else 'D'
            slip = bool(row['slip'])
            narrow = bool(row['narrow'])
            action = int_to_action_mapping[int(row['action'])]
            R1 = row['R1']
            R2 = row['R2']
            R3 = row['R3']
            GT_Context = int(row['GT_Context'])
            state = (x, y, status, slip, narrow)

            for k, v in all_ordering_pi.items():  # key: (context, tuple(obj_order)), values: {(state): action}
                context = k[0]
                pi = v # pi => {(state): action}
                for pi_state, pi_action in pi.items():
                    if state==pi_state and action==pi_action:
                        if (state, action) not in contexts:
                            contexts[(state, action)] = []
                        contexts[(state, action)].append(context)
                        break

            possible_contexts.append([
                x, y, status, slip, narrow, action, R1, R2, R3, GT_Context, contexts.get((state, action), [])])
        column_names = ['x', 'y', 'status', 'slip', 'narrow', 'action', 'R1', 'R2', 'R3', 'GT_Context', 'Possible_Contexts']
    elif domain=='taxi':
        for index, row in data.iterrows():
            x = row['x']
            y = row['y']
            status = 'X' if row['status'] == 0 else 'P' if row['status'] == 1 else 'D'
            pothole = bool(row['pothole'])
            road = bool(row['road'])
            action = int_to_action_mapping[int(row['action'])]
            R1 = row['R1']
            R2 = row['R2']
            R3 = row['R3']
            GT_Context = int(row['GT_Context'])
            state_to_write = (x, y, status, pothole, road)

            for k, v in all_ordering_pi.items():  # key: (context, tuple(obj_order)), values: {(state): action}
                context = k[0]
                pi = v # pi => {(state): action}
                for pi_state, pi_action in pi.items():
                    if pi_state==(x, y, status, pothole, 'R' if road==True else 'A') and action==pi_action:
                        if (state_to_write, action) not in contexts:
                            contexts[(state_to_write, action)] = []
                        contexts[(state_to_write, action)].append(context)
                        break

            possible_contexts.append([
                x, y, status, pothole, road, action, R1, R2, R3, GT_Context, contexts.get((state_to_write, action), [])])
        column_names = ['x', 'y', 'status', 'pothole', 'road', 'action', 'R1', 'R2', 'R3', 'GT_Context', 'Possible_Contexts']
    updated_data_df = pd.DataFrame(possible_contexts, columns=column_names)

    return updated_data_df


def evaluate_inferred_contexts(inf_data, instance_idx, perf_results_file):
    y_true = inf_data['GT_Context']
    y_pred = inf_data['inferred_context']

    f1 = f1_score(y_true, y_pred, average=None)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    perf_results_file = Path(perf_results_file)
    perf_results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(perf_results_file, 'a') as f:
        if os.stat(perf_results_file).st_size == 0:
            f.write("Env,Accuracy,F1,MCC\n")
        f.write('{},{},{},{}\n'.format(instance_idx, accuracy, f1, mcc))
        f.close()

def get_random_initial_states(env, num_samples=5):
    sample_space = [s for s in env.S if (s[2]=='X' and s[0] in range(0,5) and s[1] in range(0,5))]
    if num_samples > len(sample_space):
        raise ValueError("num_samples cannot be greater than the total number of states")
    sampled_states = random.sample(sample_space, num_samples)

    return sampled_states

def get_inferred_context_for_s(env_id, domain, start_state):

    inf_context_filename = 'context_inference/c6_output/'+start_state+'_start_illustration_all/'+domain+'/bayes/illustration'+str(env_id)+'inference_results.csv'
    # inf_context_filename = 'context_inference/output_without_exp_data/'+domain+'/bayes/illustration'+str(env_id)+'inference_results.csv'
    print('domain: ', domain)
    print('env id: ', env_id)
    print('start: ', start_state)
    print('context map filename: ', inf_context_filename)
    if domain=='salp':
        data = pd.read_csv(inf_context_filename)
        context_map = {}
        for idx, row in data.iterrows():
            x = row['x']
            y = row['y']
            status = row['status']
            coral = row['coral']
            eddy = row['eddy']
            inf_context = row['inferred_context']
            state = (x, y, status, coral, eddy)
            context_map[state] = inf_context
        return context_map

    elif domain=='warehouse':
        data = pd.read_csv(inf_context_filename)
        context_map = {}
        for idx, row in data.iterrows():
            x = row['x']
            y = row['y']
            status = row['status']
            slip = row['slip']
            narrow = row['narrow']
            inf_context = row['inferred_context']
            state = (x, y, status, slip, narrow)
            context_map[state] = inf_context
        return context_map

    elif domain=='taxi':
        data = pd.read_csv(inf_context_filename)
        context_map = {}
        for idx, row in data.iterrows():
            x = row['x']
            y = row['y']
            status = row['status']
            pothole = row['pothole']
            road = row['road']
            inf_context = row['inferred_context']
            state = (x, y, status, pothole, road)
            context_map[state] = inf_context
        return context_map