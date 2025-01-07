import copy
import random
import numpy as np
import read_grid
import value_iteration
import metareasoner as MR
import tensorflow as tf
from keras import layers, models


class TaxiEnvironment:
    def __init__(self, filename, context_sim):
        self.OMEGA = [1, 2, 0]  # meta ordering over contexts c1 > c2 > c0
        # currently setup as ordering for context i = context_ordering[i]   
        context_ordering = {0: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # 'Task only'
                            1: [[0, 2, 1], [0, 2, 1], [0, 2, 1]],  # 'LMDP using Omega'
                            2: [[0, 2, 1], [0, 2, 1], [0, 2, 1]],  # 'Scalarization using Omega'
                            3: [[0, 2, 1], [1, 0, 2], [2, 0, 1]],  # 'LMDP for Contexts'
                            4: [[0, 2, 1], [1, 0, 2], [2, 0, 1]],  # 'Yang et al. (2019)'
                            5: [[0, 2, 1], [1, 0, 2], [2, 0, 1]],  # 'Contextual Approach w/o resolver'
                            6: [[0, 2, 1], [1, 0, 2], [2, 0, 1]],  # 'Contextual Approach w/ resolver (Our Approach 1)'
                            7: [[0, 2, 1], [1, 0, 2], [2, 0, 1]],} # 'Contextual Approach w/ resolver & learned Z (Our Approach 2)'
        
        
        # Read the grid from the file and initialize the environment
        All_States, rows, columns = read_grid.grid_read_from_file(filename)
        self.All_States = All_States
        self.rows = rows
        self.columns = columns
        goal_loc = np.where(All_States == 'G')
        self.goal_location = (goal_loc[0][0], goal_loc[1][0])
        # s: <s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type>
        self.s_goal = (self.goal_location[0], self.goal_location[1], 'D', False, 'R')
        
        # Initialize parameters for the CLMDP
        self.S = self.get_state_space()
        self.objectives = [i for i in context_ordering[context_sim][0]]
        self.R_obs = [self.R1_out_context, self.R2_out_context, self.R3_out_context]
        # removing duplicates in self.objectives
        self.objectives = list(set(self.objectives))
        self.objective_names = {0: 'Passenger Drop-off', 1: 'Use Autonomy-Enabled Tracks', 2: 'Avoid Pothole Discomfort'}
        self.Contexts = list(range(len(context_ordering[context_sim])))
        self.contextual_orderings = context_ordering[context_sim]  # list of contextual orderings each a list of objectives
        self.Reward_for_obj = self.get_reward_functions()  # list of reward functions for each objective
        self.context_names = self.get_context_name(self.Contexts)
        self.context_map =  MR.get_context_map(self.S, self.Contexts, 'taxi')
        self.state2context_map = self.context_map
        self.context2state_map = {}
        # print("Context: ", self.Contexts)
        for context in self.Contexts:
            self.context2state_map[context] = []
        for s in self.S:
            self.context2state_map[self.state2context_map[s]].append(s)

    def get_state_space(self):
        S = []
        # s: <s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type>
        for i in range(self.rows):
            for j in range(self.columns):
                for passenger_status in ['X','P','D']:
                    if self.All_States[i][j] in ['P', 'R', 'G', 'B']:
                        S.append((i, j, passenger_status, self.All_States[i][j]=='P', 'R'))
                    elif self.All_States[i][j] == 'A':
                        S.append((i, j, passenger_status, False, 'A'))  # No pothole on autonomy-enabled track
        return S

    def get_reward_functions(self):
        R = [[self.R1_in_context, self.R2_out_context, self.R3_out_context],
             [self.R1_out_context, self.R2_in_context, self.R3_out_context],
             [self.R1_out_context, self.R2_out_context, self.R3_in_context]]
        return R

    def R1_in_context(self, s, a):
        # Passengar Drop-off reward
        # s: <s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type>
        s_next = self.step(s, a)
        if s_next == self.s_goal and a != 'Noop':
            return 100
        elif s_next == self.s_goal and a == 'Noop':
            return 0
        else:
            return -5
    
    def R1_out_context(self, s, a):
        # Passengar Drop-off reward
        # s: <s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type>
        s_next = self.step(s, a)
        if s_next == self.s_goal and a != 'Noop':
            return 100
        elif s_next == self.s_goal and a == 'Noop':
            return 0
        else:
            return -1
        
    def R2_in_context(self, s, a):
        # Penalty for not being on autonomy-enabled track s[4] \in {'A', 'R'}
        # operation actions = ['Noop', 'pick', 'drop', 'U', 'D', 'L', 'R']
        # s: <s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type>
        s_next = self.step(s, a)
        if s_next[4] == 'A':  # if the agent is on the normal road
            R = -10
        else:
            R = 0
        return R
    
    def R2_out_context(self, s, a):
        # Penalty for not being on autonomy-enabled track s[4] \in {'A', 'R'}
        # operation actions = ['Noop', 'pick', 'drop', 'U', 'D', 'L', 'R']
        # s: <s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type>
        s_next = self.step(s, a)
        if s_next[4] == 'A':  # if the agent is on the normal road
            R = -5
        else:
            R = 0
        return R
    
    def R3_in_context(self, s, a):
        # Pothole discomfort penalty
        # s: <s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type>
        s_next = self.step(s, a)        
        if s[2] == 'P' and s_next[3] is True:
            R = -10
        else:
            R = 0
        return R
    
    def R3_out_context(self, s, a):
        # Pothole discomfort penalty
        # s: <s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type>
        s_next = self.step(s, a)        
        if s[2] == 'P' and s_next[3] is True:
            R = -5
        else:
            R = 0
        return R
    
    def f_R(self, obj, context):
        ''''Return the reward function for the given objective'''
        if obj not in self.objectives:
            print("Invalid objective")
            return None
        return self.Reward_for_obj[context][obj]
    
    def f_w(self, context):
        '''Return the context ordering for the given context'''
        if context not in self.Contexts:
            print("Invalid context")
            return None
        return self.contextual_orderings[context]
    
    def step(self, s, a):
        # s: <s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type>
        # operation actions = ['Noop','pick', 'drop', 'U', 'D', 'L', 'R']
        s = list(s)
        if a == 'pick':
            if self.All_States[s[0]][s[1]] == 'B':
                s[2] = 'P'
        elif a == 'drop':
            if s[0] == self.goal_location[0] and s[1] == self.goal_location[1]:
                s[2] = 'D'
            else:
                s[2] = s[2]
        elif a == 'U' or a == 'D' or a == 'L' or a == 'R':
            s = self.move_correctly(s, a)
        elif a == 'Noop':
            s = s
        else:
            print("INVALID ACTION: ", a)
        s = tuple(s)
        return s
    
    def move_correctly(self, s, a):
        # s: <s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type>
        # action = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
        s_next = 0
        if a == 'U':
            if s[0] == 0:
                s_next = s
            else:
                if self.All_States[s[0] - 1][s[1]] in ['P', 'R', 'G', 'B']:
                    s_next = (s[0] - 1, s[1], s[2], self.All_States[s[0] - 1][s[1]] == 'P', 'R')
                else:
                    s_next = (s[0] - 1, s[1], s[2], False, 'A')
        elif a == 'D':
            if s[0] == self.rows - 1:
                s_next = s
            else:
                if self.All_States[s[0] + 1][s[1]] in ['P', 'R', 'G', 'B']:
                    s_next = (s[0] + 1, s[1], s[2], self.All_States[s[0] + 1][s[1]] == 'P', 'R')
                else:
                    s_next = (s[0] + 1, s[1], s[2], False, 'A')
        elif a == 'L':
            if s[1] == 0:
                s_next = s
            else:
                if self.All_States[s[0]][s[1] - 1] in ['P', 'R', 'G', 'B']:
                    s_next = (s[0], s[1] - 1, s[2], self.All_States[s[0]][s[1] - 1] == 'P', 'R')
                else:
                    s_next = (s[0], s[1] - 1, s[2], False, 'A')
        elif a == 'R':
            if s[1] == self.columns - 1:
                s_next = s
            else:
                if self.All_States[s[0]][s[1] + 1] in ['P', 'R', 'G', 'B']:
                    s_next = (s[0], s[1] + 1, s[2], self.All_States[s[0]][s[1] + 1] == 'P', 'R')
                else:
                    s_next = (s[0], s[1] + 1, s[2], False, 'A')
        return s_next
    
    def check_context(self, s):
        '''Return the context of the state'''
        return self.context_names[self.state2context_map[s]]
    
    def get_context_name(self, context):
        '''Return the name of the contexts'''
        context_names = {}
        for context in self.Contexts:
            context_name = ""
            for obj in self.contextual_orderings[context]:
                context_name += self.objective_names[obj] + " > "
            context_name = context_name[:-3]
            context_names[context] = context_name
        return context_names
    
class TaxiAgent:
    def __init__(self, Grid, start_location=(0,0), label='1'):
        
        # Initialize the agent with environment, passenger, start location, and label (to identify incase of multi-agent scenario)
        self.Grid = Grid
        self.start_location = start_location
        self.label = label
        self.IDX = int(label) - 1
        self.goal_loc = Grid.goal_location
        
        # Set the success probability of the agent
        self.p_success = 0.8
        
        # set the start state and the goal state in the right format:
        # s: <s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type>
        self.s0 = (start_location[0], start_location[1], 'X', False, 'R')
        self.s = copy.deepcopy(self.s0)
        self.s_goal = (self.goal_loc[0], self.goal_loc[1], 'D', False, 'R')
        
        # Initialize state and action space
        self.S = Grid.S
        self.A = self.get_action_space()
        self.A_initial = copy.deepcopy(self.A)
        self.R = 0
        
        # Set the scalarization weights for the context objectives (assuming 3 objectives in each context)
        self.scalarization_weights = [0.5, 0.3, 0.2]
        
        # Initialize Value function and Policies
        self.V = {}
        self.Pi = {}
        self.Pi_G = {}  # Global synthesized policy
        self.PI = {}  # policy from all contexts in a dictionary mapping context to policy
        self.r_1 = 0
        self.r_2 = 100
        self.r_3 = 100
        for context in Grid.Contexts:
            self.PI[context] = {}
        
        # variables to track the agent path and trahectory for debuging purposes
        self.path = str(self.s)  # + "->"
        self.plan = ""
        self.trajectory = []
        
    def step(self, s, a):
        # s: <s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type>
        # operation actions = ['Noop','pick', 'drop', 'U', 'D', 'L', 'R']
        s = list(s)
        if a == 'pick':
            if self.Grid.All_States[s[0]][s[1]] == 'B':
                s[2] = 'P'
        elif a == 'drop':
            if s[0] == self.goal_loc[0] and s[1] == self.goal_loc[1]:
                s[2] = 'D'
            else:
                s[2] = s[2]
        elif a == 'U' or a == 'D' or a == 'L' or a == 'R':
            # s is the states with the maximum probability
            T = self.get_transition_prob(tuple(s), a)
            s = max(T, key=T.get)
            # s = self.sample_state(tuple(s), a)
        elif a == 'Noop':
            s = s
        else:
            print("INVALID ACTION: ", a)
        s = tuple(s)
        return s
    
    def at_goal(self):
        return self.s == self.s_goal
    
    def sample_state(self, s, a):
        '''Sample the next state based on the policy pi and the current state s based on transition probabilities function'''
        p = random.uniform(0, 1)
        # print("Sampling state: ", s)
        # print("Action for sample: ", a)
        T = self.get_transition_prob(s, a)
        cumulative_prob = 0
        for s_prime in list(T.keys()):
            cumulative_prob += T[s_prime]
            if p <= cumulative_prob:
                return s_prime
        return s_prime
    
    def reset(self):
        self.s = copy.deepcopy(self.s0)
        self.path = str(self.s)
        self.trajectory = []
        self.plan = ""
        self.A = copy.deepcopy(self.A_initial)
        self.r_1 = 0
        self.r_2 = 100
        self.r_3 = 100
        self.R = 0

    def follow_policy_rollout(self, Pi=None):
        self.s = copy.deepcopy(self.s0)
        if Pi is None:
            Pi = copy.deepcopy(self.Pi_G)
        while not self.at_goal():
            # print(str(self.s)+ " -> " + str(Pi[self.s]) + " -> " + str( self.step(self.s, Pi[self.s])))
            self.r_1 += self.Grid.R1_out_context(self.s, Pi[self.s])
            self.r_2 += self.Grid.R2_out_context(self.s, Pi[self.s])
            self.r_3 += self.Grid.R3_out_context(self.s, Pi[self.s])
            self.trajectory.append((self.s, Pi[self.s]))
            self.plan += " -> " + str(Pi[self.s])
            self.s = self.sample_state(self.s, Pi[self.s])  # self.step(self.s, Pi[self.s])
            self.path = self.path + "->" + str(self.s)
            # if s is stuck in a loop or not making progress, break
            if len(self.trajectory) > 100:
                if self.trajectory[-1] == self.trajectory[-5]:
                    print("Agent " + str(self.label) + " is stuck in a loop at s = "+str(self.s)+"!")
                    break
        self.trajectory.append((self.s, Pi[self.s], None))  
              
    def get_trajectory(self, Pi=None):
        self.s = copy.deepcopy(self.s0)
        if Pi is None:
            Pi = copy.deepcopy(self.Pi_G)
        while not self.at_goal():
            s = self.s
            a = Pi[self.s]
            self.trajectory.append((s, a, 
                                    self.Grid.R1_out_context(s, a), 
                                    self.Grid.R2_out_context(s, a), 
                                    self.Grid.R3_out_context(s, a), 
                                    self.Grid.state2context_map[s]))
            self.s = self.sample_state(s, a)  # self.step(self.s, Pi[self.s])
            # if s is stuck in a loop or not making progress, break
            if len(self.trajectory) > 100:
                if self.trajectory[-1] == self.trajectory[-5]:
                    print("Agent " + str(self.label) + " is stuck in a loop at s = "+str(self.s)+"!")
                    break
        self.trajectory.append((self.s, Pi[self.s], 
                                self.Grid.R1_out_context(self.s, Pi[self.s]), 
                                self.Grid.R2_out_context(self.s, Pi[self.s]), 
                                self.Grid.R3_out_context(self.s, Pi[self.s]), 
                                self.Grid.state2context_map[self.s]))
        return self.trajectory
                    
    def get_action_space(self):
        # Get the action space for taxi agent
        A = {}
        S = self.Grid.S
        for s in S:
            A[s] = ['Noop', 'pick', 'drop', 'U', 'D', 'L', 'R']
            A[s] = ['pick', 'drop', 'U', 'D', 'L', 'R']
        # Remove actions that are not possible in certain states
        # s: <s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type>
        # passenger_status = {'X': passenger waiting, 'P': paggenger picked-up, 'D': passenger dropped-off}
        for s in S:
            if s[2] != 'X':
                if 'pick' in A[s]:
                    A[s].remove('pick')
            if s[2] == 'X':
                if 'drop' in A[s]:
                    A[s].remove('drop')
            if self.Grid.All_States[s[0]][s[1]] != 'B':
                if 'pick' in A[s]:
                    A[s].remove('pick')
            if s[4] is True:
                if 'pick' in A[s]:
                    A[s].remove('pick')
            if self.Grid.All_States[s[0]][s[1]] != 'G':
                if 'drop' in A[s]:
                    A[s].remove('drop')
        return A

    def get_transition_prob(self, s, a):
        # s: <s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type>
        # operation actions = ['Noop', 'pick', 'drop', 'U', 'D', 'L', 'R']
        p_success = copy.copy(self.p_success)
        p_fail = 1 - p_success
        action = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
        action_key = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
        if s == self.s_goal or a == 'Noop':
            T = {s: 1}  # stay at the goal with prob = 1
        elif a == 'U' or a == 'D' or a == 'L' or a == 'R':
            s_next_correct = self.move_correctly(s, a)
            s_next_slide_left = self.move_correctly(s, action[(action_key[a] - 1) % 4])
            s_next_slide_right = self.move_correctly(s, action[(action_key[a] + 1) % 4])
            if s_next_correct == s_next_slide_left:
                T = {s_next_correct: round(p_success + p_fail / 2, 3), s_next_slide_right: round(p_fail / 2, 3)}
            elif s_next_correct == s_next_slide_right:
                T = {s_next_correct: round(p_success + p_fail / 2, 3), s_next_slide_left: round(p_fail / 2, 3)}
            else:
                T = {s_next_correct: round(p_success, 3),
                    s_next_slide_left: round(p_fail / 2, 3),
                    s_next_slide_right: round(p_fail / 2, 3)}
        else:
            T = {self.step(s, a): 1}  # (same: 0.2, next: 0.8)
        # create conlficting transitions by removing stochastic slides for certain states manually 
        # but not changing the optimal policy
        if s == (1, 4, 'P', False, 'A') and (a == 'U' or a == 'D' or a == 'L' or a == 'R'):
            T = {self.move_correctly(s, a): 1}
        if s == (2, 4, 'P', False, 'A') and (a == 'U' or a == 'D' or a == 'L' or a == 'R'):
            T = {self.move_correctly(s, a): 1}
        return T

    def move_correctly(self, s, a):
        # s: <s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type>
        # action = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
        s_next = 0
        if a == 'U':
            if s[0] == 0:
                s_next = s
            else:
                if self.Grid.All_States[s[0] - 1][s[1]] in ['P', 'R', 'G', 'B']:
                    s_next = (s[0] - 1, s[1], s[2], self.Grid.All_States[s[0] - 1][s[1]] == 'P', 'R')
                else:
                    s_next = (s[0] - 1, s[1], s[2], False, 'A')
        elif a == 'D':
            if s[0] == self.Grid.rows - 1:
                s_next = s
            else:
                if self.Grid.All_States[s[0] + 1][s[1]] in ['P', 'R', 'G', 'B']:
                    s_next = (s[0] + 1, s[1], s[2], self.Grid.All_States[s[0] + 1][s[1]] == 'P', 'R')
                else:
                    s_next = (s[0] + 1, s[1], s[2], False, 'A')
        elif a == 'L':
            if s[1] == 0:
                s_next = s
            else:
                if self.Grid.All_States[s[0]][s[1] - 1] in ['P', 'R', 'G', 'B']:
                    s_next = (s[0], s[1] - 1, s[2], self.Grid.All_States[s[0]][s[1] - 1] == 'P', 'R')
                else:
                    s_next = (s[0], s[1] - 1, s[2], False, 'A')
        elif a == 'R':
            if s[1] == self.Grid.columns - 1:
                s_next = s
            else:
                if self.Grid.All_States[s[0]][s[1] + 1] in ['P', 'R', 'G', 'B']:
                    s_next = (s[0], s[1] + 1, s[2], self.Grid.All_States[s[0]][s[1] + 1] == 'P', 'R')
                else:
                    s_next = (s[0], s[1] + 1, s[2], False, 'A')
        return s_next
    
    def get_contextual_scalarized_dnn_policy(self):
        w = [0.5, 0.3, 0.2]
        # CCS orderings for
        O = [[0, 1, 2], [1, 0, 2], [2, 0, 1], [0, 2, 1], [1, 2, 0], [2, 1, 0]]
        X = []
        Y = []
        for o in O:
            input_weights = [w[[obj_idx for obj_idx in range(len(o)) if o[obj_idx]==i][0]] for i in range(len(o))]
            _, Pi_for_this_ordering = value_iteration.scalarized_value_iteration(self, o)
            x = [np.array(self.preprocess_input_state(s) + input_weights) for s in self.S]
            X += x
            y = [np.array(self.preprocess_output(Pi_for_this_ordering[s])) for s in self.S]
            Y += y
        X = np.array(X)
        Y = np.array(Y)
        input_dim = X.shape[1]
        model = self.build_model(input_dim)
        batch_size = len(self.S)
        epochs = 1000
        print("Starting to train the DNN model.")
        model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0)
        print("DNN training completed.")
        print("Compiling policy...")
        Pi_G = {}
        for s in self.S:
            ordering = self.Grid.f_w(self.Grid.state2context_map[s])
            weights = [w[[obj_idx for obj_idx in range(len(ordering)) if ordering[obj_idx]==i][0]] for i in range(len(ordering))]
            x = np.array(self.preprocess_input_state(s) + weights)
            x = x.reshape(1, -1)
            y = model.predict(x, verbose=0)
            Pi_G[s] = self.postprocess_output(y)
        print("Policy compiled.")
        return self, Pi_G
                
    def preprocess_output(self, action):
        ''''Return the one-hot encoded output for the given action'''
        # Define the action vocabulary
        action_vocab = ['Noop', 'pick', 'drop', 'U', 'D', 'L', 'R']
        # One-hot encode the action
        action_encoded = [1 if action == a else 0 for a in action_vocab]
        return action_encoded

    def postprocess_output(self, action_encoded):
        '''Return the action for the given one-hot encoded output'''
        action_vocab = ['Noop', 'pick', 'drop', 'U', 'D', 'L', 'R']
        action = action_vocab[np.argmax(action_encoded)]
        return action
    
    def preprocess_input_state(self, state):
        # Unpack the state tuple
        # s: <s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type>
        x, y, passenger_status, pothole, road_type = state
        
        # Convert the booleans to integers
        pothole = int(pothole)
        
        # One-hot encoding the passenger_status (example: 'X' -> [1, 0, 0], 'P' -> [0, 1, 0], 'D' -> [0, 0, 1])
        passenger_status_vocab = ['X', 'P', 'D']
        passenger_status_encoded = [1 if passenger_status == s else 0 for s in passenger_status_vocab]
        
        # One-hot encoding the road_type (example: 'R' -> [1, 0], 'A' -> [0, 1])
        road_type_vocab = ['R', 'A']
        road_type_encoded = [1 if road_type == s else 0 for s in road_type_vocab]
        
        # Return processed inputs as a flat vector
        return [x, y] + passenger_status_encoded + [pothole] + road_type_encoded

    # Define the model architecture
    def build_model(self, input_dim):
        model = models.Sequential()
        
        # Input layer
        model.add(layers.InputLayer(input_shape=(input_dim,)))
        
        # Hidden layers
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        
        # Output layer (7 possible outputs: 0 to 6) with one-hot encoding
        model.add(layers.Dense(7, activation='softmax'))  # Softmax for one-hot encoding
        
        # Compile the model
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',  # Use categorical cross-entropy
                    metrics=['accuracy'])
        
        return model