import copy
import random
import numpy as np
import read_grid
import metareasoner as MR

class SalpEnvironment:
    def __init__(self, filename, context_sim):
        # currently setup as ordering for context i = context_ordering[i]   
        context_ordering = {0: [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 
                            1: [[0, 1, 2], [0, 1, 2], [0, 1, 2]], 
                            2: [[1, 2, 0], [1, 2, 0], [1, 2, 0]], 
                            3: [[0, 1, 2], [1, 0, 2], [2, 0, 1]], 
                            4: [[0, 1, 2], [1, 0, 2], [2, 0, 1]],
                            5: [[0, 1, 2], [1, 0, 2], [2, 0, 1]],
                            6: [[0, 1, 2], [1, 0, 2], [2, 0, 1]],}
        
        self.OMEGA = [1, 2, 0]  # meta ordering over contexts c1 > c2 > c0
        
        # Read the grid from the file and initialize the environment
        All_States, rows, columns = read_grid.grid_read_from_file(filename)
        self.All_States = All_States
        self.rows = rows
        self.columns = columns
        goal_loc = np.where(All_States == 'G')
        self.goal_location = (goal_loc[0][0], goal_loc[1][0])
        # s: <s[0]: x, s[1]: y, s[2]: sample_status, s[3]: coral_flag, s[4]: eddy_flag>
        self.s_goal = (self.goal_location[0], self.goal_location[1], 'D', False, False)
        
        # Initialize parameters for the CLMDP
        self.S = self.get_state_space()
        self.objectives = [i for i in context_ordering[context_sim][0]]
        # removing duplicates in self.objectives
        self.objectives = list(set(self.objectives))
        self.objective_names = {0: 'Task', 1: 'Protect Coral', 2: 'Conserve Battery'}
        self.Contexts = list(range(len(context_ordering[context_sim])))
        self.contextual_orderings = context_ordering[context_sim]  # list of contextual orderings each a list of objectives
        self.Reward_for_obj = self.get_reward_functions()  # list of reward functions for each objective
        self.context_names = self.get_context_name(self.Contexts)
        self.context_map =  MR.get_context_map(self.S, self.Contexts, 'salp')
        self.state2context_map = self.context_map
        self.context2state_map = {}
        # print("Context: ", self.Contexts)
        for context in self.Contexts:
            self.context2state_map[context] = []
        for s in self.S:
            self.context2state_map[self.state2context_map[s]].append(s)

    def get_state_space(self):
        S = []
        for i in range(self.rows):
            for j in range(self.columns):
                for sample in ['X','P','D']:
                    S.append((i, j, sample, self.All_States[i][j]=='C', self.All_States[i][j]=='E'))
        return S

    def get_reward_functions(self):
        R = [self.R1, self.R2, self.R3]
        return R

    def R1(self, s, a):
        # sample delivery reward
        # state of an agent: <s[0]: x, s[1]: y, s[2]: sample_status, s[3]: coral_flag, s[4]: eddy_flag>
        s_next = self.step(s, a)
        if s_next == self.s_goal:
            return 100
        else:
            return -1
    
    def R2(self, s, a):
        # Coral NSE mitigation reward (penalty)
        weighting = {'X': 0.0, 'P': 5.0, 'D': 0.0}
        nse_penalty = 0.0
        # operation actions = ['Noop', 'pick', 'drop', 'U', 'D', 'L', 'R']
        # state of an agent: <s[0]: x, s[1]: y, s[2]: sample_status, s[3]: coral_flag, s[4]: eddy_flag>
        s_next = self.step(s, a)
        if s_next[3] is True:
            nse_penalty = - weighting[s_next[2]]
        else:
            nse_penalty = 0.0
        return nse_penalty
    
    def R3(self, s, a):
        # Eddy current battery draining (penalty)
        # state of an agent: <s[0]: x, s[1]: y, s[2]: sample_status, s[3]: coral_flag, s[4]: eddy_flag>
        s_next = self.step(s, a)        
        if s_next[4] is True:
            R = -5
        else:
            R = -1
        return R
    
    def f_R(self, obj):
        ''''Return the reward function for the given objective'''
        if obj not in self.objectives:
            print("Invalid objective")
            return None
        return self.Reward_for_obj[obj]
    
    def f_w(self, context):
        '''Return the context ordering for the given context'''
        if context not in self.Contexts:
            print("Invalid context")
            return None
        return self.contextual_orderings[context]
    
    def step(self, s, a):
        # state of an agent: <x,y,sample_with_agent,coral_flag,done_flag>
        # operation actions = ['Noop','pick', 'drop', 'U', 'D', 'L', 'R']
        s = list(s)
        if a == 'pick':
            if self.All_States[s[0]][s[1]] == 'B':
                s[2] = 'B'
        elif a == 'drop':
            s[2] = 'X'
            if s[0] == self.goal_location[0] and s[1] == self.goal_location[1]:
                s[4] = True
        elif a == 'U' or a == 'D' or a == 'L' or a == 'R':
            s = self.move_correctly(s, a)  # can be replaced with a sampling function to incorporate stochasticity
        elif a == 'Noop':
            s = s
        else:
            print("INVALID ACTION: ", a)
        s = tuple(s)
        return s
    def move_correctly(self, s, a):
        # action = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
        s_next = 0
        if a == 'U':
            if s[0] == 0:
                s_next = s
            else:
                s_next = (s[0] - 1, s[1], s[2], self.All_States[s[0] - 1][s[1]] == 'C', self.All_States[s[0] - 1][s[1]] == 'E')
        elif a == 'D':
            if s[0] == self.rows - 1:
                s_next = s
            else:
                s_next = (s[0] + 1, s[1], s[2], self.All_States[s[0] + 1][s[1]] == 'C', self.All_States[s[0] + 1][s[1]] == 'E')
        elif a == 'L':
            if s[1] == 0:
                s_next = s
            else:
                s_next = (s[0], s[1] - 1, s[2], self.All_States[s[0]][s[1] - 1] == 'C', self.All_States[s[0]][s[1] - 1] == 'E')
        elif a == 'R':
            if s[1] == self.columns - 1:
                s_next = s
            else:
                s_next = (s[0], s[1] + 1, s[2], self.All_States[s[0]][s[1] + 1] == 'C', self.All_States[s[0]][s[1] + 1] == 'E')
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
    
class SalpAgent:
    def __init__(self, Grid, start_location=(0,0), label='1'):
        
        # Initialize the agent with environment, sample, start location, and label (to identify incase of multi-agent scenario)
        self.Grid = Grid
        self.start_location = start_location
        self.label = label
        self.IDX = int(label) - 1
        self.goal_loc = Grid.goal_location
        
        # Set the scalarization weights for the context objectives (assuming 3 objectives in each context)
        self.scalarization_weights = [0.5, 0.3, 0.2]
        
        # Set the success probability of the agent
        self.p_success = 0.8
        
        # set the start state and the goal state in the right format:
        # s = (x, y, sample, coral, done)
        self.s0 = (start_location[0], start_location[1], 'X', Grid.All_States[start_location[0]][start_location[1]]=='C', Grid.All_States[start_location[0]][start_location[1]]=='E')
        self.s = copy.deepcopy(self.s0)
        self.s_goal = (self.goal_loc[0], self.goal_loc[1], 'D', False, False)
        
        # Initialize state and action space
        self.S = Grid.S
        self.A = self.get_action_space()
        self.A_initial = copy.deepcopy(self.A)
        self.R = 0
        
        # Initialize Value function and Policies
        self.V = {}
        self.Pi = {}
        self.Pi_G = {}  # Global synthesized policy
        self.PI = {}  # policy from all contexts in a dictionary mapping context to policy
        for context in Grid.Contexts:
            self.PI[context] = {}
        
        # variables to track the agent path and trahectory for debuging purposes
        self.path = str(self.s)  # + "->"
        self.plan = ""
        self.trajectory = []
            
    def step(self, s, a):
        # state of an agent: <x,y,sample_status,coral_flag,eddy>
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
        
    def follow_policy(self, Pi=None):
        if Pi is None:
            Pi = copy.deepcopy(self.Pi)
        while not self.at_goal():
            # print(str(self.s)+ " -> " + str(Pi[self.s]) + " -> " + str( self.step(self.s, Pi[self.s])))
            R = self.Grid.R1(self.s, Pi[self.s])
            self.R += R
            self.trajectory.append((self.s, Pi[self.s], R))
            self.plan += " -> " + str(Pi[self.s])
            self.s = self.step(self.s, Pi[self.s])
            self.path = self.path + "->" + str(self.s)
            # if s is stuck in a loop or not making progress, break
            if len(self.trajectory) > 20:
                if self.trajectory[-1] == self.trajectory[-5]:
                    print("Agent " + str(self.label) + " is stuck in a loop!")
                    break
        self.trajectory.append((self.s, Pi[self.s], None))
        
    def follow_policy_rollout(self, Pi=None):
        if Pi is None:
            Pi = copy.deepcopy(self.Pi_G)
        while not self.at_goal():
            # print(str(self.s)+ " -> " + str(Pi[self.s]) + " -> " + str( self.step(self.s, Pi[self.s])))
            R = self.Grid.R1(self.s, Pi[self.s])
            self.R += R
            self.trajectory.append((self.s, Pi[self.s], R))
            self.plan += " -> " + str(Pi[self.s])
            self.s = self.sample_state(self.s, Pi[self.s])  # self.step(self.s, Pi[self.s])
            self.path = self.path + "->" + str(self.s)
            # if s is stuck in a loop or not making progress, break
            if len(self.trajectory) > 20:
                if self.trajectory[-1] == self.trajectory[-5]:
                    print("Agent " + str(self.label) + " is stuck in a loop!")
                    break
        self.trajectory.append((self.s, Pi[self.s], None))
                    
    def get_action_space(self):
        # Get the action space for salp agent
        A = {}
        S = self.Grid.S
        for s in S:
            A[s] = ['Noop', 'pick', 'drop', 'U', 'D', 'L', 'R']
            A[s] = ['pick', 'drop', 'U', 'D', 'L', 'R']
        # Remove actions that are not possible in certain states
        # s = (s[0]: x, s[1]: y, s[2]: sample_status, s[3]: coral_flag, s[4]: eddy)
        # sample_status = {'X': no sample, 'P': sample with agent, 'D': sample delivered}
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
        # state of an agent: <x, y, onboard_sample, coral_flag, done_flag>
        # operation actions = ['Noop', 'pick', 'drop', 'U', 'D', 'L', 'R']
        p_success = copy.copy(self.p_success)
        p_fail = 1 - p_success
        action = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
        action_key = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
        if s == self.s_goal or a == 'Noop':
            T = {s: 1}  # stay at the goal with prob = 1
        elif a == 'U' or a == 'D' or a == 'L' or a == 'R':
            s_next_correct = self.move_correctly(self.Grid, s, a)
            s_next_slide_left = self.move_correctly(self.Grid, s, action[(action_key[a] - 1) % 4])
            s_next_slide_right = self.move_correctly(self.Grid, s, action[(action_key[a] + 1) % 4])
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
        if s == (3, 2, 'P', True, False) and (a == 'U' or a == 'D' or a == 'L' or a == 'R'):
            T = {self.move_correctly(self.Grid, s, a): 1}
        if s == (3, 1, 'P', False, False) and (a == 'U' or a == 'D' or a == 'L' or a == 'R'):
            T = {self.move_correctly(self.Grid, s, a): 1}
        return T

    def move_correctly(self, Grid, s, a):
        # action = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
        # s_next = 0
        if a == 'U':
            if s[0] == 0:
                s_next = s
            else:
                s_next = (s[0] - 1, s[1], s[2], Grid.All_States[s[0] - 1][s[1]] == 'C', Grid.All_States[s[0] - 1][s[1]] == 'E') 
        elif a == 'D':
            if s[0] == Grid.rows - 1:
                s_next = s
            else:
                s_next = (s[0] + 1, s[1], s[2], Grid.All_States[s[0] + 1][s[1]] == 'C', Grid.All_States[s[0] + 1][s[1]] == 'E')
        elif a == 'L':
            if s[1] == 0:
                s_next = s
            else:
                s_next = (s[0], s[1] - 1, s[2], Grid.All_States[s[0]][s[1] - 1] == 'C', Grid.All_States[s[0]][s[1] - 1] == 'E')
        elif a == 'R':
            if s[1] == Grid.columns - 1:
                s_next = s
            else:
                s_next = (s[0], s[1] + 1, s[2], Grid.All_States[s[0]][s[1] + 1] == 'C', Grid.All_States[s[0]][s[1] + 1] == 'E')
        return s_next