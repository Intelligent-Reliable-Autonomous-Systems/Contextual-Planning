import copy
import os
import numpy as np
import simple_colors
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import seaborn as sns
def heatmap():
    # Data for 7 methods: performance on o_1, o_2, and o_3
    # Example: replace this with your actual data
    data = {
            'Task only'+r'$(o_1)$': [71, 64, 67],
            'Lexicographic\nSingle \nPreference'+r'$(\Omega)$': [70, 63, 67],
            'Scalarization \nSingle \nPreference'+r'$(\Omega)$': [66, 67, 88],
            'Scalarization \nContextual \nPreferences': [61, 60, 91],
            'Yang et. al,\n 2019': [34, 49, 85],
            'Contextual \nApproach \nw/o Resolver': [31, 56, 86],
            'Contextual \nApproach \nw/ Resolver': [62, 68, 94]
    }

    # Max possible values for each objective (as given by you)
    max_values = {'o_1': 76, 'o_2': 100, 'o_3': 100}

    # Normalize the data (each value divided by the max possible value for that objective)
    normalized_data = {}
    for method, scores in data.items():
        normalized_data[method] = [
            scores[0] / max_values['o_1'],  # normalize o_1
            scores[1] / max_values['o_2'],  # normalize o_2
            scores[2] / max_values['o_3']   # normalize o_3
        ]

    # Convert the normalized data into a matrix (list of lists)
    heatmap_data = np.array(list(normalized_data.values()))
    # Transpose the data to switch rows and columns
    heatmap_data = heatmap_data.T  # This switches the rows and columns
    # Define the labels for the heatmap (rows as methods, columns as objectives)
    methods = list(normalized_data.keys())
    objectives = [r'$o_1$', r'$o_2$', r'$o_3$']

    # Create the heatmap
    plt.figure(figsize=(10, 3))
    ax = sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd", xticklabels=methods, yticklabels=objectives, cbar_kws={'label': 'Total Reward (Normalized)'})

    # Customize the heatmap
    ax.set_title('Normalized Performance in Objectives in Salp Domain', fontsize=14)
    plt.ylabel('Objectives', fontsize=14)
    # plt.xlabel('Methods', fontsize=14)
    ax.xaxis.set_tick_params(rotation=0)
    ax.yaxis.set_tick_params(rotation=0)
    plt.tight_layout(pad=0) 
    plt.show()
    
def report_sim_results(sim_results, trials, grid_num):
    for context_sim in range(7):
        sim_name, R1_stats, R2_stats, R3_stats, reached_goal_percentage = sim_results[context_sim]
        print(simple_colors.yellow('Results for '+ sim_name + ' [over '+str(trials)+' trials in Grid'+str(grid_num)+']:', ['bold', 'underlined']))
        print(simple_colors.cyan('Objective 1:', ['bold']), R1_stats[0], '±', R1_stats[1])
        print(simple_colors.cyan('Objective 2:', ['bold']), R2_stats[0], '±', R2_stats[1])
        print(simple_colors.cyan('Objective 3:', ['bold']), R3_stats[0], '±', R3_stats[1])
        print(simple_colors.cyan('Reached Goal:', ['bold']), reached_goal_percentage, '% times.')
        print()
        
def report_sim_results_over_grids_and_trails(sim_results_over_grids, trials):
    for context_sim in range(7):
        sim_names, R1_stats, R2_stats, R3_stats, reached_goal_percentage = sim_results_over_grids
        print(simple_colors.yellow('Results for '+ sim_names[context_sim] + ' [over '+str(trials)+' over '+str(len(R1_stats[context_sim]))+' grids (15x15)]:', ['bold', 'underlined']))
        print(simple_colors.cyan('Objective 1:', ['bold']), np.mean(R1_stats[context_sim]))
        print(simple_colors.cyan('Objective 2:', ['bold']), np.mean(R2_stats[context_sim]))
        print(simple_colors.cyan('Objective 3:', ['bold']), np.mean(R3_stats[context_sim]))
        print(simple_colors.cyan('Reached Goal:', ['bold']), np.mean(reached_goal_percentage[context_sim]), '% times.')
        print()
        
def display_animation(frames, savename='animation', fps=4):
    '''Displays the animation of the agent following the given policy.'''
    fig, ax = plt.subplots()
    ax.axis('off')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ims = []

    for frame in frames:
        img = mpimg.imread(frame)
        # remove 85 pixels from each side
        # img = img[85:img.shape[0]-85, 100:img.shape[1]-100,:]
        im = [ax.imshow(img, animated=True)]
        # print('image shape: ', img.shape)
        ims.append(im)

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)
    # plt.show()
    # ani.save('animation/' + savename + '.gif', fps=fps)
    ani.save('animation/animation.gif', fps=fps)

############  Animation for salp domain  ############

def animate_policy_salp(agent, Pi, savename='animation', stochastic_transition=True):
    '''Animates the agent following the given policy.'''
    agent.reset()
    if stochastic_transition:
        agent.follow_policy_rollout(Pi)
    else: 
        agent.follow_policy(Pi)
    if agent.trajectory is None:
        print('Agent has not followed any policy yet!')
        return
    state_list = []
    frames = []
    for tau in agent.trajectory:
        s, a, r = tau
        state_list.append(s)
    location_tracker = []
    for idx, s in enumerate(state_list):
        next_all_state, location_tracker = get_next_all_state_salp(s, agent.Grid, location_tracker)
        get_frame_salp(agent, next_all_state, idx, s, savename)   
        frames.append('animation/{}.png'.format(idx))
    display_animation(frames, savename,fps=4)
    # delete all images
    for idx in range(len(state_list)):
        path = 'animation/{}.png'.format(idx)
        os.remove(path)
    
def get_frame_salp(agent, grid, idx, state=None, savename='animation'):
    # Load the icons
    icon_paths = {
        'B': 'images/salp/sample2.png',
        'G': 'images/salp/testtube.png',
        'C': 'images/salp/coral_translucent.png',
        'D': 'images/salp/coral_opaque_damaged.png',
        'E': 'images/salp/eddy.png',
        '1': 'images/salp/salp.png',
        '2': 'images/salp/salp_s.png',
        '3': 'images/salp/salp_c.png',
        '4': 'images/salp/salp_cs.png',
        '5': 'images/salp/salp_b.png',
        '6': 'images/salp/salp_bs.png',
        '7': 'images/salp/salp_g.png',
        '8': 'images/salp/salp_gs.png',
        '9': 'images/salp/salp_gd2.png',
        '~': 'images/salp/salp_e.png',
        '@': 'images/salp/salp_es.png',
        
    }

    icons = {key: np.flipud(mpimg.imread(path)) for key, path in icon_paths.items()}

    # Determine the size of the grid
    nrows, ncols = len(grid), len(grid[0])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(ncols, nrows))
    # ax.clear()
    # Plot the icons on the grid
    for i in range(nrows):
        for j in range(ncols):
            if grid[i][j] in icons:
                icon = icons[grid[i][j]]
                rect = plt.Rectangle([j, i], 1, 1, facecolor='none', edgecolor='black')
                ax.add_patch(rect)
                ax.imshow(icon, extent=[j, j + 0.95, i, i + 0.95], aspect='auto')
            else:
                rect = plt.Rectangle([j, i], 1, 1, facecolor='white', edgecolor='black')
                ax.add_patch(rect)

    # Set the axis limits and hide the axes
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    # print('state: ', state, 'action: ', agent.Pi_G[state])
    # print('\tR1: ', agent.Grid.R1(state, agent.Pi_G[state]))
    # print('\tR2: ', agent.Grid.R2(state, agent.Pi_G[state]))
    # print('\tR3: ', agent.Grid.R3(state, agent.Pi_G[state]))
    # print()
    ax.title.set_text(savename + "\n" +
                  r"$R_1 = $" + f"{agent.r_1:3d}" + "\t" +
                  r"$R_2 = $" + f"{agent.r_2:3d}" + "\t" +
                  r"$R_3 = $" + f"{agent.r_3:3d}")
    agent.r_1 += agent.Grid.R1(state, agent.Pi_G[state])
    agent.r_2 += agent.Grid.R2(state, agent.Pi_G[state])
    agent.r_3 += agent.Grid.R3(state, agent.Pi_G[state])
    # ax.title.set_text(savename)
    # ax.title.set_text('state: ' + str(state))
    ax.invert_yaxis()
    ax.axis('off')
    
    # save plot as image
    plt.savefig('animation/{}.png'.format(idx))
    plt.close()


def get_next_all_state_salp(s, Grid, location_tracker):
    # s: <s[0]: x, s[1]: y, s[2]: sample_status, s[3]: coral_flag, s[4]: eddy_flag>
    new_all_state = copy.deepcopy(Grid.All_States)
    i, j, sample_status, coral, eddy = s
    if sample_status == 'X' and not coral and not eddy and Grid.All_States[i][j] != 'B':
        new_all_state[i][j] = '1'
    elif sample_status == 'P' and not coral and not eddy and Grid.All_States[i][j] != 'B' and Grid.All_States[i][j] != 'G':
        new_all_state[i][j] = '2'
    elif sample_status == 'X' and coral and not eddy:
        new_all_state[i][j] = '3'
    elif sample_status == 'P' and coral and not eddy:
        new_all_state[i][j] = '4'
    elif Grid.All_States[i][j] == 'B' and sample_status == 'X':
        new_all_state[i][j] = '5'
    elif Grid.All_States[i][j] == 'B' and sample_status == 'P':
        new_all_state[i][j] = '6'
    elif Grid.All_States[i][j] == 'G' and sample_status == 'X':
        new_all_state[i][j] = '7'
    elif Grid.All_States[i][j] == 'G' and sample_status == 'P':
        new_all_state[i][j] = '8'
    elif Grid.All_States[i][j] == 'G' and sample_status == 'D':
        new_all_state[i][j] = '9'
    elif Grid.All_States[i][j] == 'E' and sample_status == 'X':
        new_all_state[i][j] = '~'
    elif Grid.All_States[i][j] == 'E' and sample_status == 'P':
        new_all_state[i][j] = '@'
    for loc in location_tracker:
        i_prev, j_prev = loc
        if new_all_state[i_prev][j_prev] == 'C':
            new_all_state[i_prev][j_prev] = 'D'
    location_tracker.append((i, j))    
    return new_all_state, location_tracker
    
    
############  Animation for warehouse domain  ############

def animate_policy_warehouse(agent, Pi, savename='animation', stochastic_transition=True):
    '''Animates the agent following the given policy.'''
    agent.reset()
    if stochastic_transition:
        agent.follow_policy_rollout(Pi)
    else: 
        agent.follow_policy(Pi)
    if agent.trajectory is None:
        print('Agent has not followed any policy yet!')
        return
    state_list = []
    frames = []
    for tau in agent.trajectory:
        s, a, r = tau
        state_list.append(s)
    location_tracker = []
    for idx, s in enumerate(state_list):
        next_all_state, location_tracker = get_next_all_state_warehouse(s, agent.Grid, location_tracker)
        get_frame_warehouse(agent, next_all_state, idx, s, savename)   
        frames.append('animation/{}.png'.format(idx))
    display_animation(frames, savename,fps=2)
    # delete all images
    for idx in range(len(state_list)):
        path = 'animation/{}.png'.format(idx)
        os.remove(path)
    
def get_frame_warehouse(agent, grid, idx, state=None, savename='animation'):
    # Load the icons
    icon_paths = {
        'B': 'images/warehouse/box.png',
        'G': 'images/warehouse/goal.png',
        'S': 'images/warehouse/slip_tile.png',
        'D': 'images/warehouse/marked_slip_tile.png',
        '#': 'images/warehouse/human.png',
        'A': 'images/warehouse/angry_human.png',
        '1': 'images/warehouse/robot.png',
        '2': 'images/warehouse/robot_and_box.png',
        '3': 'images/warehouse/robot_with_box.png',
        '4': 'images/warehouse/robot_on_human.png',
        '5': 'images/warehouse/robot_with_box_on_human.png',
        '6': 'images/warehouse/robot_with_box_on_goal.png',
        '7': 'images/warehouse/robot_and_box_on_goal.png',
        '8': 'images/warehouse/robot_on_goal.png',
        '9': 'images/warehouse/robot_on_slip_tile.png',
        '0': 'images/warehouse/robot_with_box_on_slip_tile.png',
        '~': 'images/warehouse/robot_with_box_left.png',
        'g': 'images/warehouse/robot_with_box_on_goal_left.png',
        '@': 'images/warehouse/robot_and_box_on_goal_left.png',
        
    }

    icons = {key: np.flipud(mpimg.imread(path)) for key, path in icon_paths.items()}

    # Determine the size of the grid
    nrows, ncols = len(grid), len(grid[0])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(ncols, nrows))
    # ax.clear()

    # Plot the icons on the grid
    for i in range(nrows):
        for j in range(ncols):
            if grid[i][j] in icons:
                icon = icons[grid[i][j]]
                rect = plt.Rectangle([j, i], 1, 1, facecolor='none', edgecolor='black')
                ax.add_patch(rect)
                ax.imshow(icon, extent=[j, j + 1, i, i + 1], aspect='auto')
            else:
                rect = plt.Rectangle([j, i], 1, 1, facecolor='white', edgecolor='black')
                ax.add_patch(rect)

    # Set the axis limits and hide the axes
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    # print('state: ', state, 'action: ', agent.Pi_G[state])
    # print('\tR1: ', agent.Grid.R1(state, agent.Pi_G[state]))
    # print('\tR2: ', agent.Grid.R2(state, agent.Pi_G[state]))
    # print('\tR3: ', agent.Grid.R3(state, agent.Pi_G[state]))
    # print()
    ax.title.set_text(savename + "\n" +
                  r"$R_1 = $" + f"{agent.r_1:3d}" + "\t" +
                  r"$R_2 = $" + f"{agent.r_2:3d}" + "\t" +
                  r"$R_3 = $" + f"{agent.r_3:3d}")
    agent.r_1 += agent.Grid.R1(state, agent.Pi_G[state])
    agent.r_2 += agent.Grid.R2(state, agent.Pi_G[state])
    agent.r_3 += agent.Grid.R3(state, agent.Pi_G[state])
    ax.invert_yaxis()
    ax.axis('off')
    
    # save plot as image
    plt.savefig('animation/{}.png'.format(idx))
    plt.close()


def get_next_all_state_warehouse(s, Grid, location_tracker):
    # s = (s[0]: x, s[1]: y, s[2]: package_status, s[3]: slippery_tile, s[4]: narrow_corridor)
    new_all_state = copy.deepcopy(Grid.All_States)
    i, j, package_status, slippery_tile, narrow_corridor = s
    if package_status == 'X' and not slippery_tile and not narrow_corridor and Grid.All_States[i][j] != 'B':
        new_all_state[i][j] = '1'
    elif package_status == 'X' and not slippery_tile and not narrow_corridor and Grid.All_States[i][j] == 'B' and Grid.All_States[i][j] != 'G':
        new_all_state[i][j] = '2'
    elif package_status == 'P' and not slippery_tile and not narrow_corridor and Grid.All_States[i][j] != 'G':
        new_all_state[i][j] = '3'
    elif package_status == 'X' and not slippery_tile and narrow_corridor:
        new_all_state[i][j] = '4'
    elif package_status == 'P' and not slippery_tile and narrow_corridor:
        new_all_state[i][j] = '5'
    elif package_status == 'P' and not slippery_tile and not narrow_corridor and Grid.All_States[i][j] == 'G':
        new_all_state[i][j] = '6'
    elif package_status == 'D' and not slippery_tile and not narrow_corridor and Grid.All_States[i][j] == 'G':
        new_all_state[i][j] = '7'
    elif package_status == 'X' and not slippery_tile and not narrow_corridor and Grid.All_States[i][j] == 'G':
        new_all_state[i][j] = '8'
    elif package_status == 'X' and slippery_tile and not narrow_corridor:
        new_all_state[i][j] = '9'
    elif package_status == 'P' and slippery_tile and not narrow_corridor:
        new_all_state[i][j] = '0'
    for loc in location_tracker:
        i_prev, j_prev = loc
        if new_all_state[i_prev][j_prev] == 'C':
            new_all_state[i_prev][j_prev] = 'D'
        if new_all_state[i_prev][j_prev] == '#':
            new_all_state[i_prev][j_prev] = 'A'
        if new_all_state[i_prev][j_prev] == 'B':
            new_all_state[i_prev][j_prev] = '.'
        if new_all_state[i_prev][j_prev] == 'S':
            new_all_state[i_prev][j_prev] = 'D'
    if new_all_state[i][j] == '6' and location_tracker[-1][0] == i and location_tracker[-1][1] == j+1:
        new_all_state[i][j] = 'g'
    elif new_all_state[i][j] == '3' and location_tracker[-1][0] == i and location_tracker[-1][1] == j+1 or j > Grid.goal_location[1] and package_status == 'P':
        new_all_state[i][j] = '~'
    elif new_all_state[i][j] == '7' and location_tracker[-1][0] == i and location_tracker[-1][1] == j and location_tracker[-2][1] == j+1:
        new_all_state[i][j] = '@'
    
    location_tracker.append((i, j))
    return new_all_state, location_tracker


    
############  Animation for taxi domain  ############

def animate_policy_taxi(agent, Pi, savename='animation', stochastic_transition=True):
    '''Animates the agent following the given policy.'''
    agent.reset()
    if stochastic_transition:
        agent.follow_policy_rollout(Pi)
    else: 
        agent.follow_policy(Pi)
    if agent.trajectory is None:
        print('Agent has not followed any policy yet!')
        return
    state_list = []
    frames = []
    for tau in agent.trajectory:
        s, a, r = tau
        state_list.append(s)
    location_tracker = []
    for idx, s in enumerate(state_list):
        next_all_state, location_tracker = get_next_all_state_taxi(s, agent.Grid, location_tracker)
        get_frame_taxi(agent, next_all_state, idx, s, savename)   
        frames.append('animation/{}.png'.format(idx))
    display_animation(frames, savename,fps=1.5)
    # delete all images
    for idx in range(len(state_list)):
        path = 'animation/{}.png'.format(idx)
        os.remove(path)
    
def get_frame_taxi(agent, grid, idx, state=None, savename='animation'):
    # Load the icons
    icon_paths = {
        'P': 'images/taxi/pothole.png',
        'R': 'images/taxi/road.png',
        'G': 'images/taxi/goal.png',
        'B': 'images/taxi/passenger.png',
        'b': 'images/taxi/empty_passenger_spot.png',
        'A': 'images/taxi/autonomous_road.png',
        '1': 'images/taxi/taxi.png',
        '2': 'images/taxi/taxi_with_passenger.png',
        '3': 'images/taxi/taxi_with_passenger_on_pothole.png',
        '4': 'images/taxi/taxi_with_passenger_on_autonomous.png',
        '5': 'images/taxi/taxi_with_passenger_at_pickup.png',
        '6': 'images/taxi/taxi_with_passenger_at_goal.png',
        '7': 'images/taxi/taxi_on_pothole.png',
        '8': 'images/taxi/taxi_on_autonomous.png',
        '9': 'images/taxi/taxi_and_passenger_at_pickup.png',
        '0': 'images/taxi/taxi_and_passenger_at_goal.png',
        
    }

    icons = {key: np.flipud(mpimg.imread(path)) for key, path in icon_paths.items()}

    # Determine the size of the grid
    nrows, ncols = len(grid), len(grid[0])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(ncols, nrows))
    # ax.clear()

    # Plot the icons on the grid
    for i in range(nrows):
        for j in range(ncols):
            if grid[i][j] in icons:
                icon = icons[grid[i][j]]
                rect = plt.Rectangle([j, i], 1, 1, facecolor='none', edgecolor='none')
                ax.add_patch(rect)
                ax.imshow(icon, extent=[j, j + 1, i, i + 1], aspect='auto')
            else:
                rect = plt.Rectangle([j, i], 1, 1, facecolor='white', edgecolor='black')
                ax.add_patch(rect)

    # Set the axis limits and hide the axes
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    # print('state: ', state, 'action: ', agent.Pi_G[state])
    # print('\tR1: ', agent.Grid.R1(state, agent.Pi_G[state]))
    # print('\tR2: ', agent.Grid.R2(state, agent.Pi_G[state]))
    # print('\tR3: ', agent.Grid.R3(state, agent.Pi_G[state]))
    # print()
    ax.title.set_text(savename + "\n" +
                  r"$R_1 = $" + f"{agent.r_1:3d}" + "\t" +
                  r"$R_2 = $" + f"{agent.r_2:3d}" + "\t" +
                  r"$R_3 = $" + f"{agent.r_3:3d}")
    agent.r_1 += agent.Grid.R1(state, agent.Pi_G[state])
    agent.r_2 += agent.Grid.R2(state, agent.Pi_G[state])
    agent.r_3 += agent.Grid.R3(state, agent.Pi_G[state])
    ax.invert_yaxis()
    ax.axis('off')
    
    # save plot as image
    plt.savefig('animation/{}.png'.format(idx))
    plt.close()


def get_next_all_state_taxi(s, Grid, location_tracker):
    # s = (s[0]: x, s[1]: y, s[2]: passenger_status, s[3]: pothole, s[4]: road_type)
    new_all_state = copy.deepcopy(Grid.All_States)
    i, j, passenger_status, pothole, road_type = s
    if passenger_status == 'X' and not pothole and road_type == 'R' and Grid.All_States[i][j] not in ['B', 'G']:
        new_all_state[i][j] = '1'
    elif passenger_status == 'P' and not pothole and road_type == 'R' and Grid.All_States[i][j] not in ['B', 'G']:
        new_all_state[i][j] = '2'
    elif passenger_status == 'P' and pothole and road_type == 'R':
        new_all_state[i][j] = '3'
    elif passenger_status == 'P' and not pothole and road_type == 'A':
        new_all_state[i][j] = '4'
    elif passenger_status == 'P' and not pothole and road_type == 'R' and Grid.All_States[i][j] == 'B':
        new_all_state[i][j] = '5'
    elif passenger_status == 'P' and not pothole and road_type == 'R' and Grid.All_States[i][j] == 'G':
        new_all_state[i][j] = '6'
    elif passenger_status == 'X' and pothole and road_type == 'R' and Grid.All_States[i][j] == 'P':
        new_all_state[i][j] = '7'
    elif passenger_status == 'X' and not pothole and road_type == 'A':
        new_all_state[i][j] = '8'
    elif passenger_status == 'X' and not pothole and road_type == 'R' and Grid.All_States[i][j] == 'B':
        new_all_state[i][j] = '9'
    elif passenger_status == 'D' and not pothole and road_type == 'R' and Grid.All_States[i][j] == 'G':
        new_all_state[i][j] = '0'
    for loc in location_tracker:
        i_prev, j_prev = loc
        if new_all_state[i_prev][j_prev] == 'B':
            new_all_state[i_prev][j_prev] = 'b'    
    location_tracker.append((i, j))
    return new_all_state, location_tracker


# heatmap()