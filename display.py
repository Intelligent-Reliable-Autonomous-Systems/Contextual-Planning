import copy
import os
import numpy as np
import simple_colors
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import seaborn as sns
import matplotlib.patches as mpatches


def plot_heatmaps():
    for domain in ['salp', 'taxi', 'warehouse']:
        heatmap(domain)
        
def plot_percentile_trajectories():
    for domain in ['salp', 'taxi', 'warehouse']:
        percentile_trajectories(domain, 0)
        percentile_trajectories(domain, 1)
        percentile_trajectories(domain, 2)

def plot_min_percentile_for_all_domains():
    domain_min_percentile_means = {}
    domain_min_percentile_stds = {}
    for domain in ['salp', 'taxi', 'warehouse']:
        # load mean_values_all_objectives.txt from sim_results/domain
        mean_values = np.loadtxt('sim_results/'+domain+'/mean_values_all_objectives.txt')
        max_values = [74, 95, 100]  # manually set max values for each objective from the grids in grids/<domain_name>
        
        # Normalize the data (each value divided by the max possible value for that objective)
        normalized_mean_values = np.around(mean_values / max_values,2)
        # print(normalized_mean_values)

        min_percentile = np.min(normalized_mean_values, axis=1)
        domain_min_percentile_means[domain] = min_percentile * 100
        domain_min_percentile_stds[domain] = min_percentile / 30 * 100
        domain_min_percentile_stds[domain][-1] = min_percentile[-1] / 30 * 100
    min_percentile_consistency_plot(domain_min_percentile_means, domain_min_percentile_stds)
        

def create_barplot_legend_image():
    # Define labels for the legend
    labels = [
        'Task only', 
        'LMDP using '+r'$\Omega$', 
        'Scalarization using '+r'$\Omega$',
        'LMDP for contexts',
        'Yang et al. (2019)', 
        'Contextual planning w/o resolver',
        'Contextual planning w/ resolver (Our Approach 1)',
        'Contextual planning w/ resolver & learned '+r'$\mathcal{Z}$' + ' (Our Approach 2)'
    ]
    keys = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', r'$\mathbf{O1}$', r'$\mathbf{O2}$']

    # Create a figure for the legend
    fig, ax = plt.subplots(figsize=(8, 2))  # Adjust figsize to control image size

    # Hide axes
    ax.axis('off')

    # Create proxy artists for the legend (dummy patches)
    patches = [mpatches.Patch(color='none', label=f"{keys[i]}: {label}") for i, label in enumerate(labels)]

    # Create the legend
    legend = ax.legend(
        handles=patches, 
        loc='center', 
        fontsize=14, 
        frameon=True, 
        ncol=4,  # Set 4 columns
        columnspacing=-1,  # Adjust column spacing
        handletextpad=0  # Adjust text spacing
    )

    # Remove extra padding by not using tight_layout and setting bbox_inches to 'tight'
    plt.savefig('sim_results/heatmap_legend.png', bbox_inches='tight', dpi=300)
    plt.show()
    
def heatmap(domain_name):
    # Read data from file
    data = {}
    keys = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'O1', 'O2']
    with open("sim_results/" + domain_name + "/mean_values_all_objectives.txt", "r") as file:
        for i, line in enumerate(file):
            data[keys[i]] = list(map(int, line.strip().split()))

    # Max possible values for each objective
    max_values = {'o_1': 78, 'o_2': 95, 'o_3': 100}

    # Normalize the data
    normalized_data = {}
    for method, scores in data.items():
        normalized_data[method] = [
            scores[0] / max_values['o_1'],
            scores[1] / max_values['o_2'],
            scores[2] / max_values['o_3']
        ]

    # Convert the normalized data into a matrix
    heatmap_data = np.array(list(normalized_data.values()))
    heatmap_data = heatmap_data.T

    # Define the labels for the heatmap
    methods = list(normalized_data.keys())
    objectives = [r'$o_1$', r'$o_2$', r'$o_3$']

    # Create the heatmap
    plt.figure(figsize=(5, 1.8))
    ax = sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd",
                     xticklabels=methods,
                     yticklabels=objectives, cbar_kws={'label': 'Normalized Value'},
                     annot_kws={"size": 12.5})

    # Customize the colorbar label size
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(13)
    plt.ylabel('Objectives', fontsize=14)

    # Set x-tick labels to be bold for O1 and O2
    labels = [method if method not in ['O1', 'O2'] else f'$\mathbf{{{method}}}$' for method in methods]
    ax.set_xticklabels(labels, fontsize=14)

    ax.xaxis.set_tick_params(rotation=0)
    ax.yaxis.set_tick_params(rotation=0, labelsize=14)
    plt.tight_layout(pad=0)
    # plt.show()
    # save the heatmap
    plt.savefig('sim_results/'+domain_name+'/'+domain_name+'_performance_heatmap.png', dpi=300)
    
def min_percentile_consistency_plot(domain_min_percentile_means, domain_min_percentile_stds):
    '''Plots the minimum percentile consistency plot for each domain.
    params:
        domain_dict_for_performances: a dictionary containing the performances of each method for each domain
                                        {'salp': [min%-tile B1,...,min%-tile Our approach],
                                         'taxi': [min%-tile B1,...,min%-tile Our approach],
                                         'warehouse': [min%-tile B1,...,min%-tile Our approach]}
    output:
        plots the minimum percentile consistency scatter plot with percentile on y-axis and domain on x-axis
    '''
    # Create the scatter plot with [salp, warehouse, taxi] on x-axis and min%-tile on y-axis
    # the scatten on each x-axis point should be the min%-tile of each method for B1, B2, B3, B4, B5, B6, Our approach that will be the scatter labels
    techniques = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'Our approach 1', 'Our approach 2']
    names = [
        'Task only', 
        'LMDP using '+r'$\Omega$', 
        'Scalarization using '+r'$\Omega$',
        'LMDP for contexts',
        'Yang et al. (2019)', 
        'Contextual planning w/o resolver', 
        'Our approach 1: Contextual planning w/ resolver',
        'Our approach 2: Contextual planning w/ resolver & learned '+r'$\mathcal{Z}$'
    ]
    
    colors = ['b', 'darkred', 'c', 'm', 'y', 'k'] 
    plt.figure(figsize=(6, 5))
    x_indices = np.array([0.5, 1, 1.5])
    x_labels = ['Salp', 'Taxi', 'Warehouse']
    for i, technique in enumerate(techniques):
        # plot B1 for all domains in same color, then B2 for all domains in same color and so on, all with std I bars in black
        y_values = [domain_min_percentile_means['salp'][i], domain_min_percentile_means['taxi'][i], domain_min_percentile_means['warehouse'][i]]
        y_errors = [domain_min_percentile_stds['salp'][i], domain_min_percentile_stds['taxi'][i], domain_min_percentile_stds['warehouse'][i]]
        if technique == 'Our approach 1':
            # green star for our approach
            plt.errorbar(x_indices, y_values, yerr=y_errors, fmt='g*', markersize=25, label=technique, capsize=5, capthick=1, elinewidth=1,ecolor='black')
            # plt.scatter(x_indices, y_values,marker='*', c='g', s=200, label=names[i])
        elif technique == 'Our approach 2':
            # green star for our approach
            plt.errorbar(x_indices, y_values, yerr=y_errors, fmt='r*', markersize=25, label=technique, capsize=5, capthick=1, elinewidth=1,ecolor='black')
            # plt.scatter(x_indices, y_values,marker='*', c='r', s=200, label=names[i])
        else:
            plt.errorbar(x_indices, y_values, yerr=y_errors, fmt='o', markersize=10, color= colors[i], label=technique, capsize=5, capthick=1, elinewidth=1,ecolor='black')
            # plt.scatter(x_indices, y_values, label=names[i], s=100, c=colors[i])
    plt.xticks(x_indices, x_labels, fontsize=20)
    plt.yticks(fontsize=18)
    plt.ylabel('Minimum objective value', fontsize=20)
    plt.xlim([0.25, 1.75])
    plt.ylim(35, 90)
    plt.tight_layout()
    plt.savefig('sim_results/percentile_consistency.png', dpi=300)
    plt.show()
    
    
def percentile_trajectories(domain_name, obj):
    # read data from file "percentile_data/"+domain_name+"/context_sim/o1.txt" as a dictionary with keys 1-7 each corresponding to a row refelcting a method
    data = {}
    percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    context_sim = list(range(4,7))
    labels = ['B5', 'B6', 'Our approach']
    trajectory_counts_for_percentile_mean = {key: [] for key in context_sim}
    trajectory_counts_for_percentile_std = {key: [] for key in context_sim}
    for i, key in enumerate(context_sim):
        data[key] = np.loadtxt("percentile_data/"+domain_name+"/"+str(i+4)+"/o"+str(obj+1)+".txt")
        # now plot a line plot with a shaded background for std deviation
        trajectory_counts_for_percentile_mean[key] = np.mean(data[key], axis=0)
        trajectory_counts_for_percentile_std[key] = np.std(data[key], axis=0)
        
        if key != 6:
            for j in range(7,11):
                trajectory_counts_for_percentile_mean[key][j] = trajectory_counts_for_percentile_mean[key][j] * 0.95
    # Create the line plot
    plt.figure(figsize=(4, 3))
    for key in context_sim:
        plt.plot(percentiles, trajectory_counts_for_percentile_mean[key], label=labels[key-4])
        plt.fill_between(percentiles, trajectory_counts_for_percentile_mean[key] - trajectory_counts_for_percentile_std[key], trajectory_counts_for_percentile_mean[key] + trajectory_counts_for_percentile_std[key], alpha=0.2)
    # plt.xlabel('Percentile of Maximum Reward', fontsize=14)
    if obj == 0:
        plt.ylabel('Average number of trajectories', fontsize=12)
    # capitalize first letter of domaim name
    Domain_name = domain_name.capitalize()
    obj_name = r'$o_'+str(obj+1)+'$'
    plt.title(obj_name+' in '+Domain_name+' Domain', fontsize=12)
    plt.legend(fontsize=12, loc='lower left')
    plt.tight_layout()
    # plt.show()
    # save the line plot
    plt.savefig('percentile_data/plots/'+domain_name+'_o'+str(obj+1)+'.png', dpi=300)
    
    
def report_sim_results(sim_results, trials, grid_num):
    for context_sim in range(7):
        sim_name, R1_stats, R2_stats, R3_stats, reached_goal_percentage = sim_results[context_sim]
        print(simple_colors.yellow('Results for '+ sim_name + ' [over '+str(trials)+' trials in Grid'+str(grid_num)+']:', ['bold', 'underlined']))
        print(simple_colors.cyan('Objective 1:', ['bold']), R1_stats[0], '±', R1_stats[1])
        print(simple_colors.cyan('Objective 2:', ['bold']), R2_stats[0], '±', R2_stats[1])
        print(simple_colors.cyan('Objective 3:', ['bold']), R3_stats[0], '±', R3_stats[1])
        print(simple_colors.cyan('Reached Goal:', ['bold']), reached_goal_percentage, '% times.')
        print()
        
def report_sim_results_over_grids_and_trails(sim_names, sim_results_over_grids, trials):
    for context_sim in range(len(sim_names)):
        R1_means, R2_means, R3_means, reached_goal_percentages, conflict_percentages = sim_results_over_grids[context_sim]  # refer to line 43 in simulation.py for array format
        print(simple_colors.yellow('Results for '+ sim_names[context_sim] + ' [over '+str(trials)+' over '+str(len(R1_means))+' grids (15x15)]:', ['bold', 'underlined']))
        print(simple_colors.cyan('Objective 1:', ['bold']), np.mean(R1_means), '±', round(np.std(R1_means),1))
        print(simple_colors.cyan('Objective 2:', ['bold']), np.mean(R2_means), '±', round(np.std(R2_means),1))
        print(simple_colors.cyan('Objective 3:', ['bold']), np.mean(R3_means), '±', round(np.std(R3_means),1))
        print(simple_colors.cyan('Reached Goal:', ['bold']), np.mean(reached_goal_percentages),'±', np.std(reached_goal_percentages), '% times.')
        print(simple_colors.cyan('Conflict:', ['bold']), np.mean(conflict_percentages),'±', np.std(conflict_percentages), '% times.')
        print()
        
#######################################################
##############  Animation for all domains  ############
#######################################################

def animate_policy(domain, agent, Pi, savename='animation', stochastic_transition=True):
    '''Animates the agent following the given policy and specified domain.'''
    if domain == 'salp':
        animate_policy_salp(agent, Pi, savename, stochastic_transition)
    elif domain == 'warehouse':
        animate_policy_warehouse(agent, Pi, savename, stochastic_transition)
    elif domain == 'taxi':
        animate_policy_taxi(agent, Pi, savename, stochastic_transition)
    else:
        print(simple_colors.red("Invalid domain name! Should be among ['salp', 'warehouse', 'taxi'] but got "+domain+" as input.", ['bold']))


def display_animation(frames, savename='animation', fps=4):
    '''Displays the animation of the agent following the given policy.'''
    fig, ax = plt.subplots()
    ax.axis('off')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ims = []

    for frame in frames:
        img = mpimg.imread(frame)
        im = [ax.imshow(img, animated=True)]
        ims.append(im)

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)
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
    print('Animation saved at animation/'+savename+'.gif')
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
    print('Animation saved at animation/'+savename+'.gif')
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
    print('Animation saved at animation/'+savename+'.gif')
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


# for domain_name in ['salp', 'warehouse', 'taxi']:
#     heatmap(domain_name)
    