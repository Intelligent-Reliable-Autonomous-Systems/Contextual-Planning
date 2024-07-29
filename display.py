import copy
import os
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def animate_policy(agent, Pi, savename='animation', stochastic_transition=True):
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
        next_all_state, location_tracker = get_next_all_state(s, agent.Grid, location_tracker)
        get_frame(next_all_state, idx, s, savename)   
        frames.append('animation/{}.png'.format(idx))
    display_animation(frames, savename)
    # delete all images
    for idx in range(len(state_list)):
        path = 'animation/{}.png'.format(idx)
        os.remove(path)
    # display the gif
    # img = Image.open('animation/animation.gif')
    # plt.figure(figsize=(img.width / 100, img.height / 100))
    # plt.axis('off')
    # plt.imshow(img)
    # plt.show()

def display_animation(frames, savename='animation'):
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
    ani.save('animation/' + savename + '.gif', fps=4)
    ani.save('animation/animation.gif', fps=4)
    
def get_frame(grid, idx, state=None, savename='animation'):
    # Load the icons
    icon_paths = {
        'B': 'images/sample2.png',
        'G': 'images/testtube.png',
        'C': 'images/coral_translucent.png',
        'D': 'images/coral_opaque_damaged.png',
        '1': 'images/salp.png',
        '2': 'images/salp_s.png',
        '3': 'images/salp_c.png',
        '4': 'images/salp_cs.png',
        '5': 'images/salp_b.png',
        '6': 'images/salp_bs.png',
        '7': 'images/salp_g.png',
        '8': 'images/salp_gs.png',
        '9': 'images/salp_gd.png',
        
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
    # ax.title.set_text(savename)
    # ax.title.set_text('state: ' + str(state))
    ax.invert_yaxis()
    ax.axis('off')
    
    # save plot as image
    plt.savefig('animation/{}.png'.format(idx))
    plt.close()


def get_next_all_state(s, Grid, location_tracker):
    # s = (x, y, sample, coral, done)
    new_all_state = copy.deepcopy(Grid.All_States)
    i, j, sample, coral, done = s
    sample_flag = sample != 'X'
    if not sample_flag and not coral and Grid.All_States[i][j] != 'B' and not done:
        new_all_state[i][j] = '1'
    elif sample_flag and not coral and Grid.All_States[i][j] != 'B' and Grid.All_States[i][j] != 'G':
        new_all_state[i][j] = '2'
    elif not sample_flag and coral:
        new_all_state[i][j] = '3'
    elif sample_flag and coral:
        new_all_state[i][j] = '4'
    elif Grid.All_States[i][j] == 'B' and not sample_flag:
        new_all_state[i][j] = '5'
    elif Grid.All_States[i][j] == 'B' and sample_flag:
        new_all_state[i][j] = '6'
    elif Grid.All_States[i][j] == 'G' and not sample_flag and not done:
        new_all_state[i][j] = '7'
    elif Grid.All_States[i][j] == 'G' and sample_flag:
        new_all_state[i][j] = '8'
    elif Grid.All_States[i][j] == 'G' and not sample_flag and done:
        new_all_state[i][j] = '9'
    for loc in location_tracker:
        i_prev, j_prev = loc
        if new_all_state[i_prev][j_prev] == 'C':
            new_all_state[i_prev][j_prev] = 'D'
    location_tracker.append((i, j))    
    return new_all_state, location_tracker
    
    
    
# '1': 'images/salp.png',
# '2': 'images/salp_s.png',
# '3': 'images/salp_c.png',
# '4': 'images/salp_cs.png',
# '5': 'images/salp_b.png',
# '6': 'images/salp_bs.png',
# '7': 'images/salp_g.png',
# '8': 'images/salp_gs.png',
# '9': 'images/salp_gd.png',
# # Define your 2D array
# grid = [
#     ['1', '2', '3'],
#     ['4', '5', '6'],
#     ['7', '8', '9'],
# ]

# visualize(grid)