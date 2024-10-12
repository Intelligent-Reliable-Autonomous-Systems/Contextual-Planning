import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_barplot_legend_image():
    # Define labels for the legend
    labels = [
        'Task only', 
        'LMDP using '+r'$\Omega$', 
        'Scalarization using '+r'$\Omega$',
        'LMDP for contexts',
        'Yang et al. (2019)', 
        'Contextual planning w/o resolver', # Contextual Planning for Multi-Objective Reinforcement Learning
        'Contextual planning w/ resolver (Our Approach 1)',
        'Contextual planning w/ resolver w/ learned '+r'$\mathcal{Z}$' + ' (Our Approach 2)'
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
    plt.savefig('heatmap_legend.png', bbox_inches='tight', dpi=300)
    # plt.show()

# Example call to the function
create_barplot_legend_image()