import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_barplot_legend_image():
    # Define labels for the legend
    labels = [
        'Task only', 
        'Single preference ordering', 
        'Single preference scalarization',
        'Contextual preference scalarization',
        'Yang et al. (2019)', 
        'Contextual approach w/o resolver', 
        'Contextual approach w/ resolver'
    ]
    keys = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'Our approach']

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
    plt.savefig('custom_legend_image_no_padding.png', bbox_inches='tight', dpi=300)
    # plt.show()

# Call the function to create and save the legend image
create_barplot_legend_image()