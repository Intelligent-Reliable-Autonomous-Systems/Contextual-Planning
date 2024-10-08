import numpy as np
import matplotlib.pyplot as plt
import display

def plot_heatmaps():
    for domain in ['salp', 'taxi', 'warehouse']:
        display.heatmap(domain)
        
def plot_percentile_trajectories():
    for domain in ['salp', 'taxi', 'warehouse']:
        display.percentile_trajectories(domain, 0)
        display.percentile_trajectories(domain, 1)
        display.percentile_trajectories(domain, 2)
        
plot_percentile_trajectories()