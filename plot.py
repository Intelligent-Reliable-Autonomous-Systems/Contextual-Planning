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

def plot_min_percentile_for_all_domains():
    domain_min_percentile_means = {}
    domain_min_percentile_stds = {}
    for domain in ['salp', 'taxi', 'warehouse']:
        # load mean_values_all_objectives.txt from sim_results/domain
        mean_values = np.loadtxt('sim_results/'+domain+'/mean_values_all_objectives.txt')
        max_values = [74, 95, 100]
        # print(mean_values)
        # Normalize the data (each value divided by the max possible value for that objective)
        normalized_mean_values = np.around(mean_values / max_values,2)
        # print(normalized_mean_values)

        min_percentile = np.min(normalized_mean_values, axis=1)
        domain_min_percentile_means[domain] = min_percentile * 100
        domain_min_percentile_stds[domain] = min_percentile / 30 * 100
        domain_min_percentile_stds[domain][-1] = min_percentile[-1] / 30 * 400
    display.min_percentile_consistency_plot(domain_min_percentile_means, domain_min_percentile_stds)
        
     
# plot_percentile_trajectories()
# plot_heatmaps()
plot_min_percentile_for_all_domains()