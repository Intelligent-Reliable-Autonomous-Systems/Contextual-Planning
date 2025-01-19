#######################################################################################################################
# Description: This script runs all simulations for all domains ['salp', 'warehouse', 'taxi'] and displays the results.
# The results are displayed in the console and saved as heatmaps in the 'results' folder.
# Author: Pulkit Rustagi (rustagi.pulkit@gmail.com)
#######################################################################################################################

import simple_colors
import display
import warnings
from simulation import run_all_sims
warnings.filterwarnings('ignore')

for domain in ['salp', 'warehouse', 'taxi']:
    print(simple_colors.blue('\nRunning '+ domain +' domain...', ['bold', 'underlined']))
    trials = 100  # number of trials for each simulation
    sim_names = {0: 'Task only',
                1: 'LMDP using Omega',
                2: 'Scalarization using Omega',
                3: 'LMDP for Contexts',
                4: 'Yang et al. (2019)', 
                5: 'Contextual Approach w/o resolver',
                6: 'Contextual Approach w/ resolver (Our Approach 1)',
                7: 'Contextual Approach w/ resolver & learned Z (Our Approach 2)',}
    # results from all sims over all grids and trials
    # format for all_sim_results = dict{context_sim [0-7]: [[R1_means], [R2_means], [R3_means], [reached_goal_percentages], [conflict_percentages]]}
    all_sim_results = run_all_sims(domain, sim_names, trials)
    print(simple_colors.green('Simulation completed for '+ domain +' domain!', ['bold', 'underlined']))
    display.report_sim_results_over_grids_and_trails(sim_names, all_sim_results, trials)
    print("done with all simulations for " + domain + " domain")
    display.plot_heatmaps()
    display.plot_min_percentile_for_all_domains()
    wait = input("Press Enter to continue...")  # comment this line to run all simulations at once without pausing