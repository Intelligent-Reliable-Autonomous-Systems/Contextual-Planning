import simple_colors
import display
import warnings
from simulation import run_all_sims
warnings.filterwarnings('ignore')

for domain in ['salp', 'warehouse', 'taxi']:
    trials = 100  # number of trials for each simulation
    sim_name = {0: 'Task only',
                1: 'LMDP using Omega', 
                2: 'Scalarization using Omega',
                3: 'LMDP for Contexts',
                4: 'Yang et al. (2019)', 
                5: 'Contextual Approach w/o resolver',
                6: 'Contextual Approach w/ resolver (Our Approach 1)',
                7: 'Contextual Approach w/ resolver & learned Z (Our Approach 2)',}
    
    sim_results = run_all_sims(domain, sim_name, trials)
    print(simple_colors.green('Simulation completed for '+ domain +' domain!', ['bold', 'underlined']))
    display.report_sim_results_over_grids_and_trails(sim_results, trials)
    
    display.plot_heatmaps()
    display.plot_min_percentile_for_all_domains()
    wait = input("Press Enter to continue...")  # comment this line to run all simulations at once without pausing