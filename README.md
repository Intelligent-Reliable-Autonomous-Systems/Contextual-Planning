# Multi-Objective Planning with Contextual Lexicographic Reward Preferences
This code compares our contextual planning approach for multi-objective planning to state-of-the-art baselines. Refer to [project page](https://pulkitrustagi.github.io/contextual-planning-project/) and [paper](https://pulkitrustagi.github.io/contextual-planning-project/) for more information.

### Running the Simulation
Follow these steps to set up the simulation.
### Clone the repository
```bash
$ git clone git@github.com:PulkitRustagi/Contextual-Planning.git
```
### Install required packages
```bash
$ pip install -r requirements.txt
```

## Running the Simulation
All results are generated by running main.py
```bash
$ python main.py
```

## Script Usage
- `main.py` - master script to invoke simulation for all results.
- `simulation.py` - collects simulation results on objective values, goal reachability and conflicts. 
- `global_policy.py` - computes policy for each simulation.
- `metareasoner.py` - contains the conflict checker and resolver.
- `infer_context.py` - consists of modules used for inferring context for *Our Approach 2*.
- `generate_expert_trajectory.py` - generates expert trajectories to collect data for context inference.
- `bayes.py` - for probability calculations used in the context inference process.
- `helper_functions.py` - additional modules to heap read csv and text files.
- `display.py` - modules for plotting and generating animations for policy visualization.
- `read_grid.py` - to read the environment layouts from text files.
- `salp_mdp.py` - environment and agent definitions for salp domain.
- `warehouse_mdp.py` - environment and agent definitions for robotic warehouse domain.
- `taxi_mdp.py` - environment and agent definitions for semi-autonomous taxi domain.

## Folder Usage
- **grids** - text files for all instances of each domain within respective sub-folder.
- **expert_trajectories** - expert trajectories for each domain, generated by `generate_expert_trajectory.py`.
- **animation** -  stores all animations generated by `display.py` if invoked.
- **images** -  images for individual elements to generate animations (no modifications recommended).
- **sim_results** -  to store simulation results in respective sub-folder.

## Miscellaneous
We can generate animations for provided domain, agent, and policy. This can be quite helpful in observing agent behavior under computed policy.
### usage in script
```bash
import diplay
display.animate_policy(domain, agent, policy, savename='animation', stochastic_transition=True)
```
Below is an example visualization in the *salp* domain generated using,
```bash
import display
import global_policy
from salp_mdp import SalpEnvironment, SalpAgent

salp_env = SalpEnvironment("grids/salp/illustration_6x6.txt", context_sim=0)  # for 'Task Only' baseline
salp_agent = SalpAgent(salp_env)  # initialize salp agent in the salp_env environment
salp_agent, policy = global_policy.get_global_policy(salp_agent, context_sim=0)  # compute policy for context_sim = 0 (task only) 
display.animate_policy('salp', salp_agent, policy, savename='animation', stochastic_transition=True)
```
<p align="center">
  <img src="https://github.com/user-attachments/assets/884aa270-c60d-4833-a494-102dd2355cc6" alt="animation" />
</p>
