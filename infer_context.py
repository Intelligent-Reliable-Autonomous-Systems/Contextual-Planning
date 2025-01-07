from helper_functions import *
from salp_mdp import SalpEnvironment, SalpAgent
from warehouse_mdp import WarehouseEnvironment, WarehouseAgent
from taxi_mdp import TaxiEnvironment, TaxiAgent
from scm import *
from bayes import get_context
from pathlib import Path

def get_context_from_inference(domain, start_state='random', with_exp_traj=False):
    test_filenames = [
                    'grids/'+domain+'/illustration0_15x15.txt',
                    'grids/'+domain+'/illustration1_15x15.txt',
                    'grids/'+domain+'/illustration2_15x15.txt',
                    'grids/'+domain+'/illustration3_15x15.txt',
                    'grids/'+domain+'/illustration4_15x15.txt'
                ]

    for te_instance_idx, te_file in enumerate(test_filenames):

        if with_exp_traj==True:
            train_grid_idx = te_instance_idx
            train_env_file = 'grids/'+domain+'/illustration'+str(train_grid_idx)+'_15x15.txt'
            tr_filename = 'expert_trajectories/'+domain+'/illustration'+str(train_grid_idx)+'_'+start_state+'_start_states.txt'
            output_dir = 'context_inference/'+domain+'/'
            
            if domain == 'salp':
                tr_env = SalpEnvironment(train_env_file, context_sim=7)
                tr_env_agent = SalpAgent(tr_env)
            elif domain == 'warehouse':
                tr_env = WarehouseEnvironment(train_env_file, context_sim=7)
                tr_env_agent = WarehouseAgent(tr_env)
            elif domain == 'taxi':
                tr_env = TaxiEnvironment(train_env_file, context_sim=7)
                tr_env_agent = TaxiAgent(tr_env)

            tr_data = get_data(tr_filename, tr_env, tr_env_agent, domain, is_train=True)

            context_map = Path(output_dir+'train_data_context_maps_illustration'+str(train_grid_idx)+start_state+'_start.txt')
            context_map.parent.mkdir(parents=True, exist_ok=True)
            with open(context_map, 'w') as file:
                    file.write("State\tAction\tR1\tR2\tR3\tContext\tPossible_Contexts\n")
                    for index, row in tr_data.iterrows():
                        file.write("\t".join(map(str, row.values)) + "\n")
            te_inference_filename = output_dir+'bayes/illustration'+str(te_instance_idx)+'inference_results.csv'

        print('test instance: ', te_instance_idx)
        if domain == 'salp':
            te_env = SalpEnvironment(te_file, context_sim=7)
            te_env_agent = SalpAgent(te_env)
        elif domain == 'warehouse':
            te_env = WarehouseEnvironment(te_file, context_sim=7)
            te_env_agent = WarehouseAgent(te_env)
        elif domain == 'taxi':
            te_env = TaxiEnvironment(te_file, context_sim=7)
            te_env_agent = TaxiAgent(te_env)

        te_data = get_data(te_file, te_env, te_env_agent, domain, is_train=False)
        use_expert_traj = False
        if with_exp_traj==True and te_instance_idx==train_grid_idx:
            use_expert_traj = True
        if with_exp_traj==False and use_expert_traj==False:
            output_dir = 'context_inference/output_without_exp_data/'+domain+'/'
            performance_filename = output_dir+'performance_results.csv'
            te_inference_filename = output_dir+'bayes/illustration'+str(te_instance_idx)+'inference_results.csv'
            tr_data = None
        inferred_contexts_bayes = get_context(te_env, te_env_agent, tr_data, te_data, domain, te_inference_filename, use_expert_traj)
        # print(inferred_contexts_bayes)
        evaluate_inferred_contexts(inferred_contexts_bayes, te_instance_idx, performance_filename)