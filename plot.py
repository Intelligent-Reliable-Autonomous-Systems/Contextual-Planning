import numpy as np
import matplotlib.pyplot as plt

def domain_objective_performance_bargraph(domain_name):
    '''Plots the objective performance of the agents for a given domain
    1. read all the text files in sim_results/domain_name
    2. get the mean and std of each objective column-wise
    3. plot a bar graph with the mean and std of each objective where the x_ticks are o1, o2, o3; and the y-axis is the mean value of each objective with std error bars
    '''
    # read all the text files in sim_results/domain_name
    task_only = np.loadtxt('sim_results/'+ domain_name +'/Task only.txt')
    task_nse = np.loadtxt('sim_results/'+ domain_name +'/Task > NSE Mitigation.txt')
    nse_task = np.loadtxt('sim_results/'+ domain_name +'/NSE Mitigation > Task.txt')
    scalarization_single = np.loadtxt('sim_results/'+ domain_name +'/Scalarization Single Preference.txt')
    scalarization_contextual = np.loadtxt('sim_results/'+ domain_name +'/Scalarization Contextual Preferences.txt')
    contextual_dnn = np.loadtxt('sim_results/'+ domain_name +'/Contextual Scalarization DNN.txt')
    contextual_no_conflict = np.loadtxt('sim_results/'+ domain_name +'/Contextual Approach without Conflict Resolution.txt')
    contextual_conflict = np.loadtxt('sim_results/'+ domain_name +'/Contextual Approach with Conflict Resolution.txt')
    # get the mean and std of each objective column-wise
    task_only_mean = np.mean(task_only, axis=0)
    task_only_std = np.std(task_only, axis=0)
    task_nse_mean = np.mean(task_nse, axis=0)
    task_nse_std = np.std(task_nse, axis=0)
    nse_task_mean = np.mean(nse_task, axis=0)
    nse_task_std = np.std(nse_task, axis=0) 
    scalarization_single_mean = np.mean(scalarization_single, axis=0)
    scalarization_single_std = np.std(scalarization_single, axis=0)
    scalarization_contextual_mean = np.mean(scalarization_contextual, axis=0)
    scalarization_contextual_std = np.std(scalarization_contextual, axis=0)
    contextual_dnn_mean = np.mean(contextual_dnn, axis=0)
    contextual_dnn_std = np.std(contextual_dnn, axis=0)
    contextual_no_conflict_mean = np.mean(contextual_no_conflict, axis=0)
    contextual_no_conflict_std = np.std(contextual_no_conflict, axis=0)
    contextual_conflict_mean = np.mean(contextual_conflict, axis=0)
    contextual_conflict_std = np.std(contextual_conflict, axis=0)
    # plot a bar graph with the mean and std of each objective where the x_ticks are o1, o2, o3; and the y-axis is the mean value of each objective with std error bars
    fig, ax = plt.subplots()
    x = np.arange(3)
    width = 0.05
    ax.bar(x - 3.5*width, task_only_mean, width, yerr=task_only_std, label='Task only')
    ax.bar(x - 2.5*width, task_nse_mean, width, yerr=task_nse_std, label='Task > NSE Mitigation')
    ax.bar(x + -1.5*width, nse_task_mean, width, yerr=nse_task_std, label='NSE Mitigation > Task')
    ax.bar(x + -0.5*width, scalarization_single_mean, width, yerr=scalarization_single_std, label='Scalarization Single Preference')
    ax.bar(x + 0.5*width, scalarization_contextual_mean, width, yerr=scalarization_contextual_std, label='Scalarization Contextual Preferences')
    ax.bar(x + 1.5*width, contextual_dnn_mean, width, yerr=contextual_dnn_std, label='Contextual Scalarization DNN')
    ax.bar(x + 2.5*width, contextual_no_conflict_mean, width, yerr=contextual_no_conflict_std, label='Contextual Approach without Conflict Resolution')
    ax.bar(x + 3.5*width, contextual_conflict_mean, width, yerr=contextual_conflict_std, label='Contextual Approach with Conflict Resolution')
    ax.set_xticks(x)
    ax.set_xticklabels(['Objective 1', 'Objective 2', 'Objective 3'])
    
    ax.set_ylabel('Mean Objective Value')
    ax.set_title('Objective Performances in ' + domain_name + ' domain')
    
    ax.legend()
    plt.show()
    
domain_objective_performance_bargraph('salp')
domain_objective_performance_bargraph('warehouse')
domain_objective_performance_bargraph('taxi')
    
