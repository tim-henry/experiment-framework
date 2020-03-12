import os
from os.path import basename
if basename(os.getcwd()) == 'util':
    os.chdir("../")
import config.model
if basename(os.getcwd()) == 'experiment-framework':
    os.chdir("util/")
import numpy as np
import torch
import pickle
import copy
from collections import OrderedDict
import matplotlib
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import torch.utils.data as utils
import torch.nn.functional as F




#-------------------------------- PLOTTING --------------------------------
'''         PLOT_COMBINATION_ACTS
For 10x10 individual neuron activation plots

Plot num_classes x num_classes heat map for each neuron in [neuron_idx_start, neuron_idx_stop]

combo_matrix: 2D float array - array of matrices representing num_classes x num_classes heat maps
        for EVERY neuron (Create full array before running method)
normalize: string - individual -> based on individual neurons, group -> based on all neurons
num_classes: int - the dimension of the matrix
plot_single_column: bool - True if want to plot each matrix separately
dataset_name: string - name of the dataset used
plot_size: int - use to adjust size of the plot
'''
def plot_combination_acts(combo_matrix, neuron_idx_start, neuron_idx_end,
                          normalize='individual', num_classes=10,
                          plot_single_column=True, dataset_name=None,
                          plot_size=6,save=False,save_path=""):

    for i in range(neuron_idx_start, neuron_idx_end):
        if np.max(combo_matrix[i]) == 0:
            normalized_matrix = combo_matrix[i]

        elif normalize == 'group':
            normalized_matrix = combo_matrix[i] / np.max(combo_matrix[neuron_idx_start: neuron_idx_end])

        elif normalize == 'individual':
            normalized_matrix = combo_matrix[i] / np.max(combo_matrix[i])

        else:
            normalized_matrix = combo_matrix[i][:]

        fig, ax = plt.subplots()

        num_rows = len(combo_matrix[i])
        num_cols = len(combo_matrix[i][0])
        for a in range(num_rows):
            for b in range(num_cols):
                text = ax.text(b, a, round(normalized_matrix[a,b],2),
                              ha = 'center', va='center', color='w',size=7)

        ax.title.set_text("neuron "+str(i))
        ax.imshow(normalized_matrix)
        if dataset_name == "left_out_colored_mnist":
            ax.set_xlabel('color')
            ax.set_ylabel('shape')
        elif dataset_name == "left_out_varied_location_mnist":
            ax.set_xlabel('position')
            ax.set_ylabel('shape')
        elif dataset_name == "left_out_many_scale_mnist":
            ax.set_xlabel('scale')
            ax.set_ylabel('shape')

        fig.set_size_inches(plot_size,plot_size)
        if save:
            if save_path == "":
                plt.savefig("neuron"+str(i)+".pdf")
            else:
                plt.savefig(save_path+"/neuron"+str(i)+".pdf")
        plt.show()

plt.show()


''' PLOT_ACT_MATRIX
Plot a given activation matrix/heatmap

act_matrix: 2D float array - activation matrix
            x-axis -> neuron indices
            y-axis -> combinations
combinations_list: (int,int) list - list of all possible (task1,task2) combos
n_clusters: int - number of clusters used to organize similar neuron units.
            setting to <= 1 means no clustering
'''
def plot_act_matrix(act_matrix, combinations_list, n_clusters=1, title_dataset=None, save=False, save_path=""):
    if title_dataset is None:
        title_dataset = ""
    elif title_dataset.lower() == "color":
        title_dataset = "Shape/Color "
    elif title_dataset.lower() in ["location","position"]:
        title_dataset = "Shape/Position "
    elif title_dataset.lower() == "scale":
        title_dataset = "Shape/Scale "

    fig, ax = plt.subplots()
    out = act_matrix[:]

    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(out.T)
        order = kmeans.labels_
        osort = order.argsort()
        out = out[:,osort[:]]
        sorted_order = order[osort[:]]

        xticks = np.array([])
        for i in range(n_clusters):
            to_append = np.where(sorted_order == i)
            xticks = np.append(xticks, to_append[0][0])

        ax.set_xticks(xticks)
        ax.set_xticklabels([n+1 for n in range(n_clusters)], fontsize=14)

        plt.xlabel("cluster",fontsize=25)
    else:
        plt.xlabel("neuron index")

    plt.ylabel("composition",fontsize=25)

    fig.set_size_inches(20,12)
    im = ax.imshow(out, aspect='auto')

    title = "Activations Plot for "+title_dataset+"Compositions"

    plt.title(title,fontsize=25)
    #ax.set_yticks([i for i in range(len(combinations_list))])
    ax.set_yticklabels(["" for c in combinations_list])
    #ax.set_yticklabels([str(c) for c in combinations_list], fontsize=11)
    if save:
        if save_path == "":
            plt.savefig("full_acts.pdf")
        else:
            plt.savefig(save_path)
    plt.show()


''' PLOT_ACCURACY_TABLE *DEBUG*
plot table of accuracies.

data: 2D int array - num_classes x num_classes size array showing ablation accuracies.
            rows -> class being ablated, cols -> class whose accuracies are being evaluated
            (fill out data before passing)

title: string - what to call the table
num_classes: int - number of classes. determines table dimensions
'''
def plot_accuracy_table(data, title="", num_classes=10):
    row_labels=['ablated class '+str(n) for n in range(num_classes)]
    col_labels=['class '+str(n)+ ' acc' for n in range(num_classes)]
    fig, ax = plt.subplots()

    table = plt.table(cellText = data,
              rowLabels=row_labels,
              colLabels=col_labels,
              loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(4,4)

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    plt.show()

'''PLOT_ABLATION_ACCURACIES

task_names: (str,str) list - list of unique task names
               will be used to make ablation_list: (str,str) list - (task1, task1), (task1, task2),
               (task2,task1), (task2,task2), (random,task1), (random,task2) *(in that order)*.
               To be used to build the name of the filepath to ablation data.
               First idx = what group neurons were ablated from.
               Sec idx = the group whose examples were used to test resulting accuracy

model_name: str - the name of the model used
dataset_name: str - the name of the dataset used
num_classes: int - number of classes in each task (should be a single number, same # classes for both tasks)
'''
def plot_ablation_accuracies(task_names, model_name, dataset_name, num_classes=10, showmedians=True, showmeans=False, showextrema=True, save=False, name=None, save_path=""):
    if len(task_names) > 2:
        print("Haven't added code to deal with more than 2 tasks at a time")
        return

    file = "../analysis/" + model_name + "/ablation_data/" +\
            dataset_name + "/ablate_{}-test_{}.npy"
    file = file.format

    ablation_list = [(task_names[0],task_names[0]), (task_names[0],task_names[1]),
                     (task_names[1],task_names[0]), (task_names[1],task_names[1]),
                     ("random",task_names[0]),("random",task_names[1])]

    d1,d2,d3,d4,d5,d6 = [assign_val(file(str1,str2)) for str1,str2 in ablation_list]
    if d1 is None or d2 is None or d3 is None or d4 is None or d5 is None or d6 is None:
        print("No "+task_names[0]+"/"+task_names[1]+" ablation accuracy data, or wrong filenames:")
        print([file(str1,str2) for (str1,str2) in ablation_list])
        return


    task1_diagonal_vals = np.diagonal(d1)
    task2_diagonal_vals = np.diagonal(d4)
    task1_nondiagonal_vals = d1[np.identity(num_classes)==0]
    task2_nondiagonal_vals = d4[np.identity(num_classes)==0]
    task1_ablate_task2_vals = d3
    task2_ablate_task1_vals = d2
    task1_ablate_random_vals = d5
    task2_ablate_random_vals = d6

    yticklabels = [str(r)+'%' for r in range(0,110,10)]

    def plot_task_violin(task_name, task_diag, task_nondiag, task_ablate, task_ablate_random):
        fig,ax = plt.subplots()
        task_violin = [task_diag, task_nondiag, task_ablate, task_ablate_random]
        ax.set_title(make_first_upper(task_name) + " Ablation Accuracies for "+make_first_upper(model_name))
        ax.set_xticks([1,2,3,4])
        ax.set_xticklabels(['same class','other classes','other task(s)','random ablations'])
        ax.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
        ax.set_yticklabels(yticklabels)
        plt.violinplot(task_violin, showmedians=showmedians, showextrema=showextrema, showmeans=showmeans)
        if save:
            if save_path == "":
                plt.savefig(name + "_ablation_accuracies_"+task_name+".pdf")
            else:
                plt.savefig(save_path + name + "_ablation_accuracies_"+task_name+".pdf")
        plt.show()

    plot_task_violin(task_names[0], task1_diagonal_vals, task1_nondiagonal_vals, task1_ablate_task2_vals, task1_ablate_random_vals)
    plot_task_violin(task_names[1], task2_diagonal_vals, task2_nondiagonal_vals, task2_ablate_task1_vals, task2_ablate_random_vals)



#--------------------- POPULATING DICTIONARIES -------------------------------
''' UPDATE_ACTS_DICTS
Populate activation dictionaries for all combinations,
    all averaged over task1(task2_acts_dict), and all
    averaged over task2(task1_acts_dict)

    if use_threshold, instead of just taking mean of num_examples
    for each combo and doing math, we first process by taking function
    that gets percentage of examples over 100-percentile_threshold percentile
    value

combo_list: (int,int) list - list of all possible attribute compositions
acts_dict: (int, int) -> 2D int array - each combo maps to a num_examples x num_neurons
            array representing the output activations of all num_neurons units
            in a given layer to some specific (task1,task2) combination

TODO: make function robust to updating more than just two tasks
combo_acts_dict: *empty dict*; becomes (int,int) -> float list - each combo maps to some num_neurons array
            representing the average of acts_dict across axis 0
task1_acts_dict: *emoty dict*; becomes int -> float list - each of (task1,task2)[0]
            is concatenated then averaged to get avg firing rate for the entire task
task2_acts_dict: same as task1_acts_dict, but using (task1,task2)[1]
'''
def update_acts_dicts(combo_list, acts_dict, combo_acts_dict, task1_acts_dict, task2_acts_dict, num_neurons=512, use_threshold=False, percentile_threshold=1):
    if use_threshold:
        percentile_array = get_percentile_array(acts_dict, percentile_threshold)
        combo_thres_dict = get_acts_thresholding(acts_dict, percentile_array)

    for idx,k in enumerate(combo_list):
        if not use_threshold:
            combo_acts_dict[k] = np.mean(acts_dict[k],axis=0)
        else:
            combo_acts_dict[k] = combo_thres_dict[k]

        val1 = copy.deepcopy(combo_acts_dict[k])
        val2 = copy.deepcopy(combo_acts_dict[k])

        if k[0] in task1_acts_dict:
            task1_acts_dict[k[0]] = np.concatenate([task1_acts_dict[k[0]], val1.reshape((1,num_neurons))])
        else:
            task1_acts_dict[k[0]] = val1.reshape((1,num_neurons))


        if k[1] in task2_acts_dict:
            task2_acts_dict[k[1]] = np.concatenate([task2_acts_dict[k[1]], val2.reshape((1,num_neurons))])
        else:
            task2_acts_dict[k[1]] =  val2.reshape((1,num_neurons))

    for k in task1_acts_dict:
        task1_acts_dict[k] = np.mean(task1_acts_dict[k], axis=0)
        task2_acts_dict[k] = np.mean(task2_acts_dict[k], axis=0)




#------------ SELECTIVITY AND VARIANCE CALCULATIONS -----------------
''' GET_SELECTIVITY_ARRAY *VERSION 1*
Selectivity: (max - avg)/(max + avg)

acts_dict: (int, int) -> float array - representing activations for each neuron for each
            attribute composition
choice_array: (int, int) array - list of attribute compositions
num_neurons: int - number of units in the layer being examined (length of float array values in acts_dict)
'''
def get_selectivity_array(acts_dict, choice_array, num_neurons=512):
    max_acts = np.zeros(num_neurons)
    max_lookup = np.zeros(num_neurons)
    selectivity_vals = np.zeros(num_neurons)

    for idx,k in enumerate(choice_array):
        max_lookup[acts_dict[k] > max_acts] = idx
        max_acts = np.maximum(max_acts, copy.deepcopy(acts_dict[k]))


    #selectivity = (max-avg)/(max+avg)
    for idx,k in enumerate(choice_array):
        update_idx = max_lookup != idx
        selectivity_vals[update_idx] += copy.deepcopy(acts_dict[k][update_idx])/(len(choice_array)-1)

    with np.errstate(divide='ignore', invalid='ignore'):
        selectivity_vals = (max_acts - selectivity_vals)/(max_acts + selectivity_vals)
        selectivity_vals[np.isnan(selectivity_vals)] = 0


    return selectivity_vals


''' GET_SELECTIVITY_ARRAY *VERSION 2*
Selectivity: (max - 2nd_max)/(max + 2nd_max)

acts_dict: (int, int) -> float array - representing avg activations for each neuron for each
            attribute composition
choice_array: (int, int) array - list of attribute compositions
num_neurons: int - number of units in the layer being examined (length of float array values in acts_dict)
'''
def get_selectivity_array2(acts_dict, choice_array, num_neurons=512):
    max_acts = np.zeros(num_neurons)
    max_lookup = np.zeros(num_neurons)
    selectivity_vals = np.zeros(num_neurons)

    for idx,k in enumerate(choice_array):
        max_lookup[acts_dict[k] > max_acts] = idx
        max_acts = np.maximum(max_acts, copy.deepcopy(acts_dict[k]))

    #calc selectivity
    for idx,k in enumerate(choice_array):
        update_idx = max_lookup != idx
        comparison = copy.deepcopy(acts_dict[k][update_idx])
        selectivity_vals[update_idx] = np.maximum(selectivity_vals[update_idx], comparison)

    with np.errstate(divide='ignore', invalid='ignore'):
        selectivity_vals = (max_acts - selectivity_vals)/(max_acts + selectivity_vals)
        selectivity_vals[np.isnan(selectivity_vals)] = 0

    return selectivity_vals


''' GET_SELECTIVITY_ARRAY *VERSION 3*
Selectivity: choose task1/task2. max determines which of the num_classes
    tasks to take min from. elt1 = min. elt2 = max of any combo not in
    the class being focused on

acts_dict: (int, int) -> float array - representing avg activations for each neuron for each
            attribute composition
choice_array: (int, int) array - list of attribute compositions
'''
def get_selectivity_array3(acts_dict, choice_array, num_neurons=512, task_idx=0, num_classes=None):
    max_acts = np.zeros(num_neurons)
    min_acts = np.ones(num_neurons)*9999
    max_lookup = np.zeros(num_neurons)
    selectivity_vals = np.zeros(num_neurons)

    for idx, k in enumerate(choice_array):
        max_lookup[acts_dict[k] > max_acts] = k[task_idx]
        max_acts = np.maximum(max_acts, copy.deepcopy(acts_dict[k]))

    for k in choice_array:
        update_idx = max_lookup == k[task_idx]
        min_acts[update_idx] = np.minimum(min_acts[update_idx], copy.deepcopy(acts_dict[k][update_idx]))

    for k in choice_array:
        update_idx = max_lookup != k[task_idx]
        comparison = copy.deepcopy(acts_dict[k][update_idx])
        selectivity_vals[update_idx] = np.maximum(selectivity_vals[update_idx], comparison)

    with np.errstate(divide='ignore',invalid='ignore'):
        selectivity_vals = (min_acts - selectivity_vals)/(min_acts + selectivity_vals)
        selectivity_vals[np.isnan(selectivity_vals)] = 0

    return selectivity_vals


''' PROTOTYPE_VARIANCE *Not used in final paper*
Prototype of variance:
    within group, select highest and lowest
    highest-lowest -- closer to 0 means less variance,
                      while closer to 1 means more
                      variance
'''
def prototype_variance(acts_dict, choice_array, num_neurons=512, task_idx=0):
    max_acts = np.zeros(num_neurons)
    min_acts = np.ones(num_neurons)*9999
    max_lookup = np.zeros(num_neurons)
    variance_vals = np.zeros(num_neurons)

    for idx, k in enumerate(choice_array):
        max_lookup[acts_dict[k] > max_acts] = k[task_idx]
        max_acts = np.maximum(max_acts, np.array(acts_dict[k]))

    for k in choice_array:
        update_idx = max_lookup == k[task_idx]
        min_acts[update_idx] = np.minimum(min_acts[update_idx], np.array(acts_dict[k][update_idx]))

    with np.errstate(divide='ignore', invalid='ignore'):
        variance_vals = (max_acts - min_acts)/(max_acts + min_acts)
        variance_vals[np.isnan(variance_vals)] = -2

    return variance_vals


''' GET_SELECTIVITY_TASK *Variant of get_selectivity_array3, but separates classes*

Selectivity in similar manner as get_selectivity_array3, but instead of choosing the selective class
via finding the composition that yields max firing rate, we manually choose the class (via the 'class_num' param).
'''
def get_selectivity_task(acts_dict, choice_array, class_num, task_idx=0, num_neurons=512, num_classes=10):
    class_lookup = np.ones(num_neurons)*class_num
    selectivity_vals = np.zeros(num_neurons)
    min_acts = np.ones(num_neurons)*9999

    #extract min within class
    for k in choice_array:
        update_idx = class_lookup == k[task_idx]
        min_acts[update_idx] = np.minimum(min_acts[update_idx], copy.deepcopy(acts_dict[k][update_idx]))


    for k in choice_array:
        update_idx = class_lookup != k[task_idx]
        comparison = copy.deepcopy(acts_dict[k][update_idx])
        selectivity_vals[update_idx] = np.maximum(selectivity_vals[update_idx], comparison)

    with np.errstate(divide='ignore',invalid='ignore'):
        selectivity_vals = (min_acts - selectivity_vals)/(min_acts + selectivity_vals)
        selectivity_vals[np.isnan(selectivity_vals)] = 0

    return selectivity_vals


# USE THIS FOR FINAL PLOTS
'''GET_SELECTIVITY_TASK_RELAXED *Veriant of get_selectivity_array_relaxed3, but separates classes*

Selectivity in similar manner as get_selectivity_array_relaxed3, but instead of choosing the selective class
via finding the composition that yields max firing rate, we manually choose the class (via the 'class_num' param).
This will be used for the final plots
'''
def get_selectivity_task_relaxed(acts_dict, choice_array, class_num, task_idx=0, num_neurons=512, num_classes=10):
    class_lookup = np.ones(num_neurons)*class_num
    selectivity_vals = np.zeros(num_neurons)

    avg_outside = np.zeros(num_neurons)

    #extract med within class
    med_list = [[] for i in range(num_neurons)]
    for k in choice_array:
        update_idx_inside = class_lookup == k[task_idx]
        update_idx_outside = class_lookup != k[task_idx]
        avg_outside[update_idx_outside] += acts_dict[k][update_idx_outside]

        for i,val in enumerate(update_idx_inside):
            if val:
                med_list[i].append(acts_dict[k][i])
    med_list = np.array(med_list)
    med_in = np.median(med_list, axis=1)
    avg_outside /= len(choice_array)-num_classes

    selectivity_vals = (med_in - avg_outside)/(med_in + avg_outside)
    return selectivity_vals





''' GET_SELECTIVITY_DICTS
Similar idea to update_acts_dicts, but using selectivity

TODO: Increase robustness of code, so won't need to attach more code just to add another task
    if we decide to try > 2 tasks at once
'''
#Selectivity_dicts -- layer-->task-->keep_pct
def get_selectivity_dicts(acts_dict_list, layer_names, keep_pcts, combinations, num_classes=10):
    num_neurons_list = []
    for acts_dict in acts_dict_list:
        num_neurons_list.append(acts_dict[int(100*keep_pcts[0])][combinations[0]].shape[1])

    selectivity_dicts = {}
    tasks = ['combos','task1','task2','task1_individual','task2_individual']
    for layer_idx,layer in enumerate(layer_names):
        selectivity_dicts[layer] = {}
        for task in tasks:
            selectivity_dicts[layer][task] = {}

        combo_acts_dict = {}
        task1_acts_dict = {}
        task2_acts_dict = {}

        for keep_pct in keep_pcts:
            keep_pct = int(100 * keep_pct)
            combo_acts_dict[keep_pct] = {}
            task1_acts_dict[keep_pct] = {}
            task2_acts_dict[keep_pct] = {}

            update_acts_dicts(combinations[:],acts_dict_list[layer_idx][keep_pct],
                              combo_acts_dict[keep_pct],
                              task1_acts_dict[keep_pct],
                              task2_acts_dict[keep_pct],
                              num_neurons=num_neurons_list[layer_idx])

            selectivity_dicts[layer]['combos'][keep_pct] = get_selectivity_array2(combo_acts_dict[keep_pct], combinations[:], num_neurons=num_neurons_list[layer_idx])
            selectivity_dicts[layer]['task1'][keep_pct] = get_selectivity_array3(combo_acts_dict[keep_pct], combinations[:], num_neurons=num_neurons_list[layer_idx], task_idx=0)
            selectivity_dicts[layer]['task2'][keep_pct] = get_selectivity_array3(combo_acts_dict[keep_pct], combinations[:], num_neurons=num_neurons_list[layer_idx], task_idx=1)
            selectivity_dicts[layer]['task1_individual'][keep_pct] = {}
            selectivity_dicts[layer]['task2_individual'][keep_pct] = {}

            #taskn_individual represents selectivity to a manually assigned attribute of task n
            for c in range(num_classes):
                if c not in selectivity_dicts[layer]['task1_individual'][keep_pct]:
                    selectivity_dicts[layer]['task1_individual'][keep_pct][c] = get_selectivity_task(combo_acts_dict[keep_pct],combinations[:],class_num=c,task_idx=0,num_neurons=num_neurons_list[layer_idx])
                    selectivity_dicts[layer]['task2_individual'][keep_pct][c] = get_selectivity_task(combo_acts_dict[keep_pct],combinations[:],class_num=c,task_idx=1,num_neurons=num_neurons_list[layer_idx])

    return selectivity_dicts

def get_selectivity_array_relaxed(acts_dict, choice_array, task_idx=0, num_neurons=512, num_classes=10):
    max_acts = np.zeros(num_neurons)
    second_max_acts = np.zeros(num_neurons)

    min_acts = np.ones(num_neurons)*9999
    second_min_acts = np.ones(num_neurons)*9999

    max_lookup = np.zeros(num_neurons)
    second_max_lookup = np.zeros(num_neurons)

    selectivity_vals = np.zeros(num_neurons)

    for idx, k in enumerate(choice_array):
        max_lookup[acts_dict[k] > max_acts] = k[task_idx]
        max_acts = np.maximum(max_acts, np.array(acts_dict[k]))

    for k in choice_array:
        update_idx = max_lookup == k[task_idx]
        min_acts[update_idx] = np.minimum(min_acts[update_idx], copy.deepcopy(acts_dict[k][update_idx]))

        second_update_idx = max_lookup != k[task_idx]
        second_max_lookup[second_update_idx][acts_dict[k][second_update_idx] > second_max_acts[second_update_idx]] = k[task_idx]
        second_max_acts[second_update_idx] = np.maximum(second_max_acts[second_update_idx], np.array(acts_dict[k][second_update_idx]))

    for k in choice_array:
        update_idx = second_max_lookup == k[task_idx]
        second_min_acts[update_idx] = np.minimum(second_min_acts[update_idx], np.array(acts_dict[k][update_idx]))

    for k in choice_array:
        statement1 = max_lookup != k[task_idx]
        statement2 = second_max_lookup != k[task_idx]
        update_idx = statement1 & statement2
        comparison = copy.deepcopy(acts_dict[k][update_idx])
        selectivity_vals[update_idx] = np.maximum(selectivity_vals[update_idx], comparison)

    with np.errstate(divide='ignore',invalid='ignore'):
        min_acts = np.maximum(min_acts, second_min_acts)
        selectivity_vals = (min_acts - selectivity_vals)/(min_acts + selectivity_vals)
        selectivity_vals[np.isnan(selectivity_vals)] = 0

    return selectivity_vals


def get_selectivity_array_relaxed2(acts_dict, choice_array, task_idx=0, num_neurons=512, num_classes=10):
    max_acts = np.zeros(num_neurons)
    min_acts = np.ones(num_neurons)*9999
    max_lookup = np.zeros(num_neurons)

    selectivity_vals = np.zeros(num_neurons)

    for idx, k in enumerate(choice_array):
        max_lookup[acts_dict[k] > max_acts] = k[task_idx]
        max_acts = np.maximum(max_acts, np.array(acts_dict[k]))

    for k in choice_array:
        update_idx = max_lookup == k[task_idx]
        min_acts[update_idx] = np.minimum(min_acts[update_idx], copy.deepcopy(acts_dict[k][update_idx]))


    num_combos_over = np.zeros(num_neurons)
    for k in choice_array:
        #criteria: increment value in num_combos_over by 1 for every combo with fire rate lower than min of selective attribute
        statement1 = max_lookup != k[task_idx]
        statement2 = min_acts > np.array(acts_dict[k])
        update_idx = statement1 & statement2
        if len(update_idx) > 0:
            num_combos_over[update_idx] += 1

    if len(choice_array[0]) == 2:
        num_combos_over = num_combos_over/(num_classes**2-num_classes)
    elif len(choice_array[0]) == 3:
        num_combos_over = num_combos_over/(num_classes**3-num_classes**2)

    return num_combos_over


# USE THIS FOR FINAL PLOTS
#selectivity = (med_in - avg_out)/(med_in + avg_out)
def get_selectivity_array_relaxed3(acts_dict, choice_array, task_idx=0, num_neurons=512, num_classes=10):
    max_acts = np.zeros(num_neurons)
    avg_inside = np.zeros(num_neurons)
    avg_outside = np.zeros(num_neurons)
    med_in = None
    max_lookup = np.zeros(num_neurons)
    smax = np.zeros(num_neurons)

    selectivity_vals = np.zeros(num_neurons)

    for idx, k in enumerate(choice_array):
        max_lookup[acts_dict[k] > max_acts] = k[task_idx]
        max_acts = np.maximum(max_acts, np.array(acts_dict[k]))
    '''
    for idx, k in enumerate(choice_array):
        update_idx_inside = max_lookup == k[task_idx]
        update_idx_outside = max_lookup != k[task_idx]
        avg_inside[update_idx_inside] += acts_dict[k][update_idx_inside]
        avg_outside[update_idx_outside] += acts_dict[k][update_idx_outside]
    avg_inside /= num_classes
    avg_outside /= len(choice_array)-num_classes

    selectivity_vals = (avg_inside - avg_outside)/(avg_inside + avg_outside)
    '''
    # for idx, k in enumerate(choice_array):
    #     update_idx = max_lookup != k[task_idx]
    #     smax[update_idx] = np.maximum(smax[update_idx],acts_dict[k][update_idx])


    med_list = []
    for i in range(num_neurons):
        med_list.append([])

    for idx, k in enumerate(choice_array):

        update_idx_inside = max_lookup == k[task_idx]
        update_idx_outside = max_lookup != k[task_idx]
        avg_outside[update_idx_outside] += acts_dict[k][update_idx_outside]

        for i,val in enumerate(update_idx_inside):
            if val:
                med_list[i].append(acts_dict[k][i])
    med_list = np.array(med_list)

#         med_in[,update_idx_inside] = acts_dict[k][update_idx_inside]

    med_in = np.median(med_list,axis=1)
    avg_outside /= len(choice_array)-num_classes



    selectivity_vals = (med_in - avg_outside)/(med_in + avg_outside)
    #selectivity_vals = (med_in - smax)/(med_in + smax)
    return selectivity_vals




# USE THIS FOR FINAL PLOTS
def get_selectivity_dicts2(acts_dict_list, layer_names, keep_pcts, combinations, num_classes=10):
    num_tasks = len(combinations[0])
    num_neurons_list = []
    for acts_dict in acts_dict_list:
        num_neurons_list.append(acts_dict[int(100*keep_pcts[0])][combinations[0]].shape[1])

    selectivity_dicts = {}
    #tasks = ['combos','task1','task2','task3','task1_individual','task2_individual','task3_individual','task_label']
    tasks = ['combos','task_label']
    for t in range(num_tasks):
        tasks += ['task'+str(t+1), 'task'+str(t+1)+'_individual']

    #init selectivity_dicts
    for layer_idx,layer in enumerate(layer_names):
        selectivity_dicts[layer] = {}
        for task in tasks:
            selectivity_dicts[layer][task] = {}

        combo_acts_dict = {}

        for keep_pct in keep_pcts:
            keep_pct = int(100 * keep_pct)
            combo_acts_dict[keep_pct] = {}

            for k in combinations[:]:
                combo_acts_dict[keep_pct][k] = np.mean(acts_dict_list[layer_idx][keep_pct][k],axis=0)

            selectivity_dicts[layer]['combos'][keep_pct] = get_selectivity_array2(combo_acts_dict[keep_pct], combinations[:], num_neurons=num_neurons_list[layer_idx])
            for t in range(num_tasks):
                selectivity_dicts[layer]['task'+str(t+1)][keep_pct] = get_selectivity_array_relaxed3(combo_acts_dict[keep_pct], combinations[:], num_neurons=num_neurons_list[layer_idx], task_idx=t, num_classes=num_classes)
                selectivity_dicts[layer]['task'+str(t+1)+'_individual'][keep_pct] = {}

            # #get task label
            # selectivity1 = np.array(selectivity_dicts[layer]['task1'][keep_pct])
            # selectivity2 = np.array(selectivity_dicts[layer]['task2'][keep_pct])
            # task2_idx = selectivity1 < selectivity2
            # if num_tasks == 3:
            #     selectivity3 = np.array(selectivity_dicts[layer]['task3'][keep_pct])
            #     task2_idx = (selectivity1 < selectivity2) & (selectivity2 > selectivity3)
            #     task3_idx = (selectivity3 > selectivity2) & (selectivity3 > selectivity1)
            select_list = [np.array(selectivity_dicts[layer]['task'+str(t+1)][keep_pct]) for t in range(num_tasks)]
            # task_idx_list = []
            # Fill in task_label: denotes the index of task with highest selectivity value for each neuron
            for t in range(num_tasks):
                task_idx = select_list[t] >= np.array(select_list)
                task_idx = np.all(task_idx,axis=0)
                # task_idx_list.append(task_idx)
                selectivity_dicts[layer]['task_label'][keep_pct] = np.zeros(select_list[t].shape)
                selectivity_dicts[layer]['task_label'][keep_pct][task_idx] = float(t)

            # selectivity_dicts[layer]['task_label'][keep_pct] = np.zeros(selectivity1.shape)
            # selectivity_dicts[layer]['task_label'][keep_pct][task2_idx] = 1.0
            # if num_tasks == 3:
            #     selectivity_dicts[layer]['task_label'][keep_pct][task3_idx] = 2.0


            #taskn_individual represents selectivity to a manually assigned attribute of task n
            for c in range(num_classes):
                if c not in selectivity_dicts[layer]['task1_individual'][keep_pct]:
                    # selectivity_dicts[layer]['task1_individual'][keep_pct][c] = get_selectivity_task_relaxed(combo_acts_dict[keep_pct],combinations[:],class_num=c,task_idx=0,num_neurons=num_neurons_list[layer_idx])
                    # selectivity_dicts[layer]['task2_individual'][keep_pct][c] = get_selectivity_task_relaxed(combo_acts_dict[keep_pct],combinations[:],class_num=c,task_idx=1,num_neurons=num_neurons_list[layer_idx])
                    # if num_tasks == 3:
                    #     selectivity_dicts[layer]['task3_individual'][keep_pct][c] = get_selectivity_task_relaxed(combo_acts_dict[keep_pct],combinations[:],class_num=c,task_idx=2,num_neurons=num_neurons_list[layer_idx])
                    for t in range(num_tasks):
                        selectivity_dicts[layer]['task'+str(t+1)+'_individual'][keep_pct][c] = get_selectivity_task_relaxed(combo_acts_dict[keep_pct],combinations[:],class_num=c,task_idx=t,num_neurons=num_neurons_list[layer_idx])

    return selectivity_dicts



#----------------------- MISC. HELPER CODE -----------------------
def assign_val(filepath):
    variable = None
    if os.path.exists(filepath):
        variable = np.load(filepath)
    return variable

def get_acts_matrix(combinations, acts_dict, num_neurons=512):
    acts_mat = np.zeros((len(combinations), num_neurons))
    for idx, k in enumerate(combinations):
        acts_mat[idx] = copy.deepcopy(acts_dict[k])

    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_acts_mat = acts_mat/np.max(acts_mat, axis=0)
        normalized_acts_mat = normalized_acts_mat[:,~np.isnan(normalized_acts_mat).any(axis=0)]

    return normalized_acts_mat

def get_med_and_std(array):
    med = np.median(array)
    std = np.std(array)
    return (med, std)

def make_first_upper(string):
    if len(string) < 1:
        return string
    elif len(string) == 1:
        return string.upper()
    return string[0].upper() + string[1:]

'''
find 1% top value for each neuron and return [1,num_neurons] array

activations_dict: combination --> [num_examples,num_neurons] array
'''
def get_percentile_array(activations_dict, percentile_threshold=1):
    combinations = [k for k in activations_dict]
    num_neurons = activations_dict[combinations[0]].shape[1]

    full_acts = np.array(activations_dict[combinations[0]])
    #Get full [10000, num_neurons] activation array
    for combination in combinations[1:]:
        acts = np.array(activations_dict[combination])
        full_acts = np.concatenate([full_acts, acts], axis=0)

    #Get value of 1% highest activation. Use as threshold
    percentile_array = np.percentile(full_acts, 100-percentile_threshold, axis=0)
    percentile_array = np.reshape(percentile_array, [1,num_neurons])
    return percentile_array


'''
Find fraction of num_examples in which neuron passes activation
threshold (Dependent on top 1% of activations)
'''
def get_acts_thresholding(activations_dict, percentile_array):
    combinations = [k for k in activations_dict]
    num_neurons = activations_dict[combinations[0]].shape[1]
    threshold_acts_dict = {}

    for combination in combinations:
        acts = np.array(activations_dict[combination])
        num_examples = acts.shape[0]
        threshold_percentages = (acts >= np.tile(percentile_array, [num_examples,1])).astype(float)
        threshold_percentages = np.mean(threshold_percentages,axis=0)

        threshold_acts_dict[combination] = threshold_percentages

    return threshold_acts_dict



#----------------------- CODE FOR RUNNING ABLATION EXPERIMENTS -----------------------

def get_accuracy_ablation(model_name, dataset_name, keep_pct, ablated_neurons_list=[], target_combos = [], num_classes=10, early_stop=-1, verbose=True, layer_name="",only=""):
    data_directory = "../analysis/test_datasets/" + dataset_name + "/"
    data = pickle.load(open(data_directory + "data.p","rb"))
    targets = pickle.load(open(data_directory + "targets.p","rb"))

    loader = utils.DataLoader(utils.TensorDataset(data, targets), batch_size=1)

    state_dict_directory = "../analysis/state_dicts/" + dataset_name + "/" + model_name + "/"
    if model_name in ["resnet_pretrained","resnet_pretrained_embeddings"]:
        if basename(os.getcwd()) == "util":
            os.chdir("../")
        model = config.model.options[model_name](num_classes)
        if basename(os.getcwd()) == "experiment-framework":
            os.chdir("util/")
    else:
        model = config.model.options[model_name](num_classes)

    if only in ["_color_only","_shape_only","_location_only"]:
        only = "keep_pct_readout"+only+"/"
        keep_pct = int(10*float(keep_pct))


    state_dict = torch.load(state_dict_directory + only + str(keep_pct) + ".pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    zero_out = ablated_neurons_list

    if model_name != "simple_cnn":
        neurons = 512
        if layer_name == '_layer3':
            neurons = 256
        elif layer_name == '_layer1':
            neurons = 64
    else:
        neurons = 500

    mask = np.ones(neurons)
    mask[zero_out] = 0
    if model_name in ["simple_cnn","resnet_no_pool"]:
        mask_shape = (1,neurons)
    else:
        mask_shape = (1,neurons,1,1)
    mask = torch.FloatTensor(np.reshape(mask, mask_shape))

    total = 0
    iteration_count = 0
    num_correct_count = 0
    col_correct_count = 0
    with torch.no_grad():
        num_examples = len(loader)
        for data, target in loader:
            iteration_count += 1
            if iteration_count % 100 == 0 and verbose:
                print(iteration_count/num_examples)
                print("Fraction correct: ",num_correct_count/max(total,1), col_correct_count/max(total,1))
                print("total:", total)

            digit_class = target[0][0].item()
            color_class = target[0][1].item()
            combo = (digit_class, color_class)
            if len(target_combos) > 0 and combo not in target_combos:
                continue
            else:
                total += 1

            if model_name != "simple_cnn":
                out = F.relu(model.bn1(model.conv1(data)))
                out = model.layer1(out)
                if len(mask[0]) == 64:
                    out = out*mask
                out = model.layer2(out)
                if len(mask[0]) == 128:
                    out = out*mask
                out = model.layer3(out)
                if len(mask[0]) == 256:
                    out = out*mask
                out = model.layer4(out)
                if model_name != "resnet_no_pool":
                    out = F.avg_pool2d(out, 4)
                else:
                    out = out.view(out.size(0),-1)
                    out = model.pool_fc(out)

                if len(mask[0]) == 512:
                    out = out*mask
                out = out.view(out.size(0), -1)

            else:
                out = F.relu(model.conv1(data))
                out = F.max_pool2d(out,2,2)
                out = F.relu(model.conv2(out))
                out = F.max_pool2d(out,2,2)
                out = out.view(-1, 5*5*50)
                out = F.relu(model.fc1(out))
                out = out*mask


            num_output = model.fc2_number(out)
            col_output = model.fc2_color(out)

            pred = torch.cat((num_output.argmax(dim=1, keepdim=True), col_output.argmax(dim=1, keepdim=True)), 1)
            num_correct, col_correct = pred.eq(target.view_as(pred))[:, 0], pred.eq(target.view_as(pred))[:, 1]
            num_correct_count += num_correct.sum().item()
            col_correct_count += col_correct.sum().item()
            if early_stop >=0 and iteration_count >= early_stop:
                break

        if verbose:
            print("task 1 acc: ", 100. * num_correct_count / total)
            print("task 2 acc: ", 100. * col_correct_count / total)

    print("RESULTS: " + str(num_correct_count / total), str(col_correct_count / total))
    return num_correct_count / total, col_correct_count / total
