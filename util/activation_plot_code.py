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


'''
For 10x10 individual neuron activation plots

Plot num_classes x num_classes heat map for each neuron in [neuron_idx_start, neuron_idx_stop]

combo_matrix: array of matrices representing num_classes x num_classes heat maps for EVERY neuron (Create full array before running method)
normalize: Individual -> based on individual neurons, group -> based on all neurons
'''
def plot_combination_acts(combo_matrix, neuron_idx_start, neuron_idx_end,
                          normalize='individual', num_classes=10,
                          plot_single_column=True, dataset_name=None,
                          plot_size=6):

    if not plot_single_column: # Plot in groups of 2
        col_size = 2
        num_plots = neuron_idx_end - neuron_idx_start
        plot_IDs = [i for i in range(neuron_idx_start, neuron_idx_end)]
        fig, ax = plt.subplots(num_plots//col_size + (num_plots%col_size), col_size, sharey=False)
        #fig.text(.5,0.04,'color', va='center')
        #fig.text(0.04,.5,'shape', ha='center',rotation='vertical')

        for idx,i in enumerate(plot_IDs):
            if np.max(combo_matrix[i]) == 0:
                normalized_matrix = combo_matrix[i]
                if num_plots > col_size:
                    subplot_ax = ax[(idx)//col_size][idx%col_size]
                else:
                    subplot_ax = ax[idx]
            else:
                if normalize == 'group':
                    normalized_matrix = combo_matrix[i] / np.max(combo_matrix[neuron_idx_start: neuron_idx_end])
                elif normalize == 'individual':
                    normalized_matrix = combo_matrix[i] / np.max(combo_matrix[i])
                else:
                    normalized_matrix = combo_matrix[i][:]

                if num_plots > col_size:
                    subplot_ax = ax[(idx)//col_size][idx%col_size]
                else:
                    subplot_ax = ax[idx]

            subplot_ax.set_xticks(np.arange(num_classes))
            subplot_ax.set_yticks(np.arange(num_classes))

            num_rows = len(combo_matrix[i])
            num_cols = len(combo_matrix[i][0])
            for a in range(num_rows):
                for b in range(num_cols):
                    text = subplot_ax.text(b, a, round(normalized_matrix[a,b],2),
                                  ha = 'center', va='center', color='w',size=7)
            subplot_ax.title.set_text("neuron "+str(i))

            subplot_ax.imshow(normalized_matrix, aspect = 'auto')


            fig.set_size_inches(4*col_size,2*num_plots)

    else: #Just plot each individual 10x10 one after another
        for i in range(neuron_idx_start, neuron_idx_end):
            if np.max(combo_matrix[i]) == 0:
                normalized_matrix = combo_matrix[i]
                if num_plots > col_size:
                    subplot_ax = ax[(idx)//col_size][idx%col_size]

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

            fig.set_size_inches(plot_size,plot_size)
            plt.show()

    plt.show()

'''
Plot activation matrix

x-axis -> neuron indices
y-axis -> comninations
'''
def plot_act_matrix(act_matrix, combinations_list, n_clusters=1):
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

        plt.xlabel("cluster")
    else:
        plt.xlabel("neuron index")

    plt.ylabel("combination")

    fig.set_size_inches(25,15)
    im = ax.imshow(out, aspect='auto')

    title = "Activations Plot for Combinations"

    plt.title(title)
    ax.set_yticks([i for i in range(len(combinations_list))])
    ax.set_yticklabels([str(c) for c in combinations_list], fontsize=11)

    plt.show()


'''
plot table of accuracies. rows -> class being ablated, cols -> class whose accuracies are being evaluated
(fill out data before passing)
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

def update_acts_dicts(combo_list, acts_dict, combo_acts_dict, task1_acts_dict, task2_acts_dict, num_neurons=512):
    for idx,k in enumerate(combo_list):
        combo_acts_dict[k] = np.mean(acts_dict[k],axis=0)
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


'''
Selectivity: (max - avg)/(max + avg)
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


'''
Selectivity: (max - 2nd_max)/(max + 2nd_max)
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


'''
Selectivity: choose task1/task2. max determines which of the num_classes
    tasks to take min from. elt1 = min. elt2 = max of any combo not in
    the class being focused on
'''
def get_selectivity_array3(acts_dict, choice_array, num_neurons=512, task_idx=0):
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
        selectivity_vals[np.isnan(selectivity_vals)] = -2

    return selectivity_vals

'''
Selectivity in similar manner as get_selectivity_array3, but for specific task instead of max
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
        selectivity_vals[np.isnan(selectivity_vals)] = -2

    return selectivity_vals



def get_accuracy_ablation(model_name, dataset_name, keep_pct, ablated_neurons_list=[], target_combos = [], num_classes=10, early_stop=-1, verbose=True):
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

    state_dict = torch.load(state_dict_directory + str(keep_pct) + ".pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    zero_out = ablated_neurons_list

    if model_name != "simple_cnn":
        neurons = 512
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
                out = model.layer2(out)
                out = model.layer3(out)
                out = model.layer4(out)
                if model_name != "resnet_no_pool":
                    out = F.avg_pool2d(out, 4)
                else:
                    out = out.view(out.size(0),-1)
                    out = model.pool_fc(out)

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
