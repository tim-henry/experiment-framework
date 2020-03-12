import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as utils
import numpy as np
import os
import sys
import argparse
import config.model


# ---------------------------- Process input -----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default=None,help="Choose dataset name ('left_out_colored_mnist', 'left_out_varied_location_mnist', 'left_out_many_scale_mnist', or 'left_out_colored_location_mnist')")
parser.add_argument("--keep_pct",default=None,help="Choose keep percentage (decimal representation of % compositions seen, or 'all' for iterating through each %). Corresponds to the names found in state_dicts directory")
parser.add_argument("--network",default=None,help="Choose network name (resnet, resnet_no_pool, resnet_pretrained, resnet_pretrained_embeddings, simple_cnn, resnet_early_branching)")
parser.add_argument("--only",default=None,help="For networks trained only on a single task. Choose between color only ('color'), shape only ('shape'), location only ('location'), or scale only ('scale'). Otherwise, ignore this option.")
parser.add_argument("--batch_size",default="1",help="Choose batch size. For early branching model, defaults to 2")
parser.add_argument("--state_dicts_path",default=None,help="Option to manually choose the path to the directory of the trained network's state_dicts. Otherwise, ignore this option for the default path")
parser.add_argument("--dataset_path",default=None,help="Option to manually choose the path to the directory of the dataset you want to load. Otherwise, ignore this option for the default path")
parser.add_argument("--num_classes",default='9',help="Option to manually choose the number of classes for each task in the dataset. Ignore if dataset is already mnist color/location/scale")
parser.add_argument("--num_tasks",default='2',help="Option to manually choose the number of tasks for the given dataset. Ignore if dataset is already mnist color/location/scale")

args = parser.parse_args()

keep_pct = args.keep_pct
input_dataset_path = args.dataset_path
input_state_dicts_path = args.state_dicts_path


#Enforce specific format in which input directory names end with a '/'
if input_dataset_path is not None and input_dataset_path[-1] != '/':
    input_dataset_path += '/'

if input_state_dicts_path is not None and input_state_dicts_path[-1] != '/':
    input_state_dicts_path += '/'


# Make string inputs case insensitive
dataset_name = args.dataset
if dataset_name is not None:
    dataset_name = dataset_name.lower()

model_name = args.network
if model_name is not None:
    model_name = model_name.lower()

only = args.only
if only is not None:
    only = only.lower()


# Catch several input variations for dataset_name
if dataset_name is None or dataset_name in ['left_out_colored_mnist','color','col','c']:
    dataset_name = "left_out_colored_mnist"
elif dataset_name in ['left_out_varied_location_mnist','location','loc','l']:
    dataset_name = "left_out_varied_location_mnist"
elif dataset_name in ['left_out_many_scale_mnist','scale','scale_mnist']:
    dataset_name = "left_out_many_scale_mnist"
elif dataset_name in ['three', 'three_task', 'left_out_colored_location_mnist', 'color-location-scale', '3task', '3_task', 'left_out_colored_location', 'colored_location']:
    dataset_name = "left_out_colored_location_mnist"


# Catch several input variations for model_name
if model_name is None or model_name in ['resnet','res']:
    model_name = 'resnet'
elif model_name in ['resnet_pretrained','pretrained','pretraining','resnet_pretraining']:
    model_name = 'resnet_pretrained'
elif model_name in ['resnet_pretrained_embeddings','resnet_pretrained_embed','pretrained_embeddings','pretrained_embed','embeddings','embed','embedded']:
    model_name = 'resnet_pretrained_embeddings'
elif model_name in ['resnet_no_pool','no_pool']:
    model_name = 'resnet_no_pool'
elif model_name in ['simple_cnn','simple']:
    model_name = 'simple_cnn'
elif model_name in ['resnet_early_branching','resnet_early','res_early_branch','res_early','early']:
    model_name = "resnet_early_branching"


# Catch several input variations for keep_pct value
keep_list_10_classes = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
keep_values_10_classes = [10,20,30,40,50,60,70,80,90,.1,.2,.3,.4,.5,.6,.7,.8,.9]
keep_list_9_classes=[0.1111111111111111,0.2222222222222222,0.3333333333333333,0.4444444444444444,
                    0.5555555555555556,0.6666666666666666,0.7777777777777778,0.8888888888888888]
keep_values_9_classes = [11,22,33,44,55,66,77,88,.11,.22,.33,.44,.55,.66,.77,.88,.56,.67,.78,.89]

if keep_pct == 'all':
    if dataset_name == "left_out_colored_mnist":
        keep_pct = keep_list_10_classes
    else:
        keep_pct = keep_list_9_classes
elif keep_pct is None:
    if dataset_name == "left_out_colored_mnist":
        keep_pct = 0.9
    else:
        keep_pct = 0.8888888888888888
elif round(float(keep_pct),2) in keep_values_9_classes:
    possible = keep_list_9_classes + keep_list_9_classes + [0.5555555555555556,0.6666666666666666,0.7777777777777778,0.8888888888888888]
    keep_pct = possible[keep_values_9_classes.index(round(float(keep_pct),2))]
elif float(keep_pct) in keep_values_10_classes:
    possible = keep_list_10_classes + keep_list_10_classes
    keep_pct = possible[keep_values_10_classes.index(float(keep_pct))]



# Progress Printing toggle
show_progress = True


# Set num_classes and num_tasks based on dataset

num_classes = tuple([int(args.num_classes) for i in range(int(args.num_tasks))])
if dataset_name in ["left_out_colored_mnist"]:
    num_classes = (10,10)
elif dataset_name in ["left_out_colored_location"]:
    num_classes = (9,9,9)

num_tasks = len(num_classes)


batch_size = int(args.batch_size)
if model_name in ["resnet_early_branching"] and batch_size==1:
    batch_size = 2



#------------------------ Print and save/load helper functions -----------------
def store_activations(dataset_name, model_name, keep_pct, data, only=None, layer_tag=''):
    if only == 'color':
        kind = '_color_only'
    elif only == 'shape':
        kind = '_shape_only'
    elif only == 'location':
        kind = '_location_only'
    else:
        kind = ''

    filename = '{}_{}_keep{}{}.pkl'.format(model_name, dataset_name + layer_tag, int(100*keep_pct), kind)
    file_path = 'analysis/'+model_name+'/activations_data/'+dataset_name+kind+'/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    print("Storing "+filename)
    with open(file_path + filename, "wb") as f:
        pickle.dump(data, f)


def print_progress(current_iteration, tot_iterations, correct_count_list):
    progress = current_iteration/tot_iterations * 100
    acc_list = [round(100*correct_count_list[i]/(current_iteration*batch_size),2) for i in range(len(correct_count_list))]
    out = " Progress: {}%  ".format(round(progress,1))
    for i in range(len(acc_list)):
        out += ' task' + str(i+1) + ' acc: ' + str(acc_list[i]) + '%  '
    out += "\r"
    sys.stdout.write(out)
    sys.stdout.flush()


def add_to_act_dict(out_tensor, act_dict, act_key):
    if act_key in act_dict:
        act_dict[act_key] = np.concatenate([act_dict[act_key], out_tensor],axis=0)
    else:
        act_dict[act_key] = np.array(out_tensor)



#----------------------------- Main Function -----------------------------------
def main(dataset_name, keep_pct, model_name, only):
    #----------------------------- Init/Load -----------------------------------
    # Get dataset
    if input_dataset_path is None:
        data_directory = "analysis/test_datasets/" + dataset_name + "/"
    else:
        data_directory = input_dataset_path

    data = pickle.load(open(data_directory + "data.p","rb"))
    targets = pickle.load(open(data_directory + "targets.p","rb"))
    print("loaded data")


    # Init Loader
    loader = utils.DataLoader(utils.TensorDataset(data, targets), batch_size=batch_size)

    # Format naming for path to state_dicts to account for differences in numbering conventions
    if dataset_name in ["left_out_many_scale_mnist","left_out_colored_location_mnist"] or only is not None or model_name in ["resnet_early_branching"]:
        keep_pct_str = str(int(keep_pct*10))
    else:
        keep_pct_str = str(keep_pct)

    # Account for differences in naming of color/shape/location only state_dicts.
    # No need if dir path already specified in input.
    if input_state_dicts_path is None:
        state_dict_directory = "analysis/state_dicts/" + dataset_name + "/" + model_name + "/"
        if only == 'color':
            loadname = state_dict_directory + 'keep_pct_readout_color_only/' + keep_pct_str + ".pt"
        elif only == 'shape':
            loadname = state_dict_directory + 'keep_pct_readout_shape_only/' + keep_pct_str + ".pt"
        elif only == 'location':
            loadname = state_dict_directory + 'keep_pct_readout_location_only/' + keep_pct_str + ".pt"
        else:
            loadname = state_dict_directory + keep_pct_str + ".pt"
    else:
        loadname = input_state_dicts_path + keep_pct_str + ".pt"


    #Load state dicts. Currently set to map to cpu. can change to use gpu if available
    state_dict = torch.load(loadname, map_location=torch.device('cpu'))
    model = config.model.options[model_name](num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    print("loaded state dict")



    #-------------------------- Main Loop --------------------------------------
    total = 0
    iteration_count = 0
    correct_count_list = [0 for i in range(num_tasks)]

    with torch.no_grad():
        activations_dict = {}
        activations_dict_layer1 = {}
        activations_dict_layer3 = {}
        if model_name in ['resnet_early_branching']:
            activations_dict = [{} for i in range(num_tasks)]
            activations_dict_layer3 = [{} for i in range(num_tasks)]
        num_examples = len(loader)


        # target -> digit, color/location/scale (unless 3 tasks, then digit, color, location)
        for data, target in loader:
            iteration_count += 1
            if show_progress and iteration_count % 10 == 0:
                print_progress(iteration_count, num_examples, correct_count_list)


            #Get act keys for all data points in current batch.
            #act keys are num_tasks-tuple ints representing the particular GT labels for each corresponding task.
            act_key_list = []
            for batch_idx in range(batch_size):
                act_key_list.append(tuple([target[batch_idx][i].item() for i in range(num_tasks)]))


            total += batch_size

            if model_name in ["resnet","resnet_pretrained","resnet_pretrained_embeddings","resnet_no_pool"]:
                #----------- Late branching Accuracy (RESNET variations)--------
                outputs = model(data)
                pred = torch.cat( [out.argmax(dim=1, keepdim=True) for out in outputs], 1)
                correct_list = [pred.eq(target.view_as(pred))[:, i] for i in range(num_tasks)]

                for i in range(num_tasks):
                    correct_count_list[i] += correct_list[i].sum().item()

                #--------------- EXTRACTING ACTIVATIONS ------------------------
                out = F.relu(model.bn1(model.conv1(data)))
                out = model.layer1(out)

                for i in range(batch_size):
                    out_layer1 = np.reshape(np.mean(np.array(out[i].view(out.size(1),-1)),axis=1),(1,out.size(1)))
                    add_to_act_dict(out_layer1, activations_dict_layer1, act_key_list[i])

                out = model.layer2(out)
                out = model.layer3(out)

                for i in range(batch_size):
                    out_layer3 = np.reshape(np.mean(np.array(out[i].view(out.size(1),-1)),axis=1),(1,out.size(1)))
                    add_to_act_dict(out_layer3, activations_dict_layer3, act_key_list[i])

                out = model.layer4(out)

                if model_name != "resnet_no_pool":
                    out = F.avg_pool2d(out, 4)
                else:
                    out = out.view(out.size(0),-1)
                    out = model.pool_fc(out)
                    out = F.relu(out)
                out = out.view(out.size(0), -1)


                # out should now be a [1, 512] pytorch tensor containing
                # the activations of the network before the output layers
                for i in range(batch_size):
                    fin = np.reshape(out[i], (1,out.shape[-1]))
                    add_to_act_dict(fin, activations_dict, act_key_list[i])


            elif model_name in ['simple_cnn']:
                #---------- Late Branching Accuracy (SimpleCNN) ----------------
                num_output, col_output = model(data)
                pred = torch.cat((num_output.argmax(dim=1, keepdim=True), col_output.argmax(dim=1, keepdim=True)), 1)
                num_correct, col_correct = pred.eq(target.view_as(pred))[:, 0], pred.eq(target.view_as(pred))[:, 1]
                num_correct_count += num_correct.sum().item()
                col_correct_count += col_correct.sum().item()

                #----------- Extract Activations -------------------------------
                out = F.relu(model.conv1(data))
                out = F.max_pool2d(out, 2, 2)
                out = F.relu(model.conv2(out))
                out = F.max_pool2d(out, 2, 2)
                out = out.view(-1,5*5*50)
                out = F.relu(model.fc1(out))

                if batch_size > 1:
                    for i in range(batch_size):
                        fin = np.reshape(out[i],(1,out.shape[-1]))
                        add_to_act_dict(out, activations_dict, act_key_list[i])
                else:
                    add_to_act_dict(out, activations_dict, act_key_list[i])


            elif model_name in ['resnet_early_branching']:
                # Sort of messy addition. To account for branching, act_dicts are
                # saved in list. Each list index represents the corresponding branch number.
                #----------- Early branching Accuracy (RESNET) -----------------
                outputs = model(data)
                pred = torch.cat( [out.argmax(dim=1, keepdim=True) for out in outputs], 1)
                correct_list = [pred.eq(target.view_as(pred))[:, i] for i in range(num_tasks)]

                for i in range(num_tasks):
                    correct_count_list[i] += correct_list[i].sum().item()


                out = F.relu(model.bn1(model.conv1(data)))
                out = model.layer1(out)
                for i in range(batch_size):
                    #in case batch_size > 1, iterate through all items in batch, reshape as necessary, then add to dict
                    out_layer1 = np.reshape(np.mean(np.array(out[i].view(out.size(1),-1)),axis=1),(1,out.size(1)))
                    add_to_act_dict(out_layer1, activations_dict_layer1, act_key_list[i])

                out = model.layer2(out)
                out1 = model.layer3_1(out)
                for i in range(batch_size):
                    out_layer3 = np.reshape(np.mean(np.array(out1[i].view(out1.size(1),-1)),axis=1),(1,out1.size(1)))
                    add_to_act_dict(out_layer3, activations_dict_layer3[0], act_key_list[i])


                out1 = model.layer4_1(out1)
                if model.pool:
                    out1 = F.avg_pool2d(out1, 4)
                else:
                    out1 = out1.view(out1.size(0), -1)
                    out1 = model.pool_fc_1(out1)
                    out1 = F.relu(out1)
                out1 = out1.view(out1.size(0), -1)
                for i in range(batch_size):
                    fin = np.reshape(out1[i], (1,out1.shape[-1]))
                    add_to_act_dict(fin, activations_dict[0], act_key_list[i])


                #2+ branches
                if model.num_branches > 1:
                    out2 = model.layer3_2(out)
                    for i in range(batch_size):
                        out_layer3 = np.reshape(np.mean(np.array(out2[i].view(out2.size(1),-1)),axis=1),(1,out2.size(1)))
                        add_to_act_dict(out_layer3, activations_dict_layer3[1], act_key_list[i])

                    out2 = model.layer4_2(out2)
                    if model.pool:
                        out2 = F.avg_pool2d(out2,4)
                    else:
                        out2 = out2.view(out2.size(0),-1)
                        out2 = model.pool_fc_2(out2)
                        out2 = F.relu(out2)
                    out2 = out2.view(out2.size(0), -1)
                    for i in range(batch_size):
                        fin = np.reshape(out2[i], (1,out2.shape[-1]))
                        add_to_act_dict(fin, activations_dict[1], act_key_list[i])

                #3+ branches
                if model.num_branches > 2:
                    out3 = model.layer3_3(out)
                    for i in range(batch_size):
                        out_layer3 = np.reshape(np.mean(np.array(out3[i].view(out3.size(1),-1)),axis=1),(1,out3.size(1)))
                        add_to_act_dict(out_layer3, activations_dict_layer3[2], act_key_list[i])

                    out3 = model.layer4_3(out3)
                    if model.pool:
                        out3 = F.avg_pool2d(out3)
                    else:
                        out3 = out3.view(out3.size(0), -1)
                        out3 = model.pool_fc_3(out3)
                        out3 = F.relu(out3)
                    out3 = out3.view(out3.size(0), -1)
                    for i in range(batch_size):
                        fin = np.reshape(out3[i], (1,out3.shape[-1]))
                        add_to_act_dict(fin, activations_dict[2], act_key_list[i])

                #4 branches
                if model.num_branches == 4:
                    out4 = model.layer3_4(out)
                    for i in range(batch_size):
                        out_layer3 = np.reshape(np.mean(np.array(out4[i].view(out4.size(1),-1)),axis=1),(1,out4.size(1)))
                        add_to_act_dict(out_layer3, activations_dict_layer3[3], act_key_list[i])

                    out4 = model.layer4_4(out4)
                    if model.pool:
                        out4 = F.avg_pool2d(out4)
                    else:
                        out4 = out4.view(out4.size(0),-1)
                        out4 = model.pool_fc_4(out4)
                        out4 = F.relu(out4)
                    out4 = out4.view(out4.size(0), -1)
                    for i in range(batch_size):
                        fin = np.reshape(out4[i], (1,out4.shape[-1]))
                        add_to_act_dict(fin, activations_dict[3], act_key_list[i])


        store_activations(dataset_name, model_name, keep_pct, activations_dict, only)
        if model_name != "simple_cnn":
            store_activations(dataset_name, model_name, keep_pct, activations_dict_layer1, only, layer_tag='_layer1')
            store_activations(dataset_name, model_name, keep_pct, activations_dict_layer3, only, layer_tag='_layer3')



if __name__ == '__main__':
    if type(keep_pct) == type([]):
        for kp in keep_pct:
            main(dataset_name, kp, model_name, only)
    else:
        main(dataset_name, keep_pct, model_name, only)
