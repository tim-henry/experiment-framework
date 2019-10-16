import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as utils
import numpy as np
import os
import sys

import config.model

dataset_name = "left_out_colored_mnist"
hold_out = False
num_classes = 10
model_name = "resnet_pretrained"
keep_pct = 1
show_progress = True

def store_activations(dataset_name, model_name, data):
    if hold_out:
        dataset_kind = "_hold_out"
    else:
        dataset_kind = ""

    filename = '{}_{}_keep{}{}.pkl'.format(model_name, dataset_name, int(100*keep_pct), dataset_kind)
    file_path = 'analysis/'+model_name+'/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    print("Storing "+filename)
    with open(file_path + filename, "wb") as f:
        pickle.dump(data, f)

def print_progress(current_iteration, tot_iterations):
    progress = current_iteration/tot_iterations * 100
    sys.stdout.write( " Progress: %d%%    \r" % (progress))
    sys.stdout.flush()

def restore_activations(path):
    with open(path,'rb') as f:
        act = pickle.load(f)
        for k in act:
            print(len(act[k]))
        print("loaded")


def main():

    # Get held-out/full dataset
    data_directory = "analysis/test_datasets/" + dataset_name + "/"
    if hold_out:
        data = pickle.load(open(data_directory + "hold_out_data.p", "rb"))
        targets = pickle.load(open(data_directory + "hold_out_targets.p", "rb"))
        print("loaded data (holdout)")
    else:
        data = pickle.load(open(data_directory + "data.p","rb"))
        targets = pickle.load(open(data_directory + "targets.p","rb"))
        print("loaded data (no holdout)")

    loader = utils.DataLoader(utils.TensorDataset(data, targets), batch_size=1)

    # Get model function
    state_dict_directory = "analysis/state_dicts/" + dataset_name + "/" + model_name + "/"
    model = config.model.options[model_name](num_classes)
    state_dict = torch.load(state_dict_directory + str(keep_pct) + ".pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # Get activations
    iteration_count = 0
    num_correct_count = 0
    col_correct_count = 0

    with torch.no_grad():
        activations_dict = {}
        num_examples = len(loader)

        # target -> digit, color
        for data, target in loader:
            iteration_count += 1
            if show_progress and iteration_count % 100 == 0:
                print_progress(iteration_count, num_examples)

            digit_class = target[0][0].item()
            color_class = target[0][1].item()
            act_key = (digit_class, color_class)
            if model_name in ["resnet","resnet_pretrained","resnet_pretrained_embeddings"]:
                #Accuracy
                num_output, col_output = model(data)
                pred = torch.cat((num_output.argmax(dim=1, keepdim=True), col_output.argmax(dim=1, keepdim=True)), 1)
                num_correct, col_correct = pred.eq(target.view_as(pred))[:, 0], pred.eq(target.view_as(pred))[:, 1]
                num_correct_count += num_correct.sum().item()
                col_correct_count += col_correct.sum().item()

                #Activations
                out = F.relu(model.bn1(model.conv1(data)))
                out = model.layer1(out)
                out = model.layer2(out)
                out = model.layer3(out)
                out = model.layer4(out)
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)
                # out should now be a [1, 512] pytorch tensor containing
                # the activations of the network before the output layers
                if act_key in activations_dict:
                    activations_dict[act_key] = np.concatenate([activations_dict[act_key],out],axis=0)
                else:
                    activations_dict[act_key] = np.array(out)

        print("task 1 acc: ", 100. * num_correct_count / len(loader.dataset))
        print("task 2 acc: ", 100. * col_correct_count / len(loader.dataset))

        store_activations(dataset_name, model_name, activations_dict)



if __name__ == '__main__':
    main()
