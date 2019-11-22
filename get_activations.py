import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as utils
import numpy as np
import os
import sys
import argparse

import config.model

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default=None,help="Choose dataset type ('color' or 'location')")
parser.add_argument("--keep_pct",default=None,help="Choose keep percentage (decimal representation of % combinations seen, or 'all' for iterating through each %)")
parser.add_argument("--network",default=None,help="Choose network (resnet, resnet_no_pool, resnet_pretrained, resnet_pretrained_embeddings, simple_cnn)")

args = parser.parse_args()

dataset_name = args.dataset
keep_pct = args.keep_pct
model_name = args.network

color_keep_list = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
possible_color_keep_values = [10,20,30,40,50,60,70,80,90,.1,.2,.3,.4,.5,.6,.7,.8,.9]

location_keep_list=[0.1111111111111111,0.2222222222222222,0.3333333333333333,0.4444444444444444,
                    0.5555555555555556,0.6666666666666666,0.7777777777777778,0.8888888888888888]
possible_location_keep_values = [11,22,33,44,55,66,77,88,.11,.22,.33,.44,.55,.66,.77,.88,.56,.67,.78,.89]

if dataset_name is None or dataset_name.lower() in ['left_out_colored_mnist','color','col','c']:
    dataset_name = "left_out_colored_mnist"
elif dataset_name.lower() in ['left_out_varied_location_mnist','location','loc','l']:
    dataset_name = "left_out_varied_location_mnist"

if model_name is None or model_name.lower() in ['resnet','res']:
    model_name = 'resnet'
elif model_name.lower() in ['resnet_pretrained','pretrained','pretraining','resnet_pretraining']:
    model_name = 'resnet_pretrained'
elif model_name.lower() in ['resnet_pretrained_embeddings','resnet_pretrained_embed','pretrained_embeddings','pretrained_embed','embeddings','embed','embedded']:
    model_name = 'resnet_pretrained_embeddings'
elif model_name.lower() in ['resnet_no_pool','no_pool']:
    model_name = 'resnet_no_pool'
elif model_name.lower() in ['simple_cnn','simple']:
    model_name = 'simple_cnn'

if keep_pct == 'all':
    if dataset_name == "left_out_colored_mnist":
        keep_pct = color_keep_list
    else:
        keep_pct = location_keep_list

elif keep_pct is None:
    if dataset_name == "left_out_colored_mnist":
        keep_pct = 0.9
    else:
        keep_pct = 0.8888888888888888

elif round(float(keep_pct),2) in possible_location_keep_values:
    possible = location_keep_list + location_keep_list + [0.5555555555555556,0.6666666666666666,0.7777777777777778,0.8888888888888888]
    keep_pct = possible[possible_location_keep_values.index(round(float(keep_pct),2))]

elif float(keep_pct) in possible_color_keep_values:
    possible = color_keep_list + color_keep_list
    keep_pct = possible[possible_color_keep_values.index(float(keep_pct))]




hold_out = False
#model_name = "resnet_pretrained_embeddings"
#keep_pct = 0.2222222222222222 #0.3333333333333333 #0.4444444444444444 #0.5555555555555556 #0.6666666666666666 #0.7777777777777778 #0.8888888888888888
show_progress = True
num_classes = 10
if dataset_name == "left_out_varied_location_mnist":
    num_classes = 9


# shape/color
if dataset_name == "left_out_colored_mnist":
    unseen = set([(5,0),(6,0),(6,1),(7,1),(7,2),(8,2),(8,3),(9,3),(9,4),(0,4),
                  (0,5),(1,5),(1,6),(2,6),(2,7),(3,7),(3,8),(4,8),
                  (4,9),(5,9)])
    seen = set([(0,0),(1,0),(1,1),(2,1),(2,2),(3,2),(3,3),(4,3),
                (4,4),(5,4),(5,5),(6,5),(6,6),(7,6),(7,7),(8,7),
                (8,8),(9,8),(9,9),(0,9)])

# Shape/location
else:
    unseen = set([(4,0),(5,0),(5,1),(6,1),(6,2),(7,2),(7,3),(8,3),(8,4),
                  (0,4),(1,6),(2,6),(2,7),(3,7),(3,8),(4,8),(0,5),(1,5)])
    seen  =  set([(0,0),(1,0),(1,1),(2,1),(2,2),(3,2),(3,3),(4,3),(4,4),
                  (5,4),(5,5),(6,5),(6,6),(7,6),(7,7),(8,7),(8,8),(0,8)])


def store_activations(dataset_name, model_name, keep_pct, data):
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

def print_progress(current_iteration, tot_iterations, num_correct_count=None, col_correct_count=None):
    progress = current_iteration/tot_iterations * 100
    #sys.stdout.write( " Progress: %d%%    \r" % (progress))
    if num_correct_count is not None and col_correct_count is not None:
        num_acc = round(100*num_correct_count/current_iteration, 2)
        col_acc = round(100*col_correct_count/current_iteration, 2)
        sys.stdout.write( " Progress: {}%  num acc: {}%  col acc: {}% \r".format(round(progress,1), num_acc, col_acc))
    else:
        sys.stdout.write( " Progress: {}%   \r".format(round(progress,1)))
    sys.stdout.flush()

def restore_activations(path):
    with open(path,'rb') as f:
        act = pickle.load(f)
        for k in act:
            print(len(act[k]))
        print("loaded")

def add_to_act_dict(out_tensor, act_dict, act_key):
    if act_key in act_dict:
        act_dict[act_key] = np.concatenate([act_dict[act_key], out_tensor],axis=0)
    else:
        act_dict[act_key] = np.array(out_tensor)




def main(dataset_name, keep_pct, model_name):

    # Get dataset
    data_directory = "analysis/test_datasets/" + dataset_name + "/"
    data = pickle.load(open(data_directory + "data.p","rb"))
    targets = pickle.load(open(data_directory + "targets.p","rb"))
    print("loaded data")

    loader = utils.DataLoader(utils.TensorDataset(data, targets), batch_size=1)

    # Get model function
    state_dict_directory = "analysis/state_dicts/" + dataset_name + "/" + model_name + "/"
    model = config.model.options[model_name](num_classes)
    state_dict = torch.load(state_dict_directory + str(keep_pct) + ".pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # Get activations
    total = 0
    iteration_count = 0
    num_correct_count = 0
    col_correct_count = 0
    deb = 0

    with torch.no_grad():
        #layer before output
        activations_dict = {}

        activations_dict_layer1 = {}
        activations_dict_layer3 = {}

        num_examples = len(loader)

        # target -> digit, color
        for data, target in loader:

            iteration_count += 1
            if show_progress and iteration_count % 100 == 0:
                print_progress(iteration_count, num_examples, num_correct_count, col_correct_count)

            digit_class = target[0][0].item()
            color_class = target[0][1].item()
            act_key = (digit_class, color_class)
            if hold_out and act_key not in seen and act_key not in unseen:
                continue
            else:
                total += 1

            if model_name in ["resnet","resnet_pretrained","resnet_pretrained_embeddings","resnet_no_pool"]:
                #Accuracy
                num_output, col_output = model(data)
                pred = torch.cat((num_output.argmax(dim=1, keepdim=True), col_output.argmax(dim=1, keepdim=True)), 1)
                num_correct, col_correct = pred.eq(target.view_as(pred))[:, 0], pred.eq(target.view_as(pred))[:, 1]
                num_correct_count += num_correct.sum().item()
                col_correct_count += col_correct.sum().item()
                '''
                if act_key[0] == 9:
                    deb += 1
                    import matplotlib
                    from matplotlib import pyplot as plt

                    #im = np.moveaxis(np.array(data),1,3)
                    #print(num_output)
                    #print(col_output)
                    #plt.imshow(im[0])
                    #plt.show()

                    print(pred, act_key, deb)
                    if iteration_count > 100:
                        quit()
                '''
                #Activations
                out = F.relu(model.bn1(model.conv1(data)))
                out = model.layer1(out)

                if model_name == "resnet":
                    out_layer1 = np.reshape(np.mean(np.array(out.view(out.size(1),-1)),axis=1),(1,out.size(1)))
                    add_to_act_dict(out_layer1, activations_dict_layer1, act_key)

                out = model.layer2(out)
                out = model.layer3(out)

                if model_name == "resnet":
                    out_layer3 = np.reshape(np.mean(np.array(out.view(out.size(1),-1)),axis=1),(1,out.size(1)))
                    add_to_act_dict(out_layer3, activations_dict_layer3, act_key)

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
                add_to_act_dict(out, activations_dict, act_key)



            elif model_name in ['simple_cnn']:
                #Accuracy
                num_output, col_output = model(data)
                pred = torch.cat((num_output.argmax(dim=1, keepdim=True), col_output.argmax(dim=1, keepdim=True)), 1)
                num_correct, col_correct = pred.eq(target.view_as(pred))[:, 0], pred.eq(target.view_as(pred))[:, 1]
                num_correct_count += num_correct.sum().item()
                col_correct_count += col_correct.sum().item()
                '''
                if act_key[0] == 9:
                    deb += 1
                    import matplotlib
                    from matplotlib import pyplot as plt
                    im = np.moveaxis(np.array(data),1,3)

                    print(num_output)
                    print(col_output)

                    plt.imshow(im[0])
                    plt.show()

                    print(pred, act_key, deb)
                    if iteration_count > 100:
                        quit()
                '''

                #Activations
                out = F.relu(model.conv1(data))
                out = F.max_pool2d(out, 2, 2)
                out = F.relu(model.conv2(out))
                out = F.max_pool2d(out, 2, 2)
                out = out.view(-1,5*5*50)
                out = F.relu(model.fc1(out))

                add_to_act_dict(out, activations_dict, act_key)


        print("task 1 acc: ", 100. * num_correct_count / total)
        print("task 2 acc: ", 100. * col_correct_count / total)

        store_activations(dataset_name, model_name, keep_pct, activations_dict)
        if model_name in ["resnet"]:
            store_activations(dataset_name + '_layer1', model_name, keep_pct, activations_dict_layer1)
            store_activations(dataset_name + '_layer3', model_name, keep_pct, activations_dict_layer3)



if __name__ == '__main__':
    if type(keep_pct) == type([]):
        for kp in keep_pct:
            main(dataset_name, kp, model_name)
    else:
        main(dataset_name, keep_pct, model_name)
