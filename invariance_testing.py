import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as utils
import numpy as np
import os
import sys
from get_activations import print_progress

import config.model
import argparse

#---------------------- Parse input ----------------------------------
# dataset: name of dataset being used
# only: specifies if network is trained on single task (as well as which task)
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default=None,help="Choose dataset type ('color','location','scale')")
parser.add_argument("--only",default="",help="Choose if network models are only trained on single task (shape, location, or scale)")

args = parser.parse_args()

if args.dataset is None:
    dataset_name = "left_out_colored_mnist"
elif dataset_name in ['color','location','scale']:
    dataset_name = {'color':"left_out_colored_mnist",
                    'location':"left_out_varied_location_mnist",
                    'scale':"left_out_many_scale_mnist"}[args.dataset.lower()]


only = {"shape":"_shape_only",
        "location":"_location_only",
        "scale":"_scale_only",
        "":""}[args.only.lower()]


# For directory navigation purposes. Tells what state_dict files are named depending on dataset
keep_list1 = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
keep_list2=[0.1111111111111111,0.2222222222222222,0.3333333333333333,0.4444444444444444,
                    0.5555555555555556,0.6666666666666666,0.7777777777777778,0.8888888888888888]
alt_keep_list_10_classes = [1,2,3,4,5,6,7,8,9]
alt_keep_list_9_classes = [1,2,3,4,5,6,7,8]

keep_list_dict = {"left_out_colored_mnist":keep_list1,
                  "left_out_varied_location_mnist":keep_list2,
                  "left_out_many_scale_mnist":keep_list2}
if only != "":
    keep_list_dict = {"left_out_colored_mnist":alt_keep_list_10_classes,
                      "left_out_varied_location_mnist":alt_keep_list_9_classes,
                      "left_out_many_scale_mnist":alt_keep_list_9_classes}

show_progress = True


def main(model_name, keep_pct):
    keep_pct_str = str(int(keep_pct*100))
    noise = True
    '''
    if noise:
        file = 'analysis/invariance_data_w_color_noise.npy'
        #file = 'analysis/invariance_data_100.npy'
    else:
        file = 'analysis/invariance_data.npy'
    '''
    file = "analysis/invariance_datasets/invariance_"+dataset_name+".npy"

    #data shape = [2 , num_classes, N, num_classes, img_size, img_size, num_channels]
    data = np.load(file)
    num_tasks = data.shape[0]
    num_classes = data.shape[1]
    N = data.shape[2]
    #print(num_tasks, num_classes, N)
    #quit()

    # Get model function. If only is not empty string, files are named 1-9 instead of .1-.9
    if only != "":
        only_extension = "keep_pct_readout"+only+"/"
    else:
        only_extension = ""

    state_dict_directory = "analysis/state_dicts/" + dataset_name + "/" + model_name + "/" + only_extension
    state_dict_file = state_dict_directory + str(keep_pct) + ".pt"

    if not os.path.exists(state_dict_directory):
        print("path doesn't exist:",state_dict_directory)
        return

    if not os.path.exists(state_dict_file):
        print(keep_pct)
        state_dict_file = state_dict_directory + str(int(10*keep_pct)) + ".pt"
        if not os.path.exists(state_dict_file):
            print("path doesn't exist :", state_dict_directory)
            return


    model = config.model.options[model_name](num_classes)
    state_dict = torch.load(state_dict_file, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    tensor_size = num_tasks*num_classes*N*num_classes
    iteration_count = 0
    if model_name != "simple_cnn":
        num_neurons = 512
    else:
        num_neurons = 500
    num_correct_count = 0
    col_correct_count = 0

    with torch.no_grad():
        #Make data shape [2,num_classes,N,num_classes,3,32,32]
        data_reshaped = np.moveaxis(np.array(data), 6,4)

        #Output shape [2,10,10,10,512]
        final_tensor = np.zeros((num_tasks,num_classes,N,num_classes,num_neurons))
        layer1_tensor = np.zeros((num_tasks,num_classes,N,num_classes,64))
        layer3_tensor = np.zeros((num_tasks,num_classes,N,num_classes,256))

        for a in range(num_tasks):
            for b in range(num_classes):
                for c in range(N):
                    for d in range(num_classes):
                        iteration_count += 1

                        #Make data tensor of shape [1,3,32,32]
                        data = torch.tensor([data_reshaped[a,b,c,d,:,:,:]], dtype=torch.float)/255

                        '''
                        import matplotlib
                        from matplotlib import pyplot as plt
                        img = np.moveaxis(np.array(data[0]),0,2)/255
                        print(img[img[:,:,0] > 0])
                        plt.imshow(img)
                        plt.show()
                        if iteration_count >=5:
                            quit()
                        '''

                        if a == 0:
                            target = torch.tensor([[b,d]])
                        else:
                            target = torch.tensor([[d,b]])

                        num_output, col_output = model(data)

                        pred = torch.cat((num_output.argmax(dim=1, keepdim=True), col_output.argmax(dim=1, keepdim=True)), 1)
                        #print(pred,target)


                        num_correct, col_correct = pred.eq(target.view_as(pred))[:, 0], pred.eq(target.view_as(pred))[:, 1]
                        num_correct_count += num_correct.sum().item()
                        col_correct_count += col_correct.sum().item()

                        if show_progress and iteration_count % 10 == 0:
                            print_progress(iteration_count, tensor_size, num_correct_count, col_correct_count)

                        if model_name != "simple_cnn":
                            out = F.relu(model.bn1(model.conv1(data)))
                            out = model.layer1(out)
                            out_layer1 = np.reshape(np.mean(np.array(out.view(out.size(1),-1)),axis=1),(1,out.size(1)))
                            layer1_tensor[a,b,c,d] = out_layer1[0]

                            out = model.layer2(out)
                            out = model.layer3(out)
                            out_layer3 = np.reshape(np.mean(np.array(out.view(out.size(1),-1)),axis=1),(1,out.size(1)))
                            layer3_tensor[a,b,c,d] = out_layer3[0]

                            out = model.layer4(out)
                            if model_name != "resnet_no_pool":
                                out = F.avg_pool2d(out, 4)
                            else:
                                out = out.view(out.size(0),-1)
                                out = model.pool_fc(out)
                            out = F.relu(out)
                            out = out.view(out.size(0), -1)

                        else:
                            out = F.relu(model.conv1(data))
                            out = F.max_pool2d(out, 2, 2)
                            out = F.relu(model.conv2(out))
                            out = F.max_pool2d(out, 2, 2)
                            out = out.view(-1,5*5*50)
                            out = F.relu(model.fc1(out))

                        final_tensor[a,b,c,d] = out[0]

        file_path = "analysis/"+model_name+'/invariance_data/'+dataset_name+"/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        if model_name != "simple_cnn":
            np.save(file_path + model_name +"_"+ dataset_name +"_keep"+ keep_pct_str +"_layer1.npy",layer1_tensor)
            np.save(file_path + model_name +"_"+ dataset_name +"_keep"+ keep_pct_str +"_layer3.npy",layer3_tensor)
        np.save(file_path + model_name +"_"+ dataset_name +"_keep"+ keep_pct_str +".npy",final_tensor)



if __name__ == '__main__':
    for model_name in ['resnet','simple_cnn','resnet_pretrained','resnet_no_pool','resnet_pretrained_embeddings']:
        for keep_pct in keep_list_dict[dataset_name]:
            main(model_name, keep_pct)
