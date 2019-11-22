import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as utils
import numpy as np
import os
import sys
from get_activations import print_progress

import config.model

dataset_name = "left_out_colored_mnist"
#dataset_name = "left_out_varied_location_mnist"
#model_name = "resnet"
keep_pct = 0.9
show_progress = True
num_classes = 10


def main(model_name):
    noise = True
    if noise:
        file = 'analysis/invariance_data_w_color_noise.npy'
        #file = 'analysis/invariance_data_100.npy'
    else:
        file = 'analysis/invariance_data.npy'

    #[2 , 10, N=10, 10, 32, 32, 3]
    data = np.load(file)

    # Get model function
    state_dict_directory = "analysis/state_dicts/" + dataset_name + "/" + model_name + "/"
    model = config.model.options[model_name](num_classes)
    state_dict = torch.load(state_dict_directory + str(keep_pct) + ".pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    tensor_size = 2*10*10*10
    iteration_count = 0
    if model_name != "simple_cnn":
        num_neurons = 512
    else:
        num_neurons = 500
    num_correct_count = 0
    col_correct_count = 0

    with torch.no_grad():
        #Make data shape [2,10,10,10,3,32,32]
        data_reshaped = np.moveaxis(np.array(data), 6,4)

        #Output shape [2,10,10,10,512]
        final_tensor = np.zeros((2,10,10,10,num_neurons))

        for a in range(2):
            for b in range(10):
                for c in range(10):
                    for d in range(10):
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
                        num_correct, col_correct = pred.eq(target.view_as(pred))[:, 0], pred.eq(target.view_as(pred))[:, 1]
                        num_correct_count += num_correct.sum().item()
                        col_correct_count += col_correct.sum().item()

                        if show_progress and iteration_count % 10 == 0:
                            print_progress(iteration_count, tensor_size, num_correct_count, col_correct_count)

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

        if noise:
            np.save("analysis/" + model_name + "_processed_array" + '_noise' + str(int(100*keep_pct)) +".npy",final_tensor)
        else:
            np.save("analysis/" + model_name + "_processed_array" + str(int(100*keep_pct)) +".npy",final_tensor)


if __name__ == '__main__':
    for model_name in ['resnet','resnet_pretrained','resnet_no_pool','resnet_pretrained_embeddings','simple_cnn']:
        main(model_name)
