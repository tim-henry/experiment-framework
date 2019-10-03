import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as utils

import config.model

dataset_name = "left_out_colored_mnist"
num_classes = 10
model_name = "resnet"
keep_pct = 0.9


def main():
    # Get held-out dataset
    data_directory = "test_datasets/" + dataset_name + "/"
    held_out_data = pickle.load(open(data_directory + "hold_out_data.p", "rb"))
    held_out_targets = pickle.load(open(data_directory + "hold_out_targets.p", "rb"))
    loader = utils.DataLoader(utils.TensorDataset(held_out_data, held_out_targets), batch_size=1)

    # Get model function

    state_dict_directory = "state_dicts/" + dataset_name + "/" + model_name + "/"
    model = config.model.options[model_name](num_classes)
    state_dict = torch.load(state_dict_directory + str(keep_pct) + ".pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # Get activations
    with torch.no_grad():
        for data, target in loader:
            if model_name == "resnet":
                out = F.relu(model.bn1(model.conv1(data)))
                out = model.layer1(out)
                out = model.layer2(out)
                out = model.layer3(out)
                out = model.layer4(out)
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)
                # out should now be a [1, 512] pytorch tensor containing
                # the activations of the network before the output layers


if __name__ == '__main__':
    main()
