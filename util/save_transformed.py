import datasets.colored_dataset
import datasets.varied_location_dataset
import datasets.scale_dataset
import datasets.many_scale_dataset
import datasets.col_loc_scale_dataset
import numpy as np
import os
import torch
import torch.utils.data as utils
import pickle
import errno


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_kwargs(use_cuda):
    return {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


training_file = 'training.pt'
test_file = 'test.pt'
batch_size = 128
in_path = '../../data'
color_indices = np.arange(10)

for dataset, dataset_name, dataset_size, out_size in \
    [
        # (datasets.many_scale_dataset.LeftOutManyScaleMNIST, "LeftOutManyScaleMNIST", 9, 2),
        # (datasets.colored_location_dataset.LeftOutColoredLocationMNIST, "LeftOutColoredLocationMNIST", 9, 3),
        (datasets.col_loc_scale_dataset.ColLocScaleMNIST, "ColLocScaleMNIST", 9, 4),
        # (datasets.colored_dataset.LeftOutColoredMNIST, "LeftOutColoredMNIST", 10),
        # (datasets.varied_location_dataset.LeftOutVariedLocationMNIST, "LeftOutVariedLocationMNIST", 9, 2),
        # (datasets.scale_dataset.LeftOutScaleMNIST, "LeftOutScaleMNIST", 3, 2)
    ]:
    print("\n" + dataset_name)
    for keep_pct in range(1, dataset_size + 1):  # number of diagonals (e.g. tenths in shape-color case)

        pct = keep_pct if dataset_name == "ColLocScaleMNIST" else keep_pct / dataset_size
        max_pct = dataset_size if dataset_name == "ColLocScaleMNIST" else 1

        print("Keep_pct:", keep_pct / dataset_size)
        out_path = "../../data/synth/" + dataset_name + "/" + str(keep_pct) + "/processed"
        mkdir_p(out_path)

        train_loader = utils.DataLoader(
                dataset(in_path, train=True, download=False, pct_to_keep=pct),
                batch_size=batch_size, shuffle=True, **get_kwargs(torch.cuda.is_available()))

        test_loader = utils.DataLoader(
                dataset(in_path, train=False, download=False, pct_to_keep=max_pct),
                batch_size=batch_size, shuffle=True, **get_kwargs(torch.cuda.is_available()))

        # Train
        print("TRAIN")
        train_data_out = torch.empty((0, 3, 96, 96))
        train_label_out = torch.empty((0, out_size), dtype=torch.int64)
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx % 10 == 0:
                print("BATCH", batch_idx)
            train_data_out = torch.cat((train_data_out, data), dim=0)
            train_label_out = torch.cat((train_label_out, target), dim=0)

        print(train_data_out.shape)
        with open(os.path.join(out_path, training_file), 'wb') as f:
            torch.save((train_data_out.numpy(), train_label_out.numpy()), f, pickle_protocol=4)

        # Test
        print("TEST")
        test_data_out = torch.empty((0, 3, 96, 96))
        test_label_out = torch.empty((0, out_size), dtype=torch.int64)
        for batch_idx, (data, target) in enumerate(test_loader):
            test_data_out = torch.cat((test_data_out, data), dim=0)
            test_label_out = torch.cat((test_label_out, target), dim=0)

        print(test_data_out.shape)
        with open(os.path.join(out_path, test_file), 'wb') as f:
            torch.save((test_data_out.numpy(), test_label_out.numpy()), f, pickle_protocol=4)
