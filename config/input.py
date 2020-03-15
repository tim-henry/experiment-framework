import datasets.colored_dataset
import datasets.varied_location_dataset
import datasets.colored_location_dataset
import datasets.col_loc_scale_dataset
import datasets.big_location_dataset
import datasets.generic_dataset
import datasets.scale_dataset
import datasets.many_scale_dataset
import numpy as np
import torch.utils.data as utils
import torchvision.datasets
from torchvision import transforms

np.random.seed(17)
random_indices = np.arange(9)
np.random.shuffle(random_indices)


def get_kwargs(use_cuda):
    return {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


def left_out_colored_mnist_train_loader(args):
    dataset = datasets.generic_dataset.LeftOutColoredMNIST('../data/synth', train=True, keep_num=args['keep_pct'])
    indices = list(range(len(dataset)))
    split = int(np.floor(args['example_pct'] * len(dataset)))
    np.random.shuffle(indices)

    return utils.DataLoader(
        dataset,
        batch_size=args['batch_size'], sampler=utils.sampler.SubsetRandomSampler(indices[:split]),
        **get_kwargs(args['use_cuda']))


def left_out_colored_mnist_test_loader(args):
    return utils.DataLoader(
        datasets.generic_dataset.LeftOutColoredMNIST('../data/synth', train=False),
        batch_size=args['test_batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))


def left_out_varied_location_mnist_train_loader(args):
    dataset = datasets.generic_dataset.LeftOutVariedLocationMNIST('../data/synth', train=True, download=True, keep_num=args['keep_pct'])
    indices = list(range(len(dataset)))
    split = int(np.floor(args['example_pct'] * len(dataset)))
    np.random.shuffle(indices)

    return utils.DataLoader(
        dataset,
        batch_size=args['batch_size'], sampler=utils.sampler.SubsetRandomSampler(indices[:split]),
        **get_kwargs(args['use_cuda']))


def left_out_varied_location_mnist_test_loader(args):
    return utils.DataLoader(
        datasets.generic_dataset.LeftOutVariedLocationMNIST('../data/synth', train=False),
        batch_size=args['test_batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))


def left_out_scale_mnist_train_loader(args):
    dataset = datasets.generic_dataset.LeftOutScaleMNIST('../data/synth', train=True, keep_num=args['keep_pct'])
    indices = list(range(len(dataset)))
    split = int(np.floor(args['example_pct'] * len(dataset)))
    np.random.shuffle(indices)

    return utils.DataLoader(
            dataset,
            batch_size=args['batch_size'], sampler=utils.sampler.SubsetRandomSampler(indices[:split]),
            **get_kwargs(args['use_cuda']))


def left_out_scale_mnist_test_loader(args):
    return utils.DataLoader(
        datasets.generic_dataset.LeftOutScaleMNIST('../data/synth', train=False),
        batch_size=args['test_batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))


def left_out_many_scale_mnist_train_loader(args):
    dataset = datasets.generic_dataset.LeftOutManyScaleMNIST('../data/synth', train=True, keep_num=args['keep_pct'])
    indices = list(range(len(dataset)))
    split = int(np.floor(args['example_pct'] * len(dataset)))
    np.random.shuffle(indices)

    return utils.DataLoader(
            dataset,
            batch_size=args['batch_size'], sampler=utils.sampler.SubsetRandomSampler(indices[:split]),
            **get_kwargs(args['use_cuda']))


def left_out_many_scale_mnist_test_loader(args):
    return utils.DataLoader(
        datasets.generic_dataset.LeftOutManyScaleMNIST('../data/synth', train=False),
        batch_size=args['test_batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))


def left_out_many_scale_biased_mnist_train_loader(args):
    dataset = datasets.many_scale_dataset.LeftOutManyScaleMNIST('../data/', train=True, download=False,
                                                     pct_to_keep=args['keep_pct'], color_indices=np.arange(9))
    indices = list(range(len(dataset)))
    split = int(np.floor(args['example_pct'] * len(dataset)))
    np.random.shuffle(indices)

    return utils.DataLoader(
            dataset,
            batch_size=args['batch_size'], sampler=utils.sampler.SubsetRandomSampler(indices[:split]),
            **get_kwargs(args['use_cuda']))


def left_out_many_scale_biased_mnist_test_loader(args):
    dataset = datasets.many_scale_dataset.LeftOutManyScaleMNIST(
        '../data/', train=False, download=False, pct_to_keep=1, color_indices=np.arange(9))
    return utils.DataLoader(
        dataset,
        batch_size=args['test_batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))


def left_out_colored_loc_mnist_train_loader(args):
    dataset = datasets.generic_dataset.LeftOutColoredLocationMNIST('../data/synth', train=True,
                                                                   keep_num=args['keep_pct'])
    indices = list(range(len(dataset)))
    split = int(np.floor(args['example_pct'] * len(dataset)))
    np.random.shuffle(indices)

    return utils.DataLoader(
            dataset,
            batch_size=args['batch_size'], sampler=utils.sampler.SubsetRandomSampler(indices[:split]),
            **get_kwargs(args['use_cuda']))


def left_out_colored_loc_mnist_test_loader(args):
    return utils.DataLoader(
        datasets.generic_dataset.LeftOutColoredLocationMNIST('../data/synth', train=False),
        batch_size=args['test_batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))


# def left_out_colored_loc_mnist_train_loader(args):
#     return utils.DataLoader(
#         datasets.colored_location_dataset.LeftOutColoredLocationMNIST('../data/', train=True, download=False, pct_to_keep=args['keep_pct']),
#         batch_size=args['batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))
#
#
# def left_out_colored_loc_mnist_test_loader(args):
#     return utils.DataLoader(
#         datasets.colored_location_dataset.LeftOutColoredLocationMNIST('../data/', train=False, download=False, pct_to_keep=9),
#         batch_size=args['batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))


def col_loc_scale_mnist_train_loader(args):
    return utils.DataLoader(
        datasets.col_loc_scale_dataset.ColLocScaleMNIST('../data/', train=True, download=False, pct_to_keep=args['keep_pct']),
        batch_size=args['batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))


def col_loc_scale_mnist_test_loader(args):
    return utils.DataLoader(
        datasets.col_loc_scale_dataset.ColLocScaleMNIST('../data/', train=False, download=False, pct_to_keep=9),
        batch_size=args['batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))


# def col_loc_scale_mnist_train_loader(args):
#     dataset = datasets.generic_dataset.ColLocScaleMNIST('../data/synth', train=True, keep_num=args['keep_pct'])
#     indices = list(range(len(dataset)))
#     split = int(np.floor(args['example_pct'] * len(dataset)))
#     np.random.shuffle(indices)
#
#     return utils.DataLoader(
#             dataset,
#             batch_size=args['batch_size'], sampler=utils.sampler.SubsetRandomSampler(indices[:split]),
#             **get_kwargs(args['use_cuda']))
#
#
# def col_loc_scale_mnist_test_loader(args):
#     return utils.DataLoader(
#         datasets.generic_dataset.ColLocScaleMNIST('../data/synth', train=False),
#         batch_size=args['test_batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))

# TODO move
def left_out_big_location_mnist_train_loader(args):
    return utils.DataLoader(
        datasets.big_location_dataset.LeftOutBigLocationMNIST('../data', train=True, download=False,
                                                     pct_to_keep=args['keep_pct']),
        batch_size=16, shuffle=True, **get_kwargs(args['use_cuda']))


# TODO move
def left_out_big_location_mnist_test_loader(args):
    return utils.DataLoader(
        datasets.big_location_dataset.LeftOutBigLocationMNIST('../data', train=False, download=False, pct_to_keep=1),
        batch_size=16, shuffle=True, **get_kwargs(args['use_cuda']))


def cifar10_train_loader(args):
    return utils.DataLoader(torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=False, transform=transforms.ToTensor()), batch_size=4,
                                              shuffle=True, num_workers=2)


def cifar10_test_loader(args):
    return utils.DataLoader(torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=False, transform=transforms.ToTensor()), batch_size=4,
                                             shuffle=False, num_workers=2)


options = {
    "left_out_colored_mnist": (left_out_colored_mnist_train_loader,
                               left_out_colored_mnist_test_loader),
    "left_out_varied_location_mnist": (left_out_varied_location_mnist_train_loader,
                                       left_out_varied_location_mnist_test_loader),
    "left_out_colored_location_mnist": (left_out_colored_loc_mnist_train_loader,
                                        left_out_colored_loc_mnist_test_loader),
    "left_out_scale_mnist": (left_out_scale_mnist_train_loader,
                             left_out_scale_mnist_test_loader),
    "left_out_many_scale_mnist": (left_out_many_scale_mnist_train_loader,
                                  left_out_many_scale_mnist_test_loader),
    "left_out_many_scale_biased_mnist": (left_out_many_scale_biased_mnist_train_loader,
                                  left_out_many_scale_biased_mnist_test_loader),
    "cifar10": (cifar10_train_loader, cifar10_test_loader),
    "left_out_big_location_mnist": (left_out_big_location_mnist_train_loader,
                                    left_out_big_location_mnist_test_loader),
    "col_loc_scale_mnist": (col_loc_scale_mnist_train_loader,
                            col_loc_scale_mnist_test_loader),
}
