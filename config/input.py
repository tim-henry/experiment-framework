import datasets.colored_dataset
import datasets.varied_location_dataset
import datasets.colored_location_dataset
import datasets.big_location_dataset
import torch.utils.data as utils
import torchvision.datasets
from torchvision import transforms


def get_kwargs(use_cuda):
    return {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


def left_out_colored_mnist_train_loader(args):
    return utils.DataLoader(
        datasets.colored_dataset.LeftOutColoredMNIST('../data/', train=True, download=True, pct_to_keep=args['keep_pct'],
                                                     color_indices=args['color_indices']),
        batch_size=args['batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))


def left_out_colored_mnist_test_loader(args):
    return utils.DataLoader(
        datasets.colored_dataset.LeftOutColoredMNIST('../data/', train=False, download=True, pct_to_keep=1,
                                                     color_indices=args['color_indices']),
        batch_size=args['test_batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))


def left_out_varied_location_mnist_train_loader(args):
    return utils.DataLoader(
        datasets.varied_location_dataset.LeftOutVariedLocationMNIST('../data/', train=True, download=False,
                                                     pct_to_keep=args['keep_pct']),
        batch_size=args['batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))


def left_out_varied_location_mnist_test_loader(args):
    return utils.DataLoader(
        datasets.varied_location_dataset.LeftOutVariedLocationMNIST('../data/', train=False, download=False, pct_to_keep=1),
        batch_size=args['test_batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))


def left_out_colored_loc_mnist_train_loader(args):
    return utils.DataLoader(
        datasets.colored_location_dataset.LeftOutColoredLocationMNIST('../data/', train=True, download=False, pct_to_keep=args['keep_pct']),
        batch_size=args['batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))


def left_out_colored_loc_mnist_test_loader(args):
    return utils.DataLoader(
        datasets.colored_location_dataset.LeftOutColoredLocationMNIST('../data/', train=False, download=False, pct_to_keep=1),
        batch_size=args['batch_size'], shuffle=True, **get_kwargs(args['use_cuda']))


def left_out_big_location_mnist_train_loader(args):
    return utils.DataLoader(
        datasets.big_location_dataset.LeftOutBigLocationMNIST('../data/', train=True, download=False,
                                                     pct_to_keep=args['keep_pct']),
        batch_size=16, shuffle=True, **get_kwargs(args['use_cuda']))


def left_out_big_location_mnist_test_loader(args):
    return utils.DataLoader(
        datasets.big_location_dataset.LeftOutBigLocationMNIST('../data/', train=False, download=False, pct_to_keep=1),
        batch_size=16, shuffle=True, **get_kwargs(args['use_cuda']))


def cifar10_train_loader(args):
    return utils.DataLoader(torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transforms.ToTensor()), batch_size=4,
                                              shuffle=True, num_workers=2)


def cifar10_test_loader(args):
    return utils.DataLoader(torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transforms.ToTensor()), batch_size=4,
                                             shuffle=False, num_workers=2)


options = {
    "left_out_colored_mnist": (left_out_colored_mnist_train_loader,
                               left_out_colored_mnist_test_loader),
    "left_out_varied_location_mnist": (left_out_varied_location_mnist_train_loader,
                                       left_out_varied_location_mnist_test_loader),
    "left_out_colored_location_mnist": (left_out_colored_loc_mnist_train_loader,
                                        left_out_colored_loc_mnist_test_loader),
    "cifar10": (cifar10_train_loader, cifar10_test_loader),
    "left_out_big_location_mnist": (left_out_big_location_mnist_train_loader,
                                    left_out_big_location_mnist_test_loader),
}
