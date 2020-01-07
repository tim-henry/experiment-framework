import os
import torch
from torchvision import datasets


class GenericMNIST(datasets.MNIST):
    @property
    def get_root(self):
        print(os.path.join(os.getcwd(), os.path.join(self.root_prefix, self.variation_name, str(self.keep_num))))
        print("ROOT:", os.path.join(self.root_prefix, self.variation_name, str(self.keep_num)))
        return os.path.join(self.root_prefix, self.variation_name, str(self.keep_num))

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    # keep_num: numerator of fraction of compositions used, e.g. 2/9 -> 2, 10/10 -> 10, etc.
    def __init__(self, root_prefix, train=True, transform=None, target_transform=None, keep_num=None, download=False):
        print("INIT")
        self.root_prefix = root_prefix
        self.keep_num = keep_num
        self.root = self.get_root
        super().__init__(self.root, train, transform, target_transform, download=download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # Perform any non-color transformations
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        # return torch.from_numpy(img), torch.from_numpy(target)


class LeftOutColoredMNIST(GenericMNIST):
    variation_name = "LeftOutColoredMNIST"
    name = "left_out_colored_mnist"
    class_names = ("shape", "color")
    combination_space_shape = (10, 10)

    def __init__(self, root, train=True, keep_num=10):
        super().__init__(root, train=train, keep_num=keep_num)
        self.held_out = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 0), (6, 1), (7, 2), (8, 3), (9, 4)]
        self.control = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
        self.held_out_val = self.held_out[:2]
        self.held_out_test = self.held_out[2:]
        self.control_val = self.control[:2]
        self.control_test = self.control[2:]


class LeftOutVariedLocationMNIST(GenericMNIST):
    variation_name = "LeftOutVariedLocationMNIST"
    name = "left_out_varied_location_mnist"
    class_names = ("shape", "position")
    combination_space_shape = (9, 9)

    def __init__(self, root, train=True, keep_num=9, download=False):
        super().__init__(root, train=train, keep_num=keep_num, download=download)
        self.held_out = [(0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 0), (6, 1), (7, 2), (8, 3)]
        self.control = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
        self.held_out_val = self.held_out[:2]
        self.held_out_test = self.held_out[2:]
        self.control_val = self.control[:2]
        self.control_test = self.control[2:]


class LeftOutScaleMNIST(GenericMNIST):
    variation_name = "LeftOutScaleMNIST"
    name = "left_out_scale_mnist"
    class_names = ("shape", "scale")
    combination_space_shape = (9, 3)

    def __init__(self, root, train=True, keep_num=3):
        super().__init__(root, train=train, keep_num=keep_num)
        self.held_out = [(0, 1), (1, 2), (2, 0), (3, 1), (4, 2), (5, 0), (6, 1), (7, 2), (8, 0)]
        self.control = [(0, 0), (1, 1), (2, 2), (3, 0), (4, 1), (5, 2), (6, 0), (7, 1), (8, 2)]
        self.held_out_val = self.held_out[:2]
        self.held_out_test = self.held_out[2:]
        self.control_val = self.control[:2]
        self.control_test = self.control[2:]


class LeftOutColoredLocationMNIST(GenericMNIST):
    variation_name = "LeftOutColoredLocationMNIST"
    name = "left_out_colored_location_mnist"
    class_names = ("shape", "color", "position")
    combination_space_shape = (9, 9, 9)

    def __init__(self, root, train=True, keep_num=3):
        super().__init__(root, train=train, keep_num=keep_num)
        self.held_out = list({(8, 3, 2), (1, 4, 0), (1, 7, 0), (0, 2, 7), (6, 5, 3), (3, 8, 6), (4, 1, 1), (5, 3, 2), (1, 8, 6),
                (0, 8, 8), (6, 6, 7), (4, 6, 5), (5, 8, 7), (7, 5, 2), (7, 3, 2), (1, 8, 8), (8, 2, 5), (2, 3, 0),
                (2, 4, 0), (3, 4, 3), (0, 1, 6), (6, 2, 4), (5, 7, 5), (5, 1, 1), (7, 0, 5), (3, 6, 4), (2, 2, 3),
                (0, 5, 8), (4, 0, 8), (8, 1, 2), (4, 4, 3), (0, 5, 4), (6, 2, 8), (8, 1, 6), (1, 1, 4), (4, 2, 3),
                (4, 4, 1), (6, 0, 1), (3, 0, 4), (2, 5, 8), (3, 6, 8), (0, 5, 0), (6, 5, 8), (3, 7, 7), (4, 7, 5),
                (8, 3, 1), (2, 0, 6), (0, 7, 7), (3, 1, 1), (1, 7, 5), (7, 4, 4), (8, 8, 4), (2, 6, 2), (4, 1, 0),
                (5, 3, 3), (1, 2, 5), (6, 6, 0), (7, 5, 5), (0, 8, 7), (7, 5, 3), (5, 8, 4), (5, 6, 1), (3, 4, 6),
                (8, 2, 2), (1, 0, 0), (7, 6, 7), (2, 3, 7), (4, 6, 8), (5, 7, 6), (1, 0, 4), (2, 3, 3), (8, 4, 3),
                (0, 3, 2), (8, 4, 1), (5, 7, 0), (6, 1, 7), (7, 0, 2), (6, 8, 5), (3, 0, 1), (7, 7, 6), (2, 2, 6)},
            )
        self.control = list({(8, 6, 6), (0, 0, 8), (4, 7, 6), (0, 5, 7), (7, 2, 1), (3, 7, 0), (2, 5, 7), (2, 4, 8), (6, 6, 5),
                (6, 4, 8), (0, 1, 0), (5, 6, 2), (7, 8, 6), (5, 8, 3), (1, 0, 1), (7, 6, 4), (6, 2, 6), (6, 3, 7),
                (5, 6, 6), (8, 2, 7), (1, 6, 8), (1, 0, 5), (0, 0, 3), (4, 0, 4), (4, 5, 4), (6, 8, 0), (4, 4, 7),
                (2, 1, 3), (7, 1, 8), (4, 4, 5), (3, 5, 0), (4, 5, 8), (0, 3, 5), (4, 2, 1), (1, 1, 2), (2, 8, 7),
                (5, 2, 4), (8, 7, 8), (7, 2, 4), (8, 0, 7), (8, 6, 1), (5, 5, 0), (8, 3, 3), (3, 0, 6), (2, 7, 5),
                (3, 7, 1), (3, 8, 3), (2, 0, 2), (3, 8, 5), (6, 5, 0), (3, 2, 7), (0, 7, 3), (7, 6, 3), (3, 4, 4),
                (0, 4, 7), (1, 3, 4), (7, 8, 5), (8, 5, 2), (2, 3, 5), (5, 8, 2), (1, 5, 2), (1, 3, 0), (7, 8, 1),
                (5, 6, 5), (1, 0, 6), (5, 1, 6), (3, 3, 2), (2, 1, 8), (5, 4, 2), (6, 4, 3), (8, 7, 3), (6, 2, 1),
                (6, 1, 1), (2, 1, 4), (0, 3, 0), (0, 3, 6), (8, 1, 1), (7, 7, 4), (4, 2, 0), (1, 7, 8), (4, 4, 2)})
        self.held_out_val = self.held_out[:2]
        self.held_out_test = self.held_out[2:]
        self.control_val = self.control[:2]
        self.control_test = self.control[2:]


class LeftOutManyScaleMNIST(GenericMNIST):
    variation_name = "LeftOutManyScaleMNIST"
    name = "left_out_many_scale_mnist"
    class_names = ("shape", "scale")
    combination_space_shape = (9, 9)

    def __init__(self, root, train=True, keep_num=9, download=False):
        super().__init__(root, train=train, keep_num=keep_num, download=download)
        self.held_out = [(0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 0), (6, 1), (7, 2), (8, 3)]
        self.control = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
        self.held_out_val = self.held_out[:2]
        self.held_out_test = self.held_out[2:]
        self.control_val = self.control[:2]
        self.control_test = self.control[2:]
