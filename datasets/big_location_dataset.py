import numpy as np
from PIL import Image
import random
import torch
from torchvision import datasets, transforms


class LeftOutBigLocationMNIST(datasets.MNIST):
    # pct_to_keep: percentage of possible combinations to keep between 0 and 1, rounded down to nearest multiple of 1/9
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, pct_to_keep=1):
        super().__init__(root, train, transform, target_transform, download)
        print(root)
        import os
        print(os.getcwd())
        pct = pct_to_keep * 9
        self.max_left_dist = int(pct / 2)
        self.max_right_dist = int(pct / 2) if pct % 2 == 0 else int(pct / 2) + 1
        self.held_out = [(0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 0), (6, 1), (7, 2), (8, 3)]
        self.control = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
        self.combination_space_shape = (9, 9)
        self.class_names = ("shape", "position")
        self.name = "left_out_varied_location_mnist"

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
        assert target.item() < 9

        # Put grayscale image in RGB space
        img_array = np.stack((img.numpy(),) * 3, axis=-1)

        # Determine image size
        img_size = 28
        box_size = 74

        img = img_array.copy()
        # img.thumbnail((size, size))
        img_array = np.zeros((224, 224, 3))

        # Determine image location
        number_class = target.item()
        lower_bound = number_class - self.max_left_dist
        upper_bound = number_class + self.max_right_dist

        loc_class = random.randrange(lower_bound, upper_bound) % 9
        vert_loc_class = int(loc_class / 3)
        horiz_loc_class = loc_class % 3

        vert_noise = random.randrange(box_size - img_size)
        horiz_noise = random.randrange(box_size - img_size)

        vert_offset = (vert_loc_class * box_size) + vert_noise
        horiz_offset = horiz_loc_class * box_size + horiz_noise

        img_array[vert_offset:(vert_offset + img_size), horiz_offset:(horiz_offset + img_size), :] = img
        img_array = img_array.astype(dtype=np.uint8)

        # 28x28 to 32x32
        # zeros = np.zeros((img_array.shape[0] + 4, img_array.shape[1] + 4, img_array.shape[2]), dtype="uint8")
        # zeros[2:img_array.shape[0] + 2, 2:img_array.shape[1] + 2, :] = img_array
        # img_array = zeros

        img = Image.fromarray(img_array)

        # Perform any non-color transformations
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return transforms.ToTensor()(img), torch.tensor([target.item(), loc_class])
