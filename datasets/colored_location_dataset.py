import numpy as np
from PIL import Image
import random
import torch
from torchvision import datasets, transforms


class LeftOutColoredLocationMNIST(datasets.MNIST):
    # Color classes
    color_name_map = ["red", "green", "blue", "yellow", "magenta", "cyan", "purple", "lime", "orange", "white"]
    color_map = [np.array([1, 0.1, 0.1]), np.array([0.1, 1, 0.1]), np.array([0.1, 0.1, 1]),
                 np.array([1, 1, 0.1]), np.array([1, 0.1, 1]), np.array([0.1, 1, 1]),
                 np.array([0.57, 0.12, 0.71]), np.array([0.72, 0.96, 0.24]), np.array([0.96, 0.51, 0.19]),
                 np.array([1, 1, 1])]

    # Gaussian noise arguments
    mu = 0
    sigma = 50

    # pct_to_keep: percentage of possible combinations to keep between 0 and 1, rounded down to nearest multiple of 1/9
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, pct_to_keep=1, color_indices=np.arange(9)):
        super().__init__(root, train, transform, target_transform, download)
        self.combination_space_shape = (9, 9, 9)
        self.class_names = ("shape", "color", "position")
        self.name = "left_out_colored_location_mnist"
        self.color_indices = color_indices
        d = 9
        pct = int(pct_to_keep * d)
        self.d = d
        shapes = np.arange(d)
        colors = np.arange(d)
        locs = np.arange(d)  # pass in a shuffled order for more randomness

        self.pct_to_dict = []

        for keep_pct in range(pct):
            pct_valid = {}
            for i in range(d):
                for j in range(d):
                    shape = shapes[i]
                    color = colors[(keep_pct + i + j) % d]
                    loc = locs[(i + j) % d]
                    if shape not in pct_valid:
                        pct_valid[shape] = {}
                    if color not in pct_valid[shape]:
                        pct_valid[shape][color] = []
                    pct_valid[shape][color].append(loc)
            self.pct_to_dict.append(pct_valid)

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

        # Get valid color, location
        tenth = random.randrange(len(self.pct_to_dict))
        color_class = random.randrange(self.d)
        loc_idx = random.randrange(len(self.pct_to_dict[tenth][target.item()][color_class]))
        loc_class = self.pct_to_dict[tenth][target.item()][color_class][loc_idx]

        # Put grayscale image in RGB space
        img_array = np.stack((img.numpy(),) * 3, axis=-1)

        # Determine image size
        size = 9

        img = Image.fromarray(img_array)
        img.thumbnail((size, size))
        img_array = np.zeros((28, 28, 3))

        # Adjust image location
        vert_loc_class = int(loc_class / 3)
        horiz_loc_class = loc_class % 3

        vert_offset = vert_loc_class * size
        horiz_offset = horiz_loc_class * size

        img_array[vert_offset:(vert_offset + size), horiz_offset:(horiz_offset + size), :] = np.array(img)
        img_array = img_array.astype(dtype=np.uint8)

        # Color image
        img_array = img_array * self.color_map[self.color_indices[color_class]]

        # Add Gaussian noise
        noise = np.reshape(np.random.normal(self.mu, self.sigma, img_array.size), img_array.shape)
        mask = (img_array != 0).astype("uint8")
        img_array = img_array + np.multiply(mask, noise)
        img_array = np.clip(img_array, 0, 255)
        img_array = img_array.astype("uint8")

        # 28x28 to 32x32
        zeros = np.zeros((img_array.shape[0] + 4, img_array.shape[1] + 4, img_array.shape[2]), dtype="uint8")
        zeros[2:img_array.shape[0] + 2, 2:img_array.shape[1] + 2, :] = img_array
        img_array = zeros

        img = Image.fromarray(img_array)

        # Perform any non-color transformations
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return transforms.ToTensor()(img), torch.tensor([target.item(), color_class, loc_class])
