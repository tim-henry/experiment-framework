import numpy as np
from PIL import Image
import random
import torch
from torchvision import datasets, transforms


class LeftOutScaleMNIST(datasets.MNIST):
    # pct_to_keep: percentage of possible combinations to keep between 0 and 1, rounded down to nearest multiple of 1/9
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, pct_to_keep=1, color_indices=None):
        super().__init__(root, train, transform, target_transform, download)
        print(pct_to_keep)
        pct = pct_to_keep * 3
        print(pct)
        self.max_left_dist = int(pct / 2)
        self.max_right_dist = int(pct / 2) if pct % 2 == 0 else int(pct / 2) + 1
        self.held_out = [(0, 1), (1, 2), (2, 0), (3, 1), (4, 2), (5, 0), (6, 1), (7, 2), (8, 0)]
        self.control = [(0, 0), (1, 1), (2, 2), (3, 0), (4, 1), (5, 2), (6, 0), (7, 1), (8, 2)]
        self.held_out_val = self.held_out[:2]
        self.held_out_test = self.held_out[2:]
        self.control_val = self.control[:2]
        self.control_test = self.control[2:]
        self.combination_space_shape = (9, 3)
        self.class_names = ("shape", "scale")
        self.name = "left_out_scale_mnist"

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
        number_class = target.item()
        lower_bound = number_class - self.max_left_dist
        upper_bound = number_class + self.max_right_dist
        scale_class = random.randrange(lower_bound, upper_bound) % 3
        size = [28, 18, 9][scale_class]

        img = Image.fromarray(img_array)
        img.thumbnail((size, size))
        img_array = np.zeros((28, 28, 3))

        vert_offset = int((28 - size) / 2)
        horiz_offset = int((28 - size) / 2)

        img_array[vert_offset:(vert_offset + size), horiz_offset:(horiz_offset + size), :] = np.array(img)
        img_array = np.clip(img_array, 0, 254)
        img_array = img_array.astype(dtype=np.uint8)

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

        return transforms.ToTensor()(img), torch.tensor([target.item(), scale_class])
