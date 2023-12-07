import os
import random

import cv2


class MARSDataset(Dataset):
    """
    Dataset class for loading and preprocessing MARS data.
    """
    def __init__(self, data_dir, image_size, transform=None):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform

        # Read image paths and labels from annotation files
        self.image_paths, self.labels = self._load_annotations()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read image data
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)

        # Resize and apply transformations
        if self.image_size:
            image = cv2.resize(image, self.image_size)
        if self.transform:
            image = self.transform(image)

        # Get image label
        label = self.labels[idx]

        return image, label

    def _load_annotations(self):
        """
        Reads image paths and labels from annotation files.
        """
        image_paths = []
        labels = []

        # Read train or test annotations based on data directory name
        annotation_file = os.path.join(self.data_dir, "info.txt")

        with open(annotation_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                image_path, label_str = line.strip().split(" ")
                image_paths.append(os.path.join(self.data_dir, image_path))
                labels.append(int(label_str))

        return image_paths, labels