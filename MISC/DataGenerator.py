from keras.utils import Sequence
import cv2
import numpy as np

class MyDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=32):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        # Get current batch of image paths and labels
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        # Load and pre-process images
        batch_images = []
        for path in batch_paths:
            image = cv2.imread(path)
            # Apply any necessary pre-processing steps
            # ...
            # TODO: add the resizing from MARS to here. (224, 224)
            batch_images.append(image)

        # Convert to NumPy arrays
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)

        return batch_images, batch_labels