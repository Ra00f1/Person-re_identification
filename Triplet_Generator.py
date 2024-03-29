import glob
from keras.utils import Sequence
import cv2
import numpy as np
import tenserflow as tf
import os


class MyDataGenerator(Sequence):
    def __init__(self, image_path, label_path, batch_size, Test=False):
        self.image_path = image_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.Test = Test

    def __get_images__(self):
        self.images_path = glob.glob(os.path.join(self.image_path, "*.jpg"))
        np.random.shuffle(self.images_path)
        return self.images_path

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        # Get current batch of image paths and labels
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]

        # Load and pre-process images
        batch_images = []
        for path in batch_paths:
            image = cv2.imread(path)
            # if Testing show the images
            if self.Test:
                cv2.imshow("image", image)

            image = tf.resize(
                images=image,
                size=[224, 224],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                preserve_aspect_ratio=False,
                antialias=False,
                name=None
            )
            batch_images.append(image)

        # Convert to NumPy arrays
        batch_images = np.array(batch_images)

        return batch_images