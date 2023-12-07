import os
import time

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from tensorflow.python.keras import layers
from tensorflow import keras
import imageio.v2 as imageio

path = 'D:/Projects/Person re Identification/Datasets/MARS/bbox_test'

def Adaptive_Normalization():
    inputs = layers.Input(shape=(224, 224, 3))

def Loader():
    for file in os.listdir(path):
        start_time = time.time()
        # Printing percentage left
        print('Percentage left: ', round((len(os.listdir(path)) - int(file)) / len(os.listdir(path)) * 100, 3))
        # folders passed/total folders
        print('Folders passed: ', file, '/', len(os.listdir(path)))
        print("---------------------------------------------")


        if file == '0000' or file == '0001':
            continue
        for image in os.listdir(path + '/' + file ):
            image_values = imageio.imread(path + '/' + file + '/' + image)
            d = tf.image.resize(
                images=image_values,
                size=[224, 224],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                preserve_aspect_ratio=False,
                antialias=False,
                name=None
            )


#TODO: Add adaptive normalization
#TODO: Need labels ????????????????????????????

if __name__ == '__main__':
    Loader()