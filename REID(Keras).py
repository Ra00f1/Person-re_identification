import os
import cv2
import matplotlib.pyplot as plt
from keras import layers, models, optimizers
from keras.applications import ResNet50
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from torchvision import transforms
import i_LIDS_VID
import tripletloss
from keras.applications.resnet50 import preprocess_input


# ... (other imports remain the same)

# Hyperparameters
class FeatureExtractor(models.Model):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False  # Freeze base model layers
        self.base_model = base_model
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(224, activation='relu')
        self.fc3 = layers.Dense(128)    # Output embedding size is 128

    def call(self, x):
        x = self.base_model(x)
        x = layers.Flatten()(x)
        x = self.fc1(x)
        x = self.fc2(x)  # Pass through fc2 before the final embedding layer
        embeddings = self.fc3(x)  # Final embeddings from fc3
        return embeddings


def Label_Input_Generator():
    print("Generating Labels and Inputs")
    Output_path = "D:\Projects\Person re Identification\Datasets\iLIDS-VID\i-LIDS-VID\images"
    Number_of_Cams = 2

    Input_path = "D:\Projects\Person re Identification\Datasets\iLIDS-VID\i-LIDS-VID\sequences"

    Labels = []
    Input = []

    # Getting all the labels
    for cam in range(1, Number_of_Cams + 1):
        Cam_path = os.path.join(Output_path, "cam" + str(cam))
        for person in os.listdir(Cam_path):
            Person_path = os.path.join(Cam_path, person)
            for image in os.listdir(Person_path):
                Labels.append([person, image, Person_path])
    # print(Labels)

    # Getting all the inputs
    for cam in range(1, Number_of_Cams + 1):
        Cam_path = os.path.join(Input_path, "cam" + str(cam))
        for person in os.listdir(Cam_path):
            Person_path = os.path.join(Cam_path, person)
            for image in os.listdir(Person_path):
                Input.append([person, image, Person_path])

    print("Labels and Inputs Generated")
    return Labels, Input


if __name__ == '__main__':
    print("Starting")

    model = FeatureExtractor()
    # triplet_loss = tfa.losses.TripletSemiHardLoss(margin=0.3)
    triplet_loss = tripletloss.TripletLoss(margin=0.3)
    optimizer = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=triplet_loss, optimizer=optimizer)
    model.build((None, 224, 224, 3))  # Necessary for the summary to work

    model.summary()  # Keras equivalent of summary

    batch_size = 128
    Labels, Inputs = Label_Input_Generator()

    for epoch in range(10):
        print(f"Epoch {epoch + 1}")
        batch = i_LIDS_VID.Triplet_Generator(Labels, Inputs, batch_size)
        # ... (loop structure remains the same)
        for image in batch:
            # ... (data loading remains the same)

            # Convert images to tensors using Keras preprocessing
            anchor_tensor = preprocess_input(image[0])
            anchor_tensor = tf.expand_dims(anchor_tensor, axis=0)

            positive_tensor = preprocess_input(image[1])
            positive_tensor = tf.expand_dims(positive_tensor, axis=0)

            negative_tensor = preprocess_input(image[2])
            negative_tensor = tf.expand_dims(negative_tensor, axis=0)

            anchor_emb = model(anchor_tensor)
            positive_emb = model(positive_tensor)
            negative_emb = model(negative_tensor)
            loss, tape = triplet_loss(anchor_emb, positive_emb, negative_emb)
            print(loss)
            # # Backpropagation and update
            optimizer.minimize(loss, var_list=model.trainable_variables, tape=tape)  # Use optimizer.minimize for Keras

            # ... (loss calculation and backpropagation remain the same)

        # ... (logging remains the same)
        print(f"Epoch {epoch + 1}")
