import os
import cv2
from matplotlib import pyplot as plt
from torch.nn.functional import normalize
import i_LIDS_VID
import tensorflow as tf
import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Linear, ReLU, Sequential, Flatten, MaxPool2d, AvgPool2d, Sigmoid
import numpy as np
from torchsummary import summary
from torchvision import transforms


# Hyperparameters
class FeatureExtractor(Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        for param in resnet50.parameters():
            param.requires_grad = False
        self.base_model = resnet50
        self.fc1 = nn.Linear(1000, 512)  # Adjust input size based on ResNet50 output
        self.fc2 = nn.Linear(512, 224)  # Output embedding size

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)  # Output embedding
        return x


# class Classifier(Module):


def Visualize_Features(features):
    channel_1 = normalize(features[:, 0, :, :])
    channel_2 = normalize(features[:, 1, :, :])

    # Convert to numpy arrays
    channel_1 = channel_1.squeeze().detach().numpy()
    channel_2 = channel_2.squeeze().detach().numpy()

    # Create a subplot for each channel
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(channel_1, cmap="gray")
    plt.title("Channel 1")

    plt.subplot(1, 2, 2)
    plt.imshow(channel_2, cmap="hot")
    plt.title("Channel 2")

    plt.tight_layout()
    plt.show()


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

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to PyTorch tensor and scale to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean/std
    ])

    model = FeatureExtractor()
    triplet_loss = nn.TripletMarginLoss(margin=0.3)
    summary(model, (3, 224, 224))

    batch_size = 128
    Labels, Inputs = Label_Input_Generator()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        print(f"Epoch {epoch + 1}")
        for i in range(20):
            batch = i_LIDS_VID.Triplet_Generator(Labels, Inputs, batch_size)
            for image in batch:
                # ... data loading and processing ...
                # Extract features
                optimizer.zero_grad()
                anchor_numpy = image[0].numpy()
                anchor_image = transform(anchor_numpy)
                anchor_image = tf.reshape(anchor_image, [1, 3, 224, 224])

                positive_numpy = image[1].numpy()
                positive_image = transform(positive_numpy)
                positive_image = tf.reshape(positive_image, [1, 3, 224, 224])

                negative_numpy = image[2].numpy()
                negative_image = transform(negative_numpy)
                negative_image = tf.reshape(negative_image, [1, 3, 224, 224])

                # print(anchor_image)
                # print(positive_image)

                anchor_emb = model(anchor_image)
                positive_emb = model(positive_image)
                negative_emb = model(negative_image)

                # Calculate and back propagate loss
                loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
                # Backpropagation and update
                loss.backward()
                optimizer.step()

        # Print or log metrics
        print(f"Epoch {epoch + 1}")
