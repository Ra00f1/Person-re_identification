import os
import cv2
import torch
from matplotlib import pyplot as plt
from torch.nn.functional import normalize
import tripletloss
import i_LIDS_VID
from torch.nn import Module, Conv2d, Linear, ReLU, Sequential, Flatten, MaxPool2d, AvgPool2d
import numpy as np


class FeatureExtractor(Module):
    def __init__(self, in_channels, hidden_dim):
        super(FeatureExtractor, self).__init__()
        self.layers = Sequential(
            Conv2d(224, hidden_dim, kernel_size=3, padding=1),
            ReLU(),

            MaxPool2d(kernel_size=2, stride=1),
            # Add additional convolutional layers and activation functions as needed
            Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=1),
            Flatten(),
            Linear(222, 64)
        )

    def forward(self, x):
        anchor_tensor = x[0]  # Convert each list to a tensor
        positive_tensor = x[1]
        negative_tensor = x[2]

        # change data type in tensor to float
        anchor_tensor = anchor_tensor.float()
        positive_tensor = positive_tensor.float()
        negative_tensor = negative_tensor.float()

        features = torch.stack([anchor_tensor, positive_tensor, negative_tensor])
        # Assume x is of shape (batch_size, 3, 224, 224)
        # Concatenate channels of anchor, positive, and negative images
        #features = torch.cat([anchor_tensor, positive_tensor, negative_tensor], dim=0)
        print(features)
        print(features.shape)

        return self.layers(features)


class TripletLoss(Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Calculate pairwise distances
        pos_dist = torch.nn.functional.pairwise_distance(anchor, positive)
        neg_dist = torch.nn.functional.pairwise_distance(anchor, negative)

        # Apply hinge loss with margin
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


def train(feature_extractor, optimizer, data):
    # ... data loading and processing ...
    # Extract features
    anchor, positive, negative = feature_extractor(data)
    # Calculate loss
    loss = tripletloss(anchor, positive, negative)
    # Backpropagation and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


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

    return Labels, Input


if __name__ == '__main__':
    print("Starting")

    model = FeatureExtractor(3, 64)
    triplet_loss = TripletLoss(0.3)

    # Extract features
    # features = model(image)

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
                features = model(image)
                print(features.shape)
                print(features)
                anchor, positive, negative = torch.split(features, 1, dim=0)

                # Calculate and back propagate loss
                loss = triplet_loss(anchor, positive, negative)
                # Backpropagation and update
                loss.backward()
                optimizer.step()

                # Accumulate loss and accuracy

        # Print or log metrics
        print(f"Epoch {epoch + 1}")
