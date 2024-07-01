# Person Re-Identification (ReID) Project
## Overview
This project focuses on Person Re-Identification (ReID), a technology crucial in computer vision and video analysis. ReID involves recognizing and tracking individuals across multiple non-overlapping camera views, linking or matching images of the same person captured by different cameras within a surveillance system. This project leverages various methodologies, datasets, and models to improve the accuracy and efficiency of ReID systems.

## Applications
Security and Surveillance: Track individuals across cameras in public spaces to identify suspects and monitor crowds.
Retail and Marketing: Personalize advertising, identify VIP customers, and optimize store layouts based on customer movement.
Healthcare and Robotics: Monitor elderly patients, track hospital patients, and assist robot navigation by identifying individuals.
## Methodologies
### Key Papers
#### Person Re-Identification Using Color Features and CNN Features
#### Semantic-Guided Pixel Sampling for Cloth-Changing Person Re-Identification
#### Occluded Person Re-Identification Via Relational Adaptive Feature Correction Learning (attention Maps)
#### Person Re-Identification from CCTV Silhouettes Using Generic Fourier Descriptor (Human Silhouettes)
#### Long-Term Person Re-Identification Based on Appearance and Gait Feature Fusion under Covariate Changes
#### Night Person Re-Identification and a Benchmark
## Models
YOLOv5: Used for accurate and efficient real-time person detection.

ResNet-50: Extracts meaningful features from detected individuals for accurate ReID.

Triplet Loss: Enhances the learning of discriminative features for better ReID performance.
## Datasets
### iLIDS-VID
The dataset can be found via this link: https://xiatian-zhu.github.io/downloads_qmul_iLIDS-VID_ReID_dataset.html
#### Features:
320 individuals captured across 2 disjoint camera views

600 image sequences (2 per person)

Variable sequence lengths (23-192 frames, average 73)
## Challenges:
Clothing similarities

Lighting and viewpoint variations

Background clutter

MARS
## Overview:
1261 unique pedestrians across 6 cameras

Extension of Market-1501 with more challenging aspects
## Features:
Frames captured from synchronized cameras for temporal analysis

Emphasis on motion analysis with tracklets

Over 800k images in the training set alone
## Data Augmentation
Random changes in brightness, contrast, and saturation of images

Operates in the HSV color space for intuitive color manipulation

## Training Parameters
### ResNet-50:
Trained for 50 epochs with batches of 32 images
### Similarity Network:
Trained using both Adam and SGD optimizers for 500 epochs

Adam: Learning rate of 0.0001

SGD: Learning rate of 0.01 with momentum of 0.9 and decay of 0.0005
