import glob
import os
import json

import cv2
import numpy as np
import tensorflow as tf
import torch
from keras import layers, models
from yolov5 import detect
from PIL import Image

def Load_File():
    #DataSet_Directory = "D:/Projects/Person re Identification/Datasets/CNN Dataset/train2017"
#
    #data = glob.glob(os.path.join(DataSet_Directory, "*.jpg"))
    #print(data[1])
    #np.random.shuffle(data)  # Shuffle the image paths list
    #print(data[1])
    #DataSet_Directory = "D:/Projects/Person re Identification/Datasets/CNN Dataset/train2017"
#
    #x_train = []
    #for file in os.listdir(DataSet_Directory):
    #    image_values = cv2.imread(DataSet_Directory + '/' + file)
    #    x_train.append(image_values)
#
    #print(x_train[1])
    #print(x_train[1].shape)
#
    # batch_paths = data[index * 32:(index + 1) * 32]


    Json_Directory = "D:/Projects/Person re Identification/Datasets/CNN Dataset/annotations"
    Json_File = "instances_train2017.json"

    Json_Path = os.path.join(Json_Directory, Json_File)
    image_ids = []

    with open(Json_Path) as f:
        data = json.load(f)
        print(data)
        print(data['images'])
        print(data['annotations'])
    #    for i in data['images']:
    #        image_ids.append(i['id'])
    #f.close()
    #with open(Json_Path) as f:
    #    data = json.load(f)
    #    for i in data['annotations']:
    #return data


def Test():
    DataSet_Directory = "D:/Projects/Person re Identification/Datasets/CNN Dataset/train2017"
    DataSet_File = '000000000764.jpg'

    DataSet_Path = os.path.join(DataSet_Directory, DataSet_File)

    # Load the model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Load test image
    image = Image.open(DataSet_Path)

    # Perform inference
    results = model(image)

    # Visualize results (optional)
    results.save()
    results.print()
    person = results.pandas().xyxy[0][results.pandas().xyxy[0]["name"] == "person"]
    print(person)

    img = cv2.imread(DataSet_Path)
    # Draw bounding boxes and confidence scores on the image


    for person in results.pandas().xyxy[0][results.pandas().xyxy[0]["name"] == "person"].iterrows():
        person_box = person[1]
        start_point = (int(person_box["xmin"]), int(person_box["ymin"]))
        end_point = (int(person_box["xmax"]), int(person_box["ymax"]))
        confidence = round(person_box["confidence"], 2)
        cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
        cv2.putText(img, f"Person: {confidence}", start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Android_cam", img)
    #    # Extract bounding box coordinates
    #    person_box = person[1]
#
    #    # Crop the image for this person
    #    cropped_image = image.crop((person_box["xmin"], person_box["ymin"], person_box["xmax"], person_box["ymax"]))
#
    #    # Save the cropped image (optional)
    #    # You can save each cropped image with a unique identifier or based on the person's confidence score
    #    cropped_image.save(f"cropped_person_{person[0]}.jpg")






def Find_Image_Info():
    Json_Directory = "D:/Projects/Person re Identification/Datasets/CNN Dataset/annotations"
    Json_File = "instances_train2017.json"
    Json_Path = os.path.join(Json_Directory, Json_File)

    DataSet_Directory = "D:/Projects/Person re Identification/Datasets/CNN Dataset/train2017"
    DataSet_File = '000000000785.jpg'
    DataSet_Path = os.path.join(DataSet_Directory, DataSet_File)
    a = 0

    # Load Json file Test
    with open(Json_Path) as f:
        data = json.load(f)
        for i in data['images']:
            a += 1
            # if i['file_name'] in DataSet_File:
            id = i['id']
            print("For Image ID: ", id)
            for j in data['annotations']:
                if j['image_id'] == id:
                    print(j)
                    print(i)
                    break

            # temp
            if a == 100:
                break


def CNN(train_images, train_labels, val_images, val_labels, test_images):
    # Import libraries

    # Define input layer for images
    input_image = tf.keras.Input(shape=(224, 224, 3))

    # Convolutional layers for feature extraction
    conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(input_image)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Flatten the output of the convolutional layers
    flattened = tf.keras.layers.Flatten()(pool2)

    # Dense layers for classification
    dense1 = layers.Dense(128, activation="relu")(flattened)
    dense2 = layers.Dense(64, activation="relu")(dense1)

    # Output layer for binary classification (human vs. not human)
    output = layers.Dense(1, activation="sigmoid")(dense2)

    # Define the model
    model = models.Model(inputs=input_image, outputs=output)

    # Compile the model with optimizer and loss function
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model on your dataset (images with human/not human labels)
    model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

    # Use the trained model to predict the presence of humans in new images
    predictions = model.predict(test_images)


if __name__ == '__main__':
    data = Test()
