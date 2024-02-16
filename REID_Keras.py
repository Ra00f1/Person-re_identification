import os
import random
from collections import defaultdict

import cv2
import numpy as np
from keras import layers, models, optimizers
from keras.applications import ResNet50
import tensorflow as tf
from keras.src.initializers.initializers import GlorotUniform
from matplotlib import pyplot as plt
import i_LIDS_VID  # Custom iLIDS-VID dataset class
import tripletloss  # Custom triplet loss function (Couldn't use the built-in ones)
from keras.applications.resnet50 import preprocess_input
import visualkeras

"""
This is the main file for the Person Re-Identification project. The main goal of this project is to create a model that
can identify a person from a set of images. The dataset used for this project is the iLIDS-VID dataset. The dataset
contains images of people from two different cameras. The images are taken from different angles and at different times.
The model is trained using the triplet loss function. The model is trained using the ResNet50 model as the base model.
The model is trained using the Keras library. 

The model is trained using the Adam(lr = 0.0001) optimizer and SGD(learning_rate=0.01, momentum=0.9, decay=0.0005) 
However the overall performance of the model is the same in both cases which is overfitting and I suspect the reason 
is the fact that the data for the Similarity model is only generated once while for Resnet50 it is created randomly 
for each batch.(Will be fixed later) 
"""


# The Feature Extractor model class with RestNet50 as the base model
class FeatureExtractor(models.Model):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False  # Freeze base model layers
        self.base_model = base_model
        # Add fully connected layers to the base model
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(224, activation='relu')
        self.fc3 = layers.Dense(128)  # Output embedding size is 128

    # Forward pass
    def call(self, x):
        x = self.base_model(x)
        x = self.fc1(x)
        x = self.fc2(x)  # Pass through fc2 before the final embedding layer
        embeddings = self.fc3(x)  # Final embeddings from fc3
        return embeddings


# USed to create the Label and Input images for the iLIDS-VID dataset
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


def Start_Train_resnet50(batch_size=32, epoch=1, batch_count=10):
    print("Starting Training of Resnet50")

    # Create a model using the FeatureExtractor class
    model = FeatureExtractor()

    # The loss function is triplet loss with Adam optimizer
    triplet_loss = tripletloss.TripletLoss(margin=0.3)
    optimizer = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=triplet_loss, optimizer=optimizer)

    # Unfreeze the last two layers of the base model
    layers = [model.fc1, model.fc2]

    # Set the weights of the unfrozen layers to GlorotUniform initializer
    for layer in layers:
        layer.trainable = True
        layer.set_weights([GlorotUniform(shape) for shape in layer.get_weights()])
    model.build((None, 224, 224, 3))  # Necessary for the summary to work

    model.summary()  # Keras equivalent of summary

    Labels, Inputs = Label_Input_Generator()

    model = Train_resnet50(model, triplet_loss, optimizer, batch_size, epoch, batch_count, Labels, Inputs)

    Save_Model(model, "Resnet50")


# noinspection PyShadowingNames
def Train_resnet50(model, triplet_loss, optimizer, batch_size, epoch, batch_count, Labels, Inputs):
    print("Starting Training of Resnet50")
    for epoch in range(epoch):
        print(f"Epoch {epoch + 1}")
        for i in range(batch_count):

            # Generate a batch of triplets
            batch = i_LIDS_VID.Triplet_Generator(Labels, Inputs, batch_size)
            # j is ussed to keep track of the number of images in the batch
            for image in batch:
                # Convert images to tensors using Keras preprocessing
                anchor_tensor = preprocess_input(image[0])
                anchor_tensor = tf.expand_dims(anchor_tensor, axis=0)

                positive_tensor = preprocess_input(image[1])
                positive_tensor = tf.expand_dims(positive_tensor, axis=0)

                negative_tensor = preprocess_input(image[2])
                negative_tensor = tf.expand_dims(negative_tensor, axis=0)

                # Because the Resnet50 model on this test works using batches a resistant GradientTape is used to
                # calculate the gradients
                with tf.GradientTape(persistent=True) as tape:
                    anchor_emb = model(anchor_tensor, training=True)
                    positive_emb = model(positive_tensor, training=True)
                    negative_emb = model(negative_tensor, training=True)
                    loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
                    # print(tape.gradient(loss, model.trainable_variables))

                # Calculate the gradients
                gradients = tape.gradient(loss, model.trainable_variables)

                # Delete the tape to free up memory as was said in the stackoverflow post and would generate an error
                # if not used
                # (https://stackoverflow.com/questions/56072634/tf-2-0-runtimeerror-gradienttape-gradient-can-only-be-called-once-on-non-pers)
                del tape

                # Apply the gradients to the model
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Use optimizer.minimize for Keras

        print(f"Epoch {epoch + 1} completed.")
    return model


# Save the model to the path provided using the name provided
def Save_Model(model, path):
    print("Saving Model")
    model.save(path, save_format="tf")
    print("Model Saved")


def Similarity_Model():

    # This is for new and untrained model(For testing purposes only)

    # base_model = FeatureExtractor()
    # triplet_loss = tripletloss.TripletLoss(margin=0.3)
    # optimizer = optimizers.Adam(learning_rate=0.0001)
    # base_model.compile(loss=triplet_loss, optimizer=optimizer)
    # layers = [base_model.fc1, base_model.fc2]
    # for layer in layers:
    #     layer.trainable = True
    #     layer.set_weights([GlorotUniform(shape) for shape in layer.get_weights()])
    # base_model.build((None, 224, 224, 3))  # Necessary for the summary to work
    # base_model.summary()  # Keras equivalent of summary

    # -------------------------------------------- Similarity Network --------------------------------------------

    # This is for new and untrained model

    # Create a Sequential model using Keras with the following layers
    model = tf.keras.Sequential()
    # Add the input layer with the shape of the input given by the Resnet50 model
    model.add(tf.keras.Input(shape=(2, 1, 7, 7, 128)))
    model.add(tf.keras.layers.Flatten())

    # Add the following layers to the model (not backed by any research and only for testing purposes)
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # optimizer = optimizers.Adam(learning_rate=0.0001)
    optimizer = optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', 'Precision', 'Recall'])
    model.build((None, 7, 7, 128, 2))
    model.summary()

    return model, optimizer

def Start_Train_Similarity(batch_count, batch_size, epoch):
    Labels, Inputs = Label_Input_Generator()

    # -------------------------------------------- Feature Extractor --------------------------------------------

    # Load the Feature Extractor model(Resnet50 in this case)
    base_model = tf.saved_model.load("Models/Resnet50")

    # -------------------------------------------- Data Creation -------------------------------------------------------
    images = []
    similarity_labels = []

    print("Creating Data")
    for i in range(batch_count):
        print(f"Batch {i + 1}")
        batchs = i_LIDS_VID.Triplet_Generator(Labels, Inputs, batch_size)
        for batch in batchs:
            anchor_tensor = preprocess_input(batch[0])
            anchor_tensor = tf.expand_dims(anchor_tensor, axis=0)
            anchor_features = base_model(anchor_tensor)
            # print(anchor_features.shape)

            positive_tensor = preprocess_input(batch[1])
            positive_tensor = tf.expand_dims(positive_tensor, axis=0)
            postive_features = base_model(positive_tensor)

            negative_tensor = preprocess_input(batch[2])
            negative_tensor = tf.expand_dims(negative_tensor, axis=0)
            negative_feature = base_model(negative_tensor)

            # randomly append positive or negative to the lists first
            if random.randint(0, 1) == 0:
                images.append([anchor_features, postive_features])
                similarity_labels.append(1)
                images.append([anchor_features, negative_feature])
                similarity_labels.append(0)
            else:
                images.append([anchor_features, negative_feature])
                similarity_labels.append(0)
                images.append([anchor_features, postive_features])
                similarity_labels.append(1)

    similarity_labels = np.array(similarity_labels)
    images = np.array(images)

    print(similarity_labels.shape)
    print(images.shape)

    #model, history = Train_Similarity(images, similarity_labels, batch_size, epoch)
    model, history = Train_Similarity_Manual(Labels, Inputs, batch_count, batch_size, epoch)
    # -------------------------------------------- Save Model -------------------------------------------------------
    Save_Model(model, "Similarity_Network_Adam100")

    # Draw plots for accuracy and loss
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('Accuracy.png')
    plt.show()


# This function is created to just make everything look clearer and easier to understand when presenting the project
def Train_Similarity(x_train, y_train, batch_size, epoch):
    model = Similarity_Model()
    history = model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, validation_split=0.2)

    return model, history


def Data_Generator_Similarity(batch_count, batch_size, Labels, Inputs):
    images = []
    similarity_labels = []

    # Load the Feature Extractor model(Resnet50 in this case)
    base_model = tf.saved_model.load("Models/Resnet50")

    print("Creating Data")
    for i in range(batch_count):
        print(f"Batch {i + 1}")
        batchs = i_LIDS_VID.Triplet_Generator(Labels, Inputs, batch_size)
        for batch in batchs:
            anchor_tensor = preprocess_input(batch[0])
            anchor_tensor = tf.expand_dims(anchor_tensor, axis=0)
            anchor_features = base_model(anchor_tensor)
            # print(anchor_features.shape)

            positive_tensor = preprocess_input(batch[1])
            positive_tensor = tf.expand_dims(positive_tensor, axis=0)
            postive_features = base_model(positive_tensor)

            negative_tensor = preprocess_input(batch[2])
            negative_tensor = tf.expand_dims(negative_tensor, axis=0)
            negative_feature = base_model(negative_tensor)

            # randomly append positive or negative to the lists first
            if random.randint(0, 1) == 0:
                images.append([anchor_features, postive_features])
                similarity_labels.append(1)
                images.append([anchor_features, negative_feature])
                similarity_labels.append(0)
            else:
                images.append([anchor_features, negative_feature])
                similarity_labels.append(0)
                images.append([anchor_features, postive_features])
                similarity_labels.append(1)

    similarity_labels = np.array(similarity_labels)
    images = np.array(images)

    return images, similarity_labels


def Train_Similarity_Manual(Labels, Inputs, batch_count, batch_size, epoch):
    model, optimizer = Similarity_Model()
    history = defaultdict(list)     # Initialize history dictionary for losses and other metrics

    for i in range(epoch):
        x_train, y_train = Data_Generator_Similarity(batch_count, batch_size, Labels, Inputs)
        x_train = np.expand_dims(x_train, axis=0)
        print(x_train.shape)
        print(y_train.shape)
        for j in range(len(x_train)):
            with tf.GradientTape() as tape:
                output = model(x_train[j])
                print(type(output))
                output = tf.convert_to_tensor(output)
                output = [output]
                print(type(y_train))
                print(y_train)
                loss = tf.keras.losses.binary_crossentropy(y_train, output)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            history['loss'].append(loss)

            print(f"Epoch {i + 1} Batch {j + 1} Loss: {loss}")

    return model, history


# Show the model in a blockey way to visualize the layers of the model
def Visualize_Network(model):
    visualkeras.layered_view(model).show()  # display using your system viewer
    visualkeras.layered_view(model, to_file='output.png')  # write to disk
    visualkeras.layered_view(model, to_file='output.png').show()  # write and show

    visualkeras.layered_view(model)


# Function to Test the model with one set of images one anchor and one positive/negative image
def Test_REID(anchor, image, model_name):
    # load both Resnet50 and the Similarity Network as both are needed for the testing
    resnet50 = tf.saved_model.load("Resnet50")
    SimilarityNetwork = tf.keras.models.load_model(model_name)

    # changing the input to the correct format(tensor) and then getting the features from the Resnet50 model
    anchor_tensor = preprocess_input(anchor)
    # an extra dimension is added to the tensor to make it a 4D tensor(needed for the Resnet50 model to work)
    anchor_tensor = tf.expand_dims(anchor_tensor, axis=0)
    anchor_features = resnet50(anchor_tensor)

    # Doing the same things for the other image
    image_tensor = preprocess_input(image)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    image_features = resnet50(image_tensor)

    # The features are then sent to the Similarity Network to get the similarity between the two images
    # However first they need to be in the correct format(np.array)
    images = np.array([[anchor_features, image_features]])
    similarity = SimilarityNetwork.predict(images)
    print(similarity)

    # If the similarity is greater than 0.5 then the two images are the same person
    if similarity > 0.5:
        return True
    else:
        return False


# Function to batch test the model
def Start_Testing(model_name, Test_Number=1):
    # Load the LIDS-VID dataset
    Labels, Inputs = Label_Input_Generator()

    for i in range(Test_Number):
        print(f"Test Number: {i + 1}")
        # Get the batch of images
        batchs, anchor_image, positive_image, negative_image = i_LIDS_VID.Test_Triplet_Generator(Labels, Inputs, 1)

        # First show images with cv2 and then send them to the model
        for batch in batchs:
            anchor = batch[0]
            positive = batch[1]
            negative = batch[2]

            # Show anchor and positive images side by side
            combined_image = np.concatenate((anchor, positive), axis=0)
            cv2.imshow("Anchor vs. Positive", combined_image)

            # Print model output using the Test_REID function
            print("Anchor vs. Positive:", Test_REID(anchor, positive, model_name))

            # Show anchor and negative images side by side
            combined_image = np.concatenate((anchor, negative), axis=0)
            cv2.imshow("Anchor vs. Negative", combined_image)

            # Print model output using the Test_REID function
            print("Anchor vs. Negative:", Test_REID(anchor, negative, model_name))

            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Starting")

    # Start_Train_resnet50(batch_count=50, batch_size=32, epoch=50)

    Start_Train_Similarity(batch_count=10, batch_size=24, epoch=10)

    # Start_Testing(Test_Number=10, model_name="Similarity_Network_SGD2")
    # Start_Testing(Test_Number=10, model_name="Similarity_Network_Adam")