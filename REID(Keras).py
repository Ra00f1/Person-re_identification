import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models, optimizers
from keras.applications import ResNet50
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.src.initializers.initializers import GlorotUniform
import i_LIDS_VID
import tripletloss
from keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split


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
        self.fc3 = layers.Dense(128)     # Output embedding size is 128


    def call(self, x):
        x = self.base_model(x)
        #if isinstance(x, list):
        #    x = x[0]  # Access the first tensor if it's a list
        #x = layers.Flatten()(x)
        x = self.fc1(x)
        x = self.fc2(x)  # Pass through fc2 before the final embedding layer
        embeddings = self.fc3(x)  # Final embeddings from fc3
        return embeddings


class SimilarityNetwork(models.Model):
    def __init__(self):
        super(SimilarityNetwork, self).__init__()
        self.flatten = layers.Flatten()
        self.merged = layers.Concatenate()  # Concatenate embeddings
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')  # Output confidence

    def call(self, inputs):
        # print(inputs)
        # inputs = tf.reshape(inputs, (-1, 256))
        print(inputs)
        test = self.flatten(inputs)
        print(test)
        anchor_embedding = inputs[0]  # Access individual tensors
        other_embedding = inputs[1]

        # Flatten if necessary (but keep them as separate tensors)
        anchor_embedding = self.flatten(anchor_embedding)
        print(anchor_embedding)
        other_embedding = self.flatten(other_embedding)

        # Concatenate the tensors
        merged = self.merged([anchor_embedding, other_embedding])
        print(merged)
        x = self.fc1(merged)
        x = self.fc2(x)
        confidence = self.output_layer(x)
        return confidence


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


def Start_Train_resnet50():
    print("Starting Training of Resnet50")

    model = FeatureExtractor()
    triplet_loss = tripletloss.TripletLoss(margin=0.3)
    optimizer = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=triplet_loss, optimizer=optimizer)
    layers = [model.fc1, model.fc2]
    for layer in layers:
        layer.trainable = True
        layer.set_weights([GlorotUniform(shape) for shape in layer.get_weights()])
    model.build((None, 224, 224, 3))  # Necessary for the summary to work

    model.summary()  # Keras equivalent of summary

    batch_size = 32
    epoch, batch_count = 1, 10
    Labels, Inputs = Label_Input_Generator()

    model = Train(model, triplet_loss, optimizer, batch_size, epoch, batch_count, Labels, Inputs)
    model.save("my_model", save_format="tf")

    print("Model Saved")


# noinspection PyShadowingNames
def Train(model, triplet_loss, optimizer, batch_size, epoch, batch_count, Labels, Inputs):
    for epoch in range(epoch):
        print(f"Epoch {epoch + 1}")
        for i in range(batch_count):
            batch = i_LIDS_VID.Triplet_Generator(Labels, Inputs, batch_size)
            j = 0
            # ... (loop structure remains the same)
            for image in batch:
                print("Batch: ", j+1)
                j += 1

                # ... (data loading remains the same)

                # Convert images to tensors using Keras preprocessing
                anchor_tensor = preprocess_input(image[0])
                anchor_tensor = tf.expand_dims(anchor_tensor, axis=0)

                positive_tensor = preprocess_input(image[1])
                positive_tensor = tf.expand_dims(positive_tensor, axis=0)

                negative_tensor = preprocess_input(image[2])
                negative_tensor = tf.expand_dims(negative_tensor, axis=0)

                # x_train.append([anchor_tensor, positive_tensor])
                # y_train.append(1)
                # x_train.append([anchor_tensor, negative_tensor])
                # y_train.append(0)

                with tf.GradientTape(persistent=True) as tape:
                    anchor_emb = model(anchor_tensor, training=True)
                    positive_emb = model(positive_tensor, training=True)
                    negative_emb = model(negative_tensor, training=True)
                    loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
                    # print(tape.gradient(loss, model.trainable_variables))

                gradients = tape.gradient(loss, model.trainable_variables)
                del tape

                # print(gradients)
                # print(loss)

                optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Use optimizer.minimize for Keras

            # ... (loss calculation and backpropagation remain the same)
            # print(f"Batch {j + 1} completed. Loss: {loss.numpy():.4f}")
        # ... (logging remains the same)
        # print(f"Epoch {epoch + 1} completed. Loss: {loss.numpy():.4f}")
    return model


# noinspection PyShadowingNames
def Similarity_Layer():
    model = SimilarityNetwork()
    optimizer = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', 'Precision', 'Recall'])
    model.build(input_shape=[(None, 128, 64, 3), (None, 128, 64, 3)])
    model.summary()

    return model


def Train_Similarity(model, x_train, y_train):
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    return model, history


def Last_Layer():

    Labels, Inputs = Label_Input_Generator()
    x_train = []        # These are for the third network
    y_train = []
    print("Starting Similarity Network")
    Data_Count = 100                            # Total data count will be 2*Data_Count

    feature_extractor = FeatureExtractor()      #TODO: Load the model from the file
    for i in range(Data_Count):                 # This will create two inputs
        batch = i_LIDS_VID.Triplet_Generator(Labels, Inputs, 1)

        anchor_tensor = preprocess_input(batch[0])
        anchor_tensor = tf.expand_dims(anchor_tensor, axis=0)

        positive_tensor = preprocess_input(batch[1])
        positive_tensor = tf.expand_dims(positive_tensor, axis=0)

        negative_tensor = preprocess_input(batch[2])
        negative_tensor = tf.expand_dims(negative_tensor, axis=0)

        anchor_emb = feature_extractor(anchor_tensor)
        positive_emb = feature_extractor(positive_tensor)
        negative_emb = feature_extractor(negative_tensor)

        x_train.append([anchor_emb, positive_emb])
        y_train.append(1)
        x_train.append([anchor_emb, negative_emb])
        y_train.append(0)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    print("Data Generated")

    similarity_model = Similarity_Layer()
    similarity_model, history = Train_Similarity(similarity_model, x_train, y_train)

    similarity_model.save("D:\Projects\Person re Identification\Similarity_model.h5")
    print("Similarity Model Saved")

    print("Finished")

    print("Visualizing")

    # Visualize the training process
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.legend()
    plt.show()

    plt.plot(history.history['recall'], label='Training Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.legend()
    plt.show()


def Temp():
    print("Testing")
    Labels, Inputs = Label_Input_Generator()

    testbatch = i_LIDS_VID.Triplet_Generator(Labels, Inputs, 1)

    # model = tf.saved_model.load("my_model")
    model = FeatureExtractor()

    X, Y = [], []
    train_size = 10

    for i in range(train_size):
        if i % 100 == 0:
            print(i)
        batchs = i_LIDS_VID.Triplet_Generator(Labels, Inputs, 1)
        for batch in batchs:
            anchor_tensor = preprocess_input(batch[0])
            anchor_tensor = tf.expand_dims(anchor_tensor, axis=0)

            positive_tensor = preprocess_input(batch[1])
            positive_tensor = tf.expand_dims(positive_tensor, axis=0)

            negative_tensor = preprocess_input(batch[2])
            negative_tensor = tf.expand_dims(negative_tensor, axis=0)

            anchor_emb = model(anchor_tensor)
            positive_emb = model(positive_tensor)
            negative_emb = model(negative_tensor)

            X.append([anchor_emb, positive_emb])
            Y.append(1)
            X.append([anchor_emb, negative_emb])
            Y.append(0)

    X = np.array(X)
    Y = np.array(Y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    print("Data Generated")

    similarity_model = Similarity_Layer()
    similarity_model, history = Train_Similarity(similarity_model, x_train, y_train)

    similarity_model.save("D:\Projects\Person re Identification\Similarity_model.h5")
    print("Similarity Model Saved")

    print("Finished")


if __name__ == '__main__':
    print("Starting")

    Labels, Inputs = Label_Input_Generator()


    print("Finished")