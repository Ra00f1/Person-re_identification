import os
import cv2
import numpy as np
import tensorflow as tf


def color_jitter(image, brightness=0.4, hue=0.1, saturation=0.4):
  """
  Applies random color jittering to an image.

  Args:
    image: The image to be jittered. (numpy array)
    brightness: The range of brightness jitter. (float)
    contrast: The range of contrast jitter. (float)
    saturation: The range of saturation jitter. (float)

  Returns:
    The jittered image. (numpy array)
  """

  # Convert image to HSV color space
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # Randomly adjust brightness, contrast, and saturation
  hue = np.random.uniform(-hue, hue)
  sat = np.random.uniform(1 - saturation, 1 + saturation)
  val = np.random.uniform(1 - brightness, 1 + brightness)

  h = hsv[:, :, 0] + hue
  s = hsv[:, :, 1] * sat
  v = hsv[:, :, 2] * val

  # Clip values to valid range
  h = np.clip(h, 0, 255)
  s = np.clip(s, 0, 255)
  v = np.clip(v, 0, 255)

  # Convert back to BGR color space
  hsv[:, :, 0] = h
  hsv[:, :, 1] = s
  hsv[:, :, 2] = v
  image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

  return image


def Triplet_Generator(Labels, Inputs, Batch_Size=128):
    print("Generating data...")
    data = []
    labels = len(Labels)

    for i in range(Batch_Size):

        # Selecting a random image from the dataset
        anchor_index = np.random.randint(0, labels)
        anchor = cv2.imread(Labels[anchor_index][2] + '/' + Labels[anchor_index][1])
        anchor = cv2.resize(anchor, (224, 224))
        anchor = color_jitter(anchor)
        # anchor_image = anchor
        anchor = tf.convert_to_tensor(anchor)

        # Selecting a random image from the dataset
        person_name = Labels[anchor_index][0]
        positive_person_list = []
        negative_person_list = []
        for j in Inputs:
            if j[0] == person_name:
                positive_person_list.append(j)
            else:
                negative_person_list.append(j)

        # Selecting a random positive image from the dataset
        positive_index = np.random.randint(0, len(positive_person_list))
        positive = cv2.imread(positive_person_list[positive_index][2] + '/' + positive_person_list[positive_index][1])
        positive = cv2.resize(positive, (224, 224))
        positive = color_jitter(positive)
        # positive_image = positive
        positive = tf.convert_to_tensor(positive)

        # Selecting a random negative image from the dataset
        negative_index = np.random.randint(0, len(negative_person_list))
        negative = cv2.imread(negative_person_list[negative_index][2] + '/' + negative_person_list[negative_index][1])
        negative = cv2.resize(negative, (224, 224))
        negative = color_jitter(negative)
        # negative_image = negative
        negative = tf.convert_to_tensor(negative)

        data.append([anchor, positive, negative])

    print("Data generated successfully")

    # print(data[0])
    # cv2.imshow("Anchor", anchor_image)
    # cv2.imshow("Positive", positive_image)
    # cv2.imshow("Negative", negative_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return data


# noinspection PyUnresolvedReferences
def Read_Image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
    return image


if __name__ == '__main__':
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
    # print(Input)

    data = Triplet_Generator(Labels, Input, 200)
