import os

import cv2
import torch
import torch.nn as nn


if __name__ == '__main__':
    Output_path = "D:\Projects\Person re Identification\Datasets\iLIDS-VID\i-LIDS-VID\images"
    Number_of_Cams = 2

    Input_path = "D:\Projects\Person re Identification\Datasets\iLIDS-VID\i-LIDS-VID\sequences"

    Labels = []
    Input = []

    for cam in range(1, Number_of_Cams + 1):
        Cam_path = os.path.join(Output_path, "cam" + str(cam))
        for person in os.listdir(Cam_path):
            Person_path = os.path.join(Cam_path, person)
            for image in os.listdir(Person_path):
                Labels.append([person, image, Person_path])
    #print(Labels)

    for cam in range(1, Number_of_Cams + 1):
        Cam_path = os.path.join(Input_path, "cam" + str(cam))
        for person in os.listdir(Cam_path):
            Person_path = os.path.join(Cam_path, person)
            for image in os.listdir(Person_path):
                Input.append([person, image, Person_path])
    #print(Input)

    a = 0
    list = []
    for j in Labels:
        for i in Input:
            if i[0] != j[0]:

                if a <= 10:
                    if i[1] == ".DS_Store":
                        continue
                    print(i[2] + '/' + i[1])
                    anchor = cv2.imread(i[2] + '/' + i[1])
                    positive = cv2.imread(j[2] + '/' + j[1])
                    anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)
                    positive = cv2.cvtColor(positive, cv2.COLOR_BGR2RGB)
                    anchor = torch.from_numpy(anchor)
                    positive = torch.from_numpy(positive)
                    anchor = anchor.permute(2, 0, 1)
                    positive = positive.permute(2, 0, 1)
                    anchor = anchor.unsqueeze(0)
                    positive = positive.unsqueeze(0)
                    distance = torch.nn.functional.pairwise_distance(anchor, positive, p=2)
                    print(f"Distance between {i[2] + '/' + i[1]} and reference: {distance}")
                    list.append(distance)

                    a += 1




    # print("--------------------------------------------------")
    # print("Referance vs Referance")
    # print("--------------------------------------------------")
    # j = Labels[0]
    # anchor = cv2.imread(j[2] + '/' + j[1])
    # positive = cv2.imread(j[2] + '/' + j[1])
    # anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)
    # positive = cv2.cvtColor(positive, cv2.COLOR_BGR2RGB)
    # anchor = torch.from_numpy(anchor)
    # positive = torch.from_numpy(positive)
    # anchor = anchor.permute(2, 0, 1)
    # positive = positive.permute(2, 0, 1)
    # anchor = anchor.unsqueeze(0)
    # positive = positive.unsqueeze(0)
#
    # print(
    #     f"Distance between {j[2] + '/' + j[1]} and reference: {torch.nn.functional.pairwise_distance(anchor, positive, p=2)}")





