import os
import json

import cv2
from pycocotools import coco as COCO


def Load_File():
    data = {}
    return data

def Find_Image_Info():
    Json_Directory = "D:/Projects/Person re Identification/Datasets/CNN Dataset/annotations"
    Json_File = "instances_train2017.json"
    Json_Path = os.path.join(Json_Directory, Json_File)

    DataSet_Directory = "D:/Projects/Person re Identification/Datasets/CNN Dataset/train2017"
    DataSet_File = '000000000009.jpg'
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

if __name__ == '__main__':
    data = Load_File()
