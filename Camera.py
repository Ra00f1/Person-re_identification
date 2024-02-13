# Import essential libraries
import requests
import cv2
import numpy as np
import imutils
import torch
import tensorflow as tf
import REID_Keras as reid

if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
    url = "http://192.168.1.128:8080/shot.jpg"

    # While loop to continuously fetching data from the Url
    while True:
        # Fetch image from camera
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)

        # Resize image for better performance (optional)
        img = imutils.resize(img, width=650, height=850)

        # Perform YOLOv5 inference
        results = model(img)
        results.print()
        # Extract detected person information (bounding boxes and confidence scores)

        try:
            people = results.pandas().xyxy[0][results.pandas().xyxy[0]["name"] == "person"]

            # Check if Key "S" is pressed save the image to anchor after resizing it to 224x224
            if cv2.waitKey(1) & 0xFF == ord('s') and people is not None and len(people) > 0:
                for person in results.pandas().xyxy[0][results.pandas().xyxy[0]["name"] == "person"].iterrows():
                    temp_anchor = person[1]
                    x1, y1, x2, y2 = int(temp_anchor['xmin']), int(temp_anchor['ymin']), int(temp_anchor['xmax']), int(temp_anchor['ymax'])
                    cropped_image = img[y1:y2, x1:x2]
                    temp_anchor = cv2.resize(cropped_image, (224, 224))
                    anchor = temp_anchor
                    cv2.imwrite("anchor.jpg", anchor)
                    anchor = np.array(anchor)
                    print(anchor.shape)

                    anchor = anchor / 255  # normalize
                    anchor = anchor * 255
                    anchor = anchor.astype(np.uint8)

                    # add a new axis

                    print(anchor.shape)
                    break

            # Draw bounding boxes and confidence scores on the image
            elif anchor is not None:
                for person in results.pandas().xyxy[0][results.pandas().xyxy[0]["name"] == "person"].iterrows():
                    temp_person = person[1]
                    x1, y1, x2, y2 = int(temp_person['xmin']), int(temp_person['ymin']), int(temp_person['xmax']), int(temp_person['ymax'])
                    cropped_image = img[y1:y2, x1:x2]
                    temp_person = cv2.resize(cropped_image, (224, 224))
                    person_np = np.array(temp_person)
                    print(person_np.shape)

                    person_np = person_np / 255  # normalize
                    person_np = person_np * 255
                    person_np = person_np.astype(np.uint8)

                    # add a new axis
                    print(person_np.shape)

                    # Compare the anchor with the person
                    similarity = reid.Test_REID(anchor, person_np)

                    # If the person is the same as the anchor, draw a green bounding box
                    if similarity > 0.5:
                        start_point = (int(person[1]["xmin"]), int(person[1]["ymin"]))
                        end_point = (int(person[1]["xmax"]), int(person[1]["ymax"]))
                        cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
                        cv2.putText(img, f"Person: {round(similarity, 2)}", start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # If the person is not the same as the anchor, draw a red bounding box
                    else:
                        start_point = (int(person[1]["xmin"]), int(person[1]["ymin"]))
                        end_point = (int(person[1]["xmax"]), int(person[1]["ymax"]))
                        cv2.rectangle(img, start_point, end_point, (0, 0, 255), 2)
                        cv2.putText(img, f"Person: {round(similarity, 2)}", start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Display the image with detections
                    cv2.imshow("Android_cam", img)

        except Exception as e:
            print(e)
            cv2.imshow("Android_cam", img)

            # Display the image with detections

        # Press Esc key to exit
        if cv2.waitKey(1) == 27:
            break

    # Clean up resources
    cv2.destroyAllWindows()