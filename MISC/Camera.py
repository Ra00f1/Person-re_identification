# Import essential libraries
import requests
import cv2
import numpy as np
import imutils
import torch

if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
    url = "http://10.123.13.160:8080/shot.jpg"

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
            #img = cv2.cvtColor(img, cv2.COLOR_UNCHANGED)
            # Draw bounding boxes and confidence scores on the image
            for person in people.iterrows():
                person_box = person[1]
                start_point = (int(person_box["xmin"]), int(person_box["ymin"]))
                end_point = (int(person_box["xmax"]), int(person_box["ymax"]))
                confidence = round(person_box["confidence"], 2)
                cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
                cv2.putText(img, f"Person: {confidence}", start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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