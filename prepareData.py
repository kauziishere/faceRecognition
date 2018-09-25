import cv2
import sys
import os
import numpy as np
import time
import pickle
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

def preprocess_data(filename, labels):
    num_images = len([_ for _ in os.listdir(filename)])
    data = np.zeros((num_images, 105, 105))
    outputs = dict(zip(labels ,[[] for _ in range(0, len(labels))]))
    print(outputs)
    loc = 0
    for file in os.listdir(filename):
        image = cv2.imread(filename+'/'+file)
        scale_percent = 20
        width = int(image.shape[1]*(scale_percent/100))
        height = int(image.shape[0]*(scale_percent/100))
        image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces 		= faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors = 8, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            crop_image = image[y:y+h, x:x+w]
            gray  = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(filename+"2/"+file, gray)
            gray = cv2.resize(gray, (105, 105))
            outputs[file.split(".")[0]].append(gray)
    with open("train_data.pkl", "wb") as f:
        pickle.dump(outputs, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    labels = ["arshad", "kauzi", "rahul", "priyam"]
    preprocess_data("Train_image", labels)
#    video_capture.release()
#    cv2.destroyAllWindows()
