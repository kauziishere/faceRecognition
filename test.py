from model import *
from prepareData import *
import numpy as np
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

def iterate_over_all_values(test_image, load_weight):
    onet = model_fin()
    data = 0
    with open("train_data.pkl", "rb") as f:
        data = pickle.load(f)
    max = 999
    onet.load_weights(load_weight)
    lab = 0
    for label in data.keys():
        data[label]  = np.array(data[label]).astype("float32")
        data[label] /= 255
    test_image = test_image.astype("float32")
    test_image /= 255
    test_image = test_image.reshape(1, test_image.shape[0], test_image.shape[1], 1)
    for label in data.keys():
        for i in range(0, data[label].shape[0]):
            x = np.array(data[label][i])
            x = x.reshape(1, x.shape[0], x.shape[1], 1)
            val = onet.predict([x, test_image])
            if(val[0] < max):
                max = val[0]
                lab = label
    return lab

def check(load_weights, labels):
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        scale_percent = 30
        width         = int(frame.shape[1]*(scale_percent/100))
        height        = int(frame.shape[0]*(scale_percent/100))
        imag          = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(imag, (x, y), (x+w, y+h), (0, 255, 0), 2)
            crop_image = imag[y:y+h, x:x+w]
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (105, 105))
            label = iterate_over_all_values(gray, load_weights)
            cv2.putText(imag, label,(y, x),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(225,225,225))

        cv2.imshow('Video', imag)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


def start(load_weights, labels):
    video_capture = cv2.VideoCapture(0)
    flag = 0
    while True:
        ret, image = video_capture.read()
        scale_percent = 30
        width         = int(image.shape[1]*(scale_percent/100))
        height        = int(image.shape[0]*(scale_percent/100))
        imag          = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
        gray          = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
        faces         = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 8, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
        print(faces)
        for (x, y, w, h) in faces:
            crop_image = image[y:y+h, x:x+w]
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (105, 105))
            flag = 1
            print("in the function")
            print(iterate_over_all_values(gray, labels))

if __name__ == "__main__":
    labels = dict(zip([0, 1, 2, 3, 4], ["Aniket", "Kauzi", "Nagnath", "Priyam", "Shubham"]))
    check("weights.h5", labels)
