from prepareData import *
from model import *
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
def train():
    data = 0
    with open("train_data.pkl", "rb") as f:
        data = pickle.load(f)
    y       = [x for x in data.keys()]
    X, y = create_pairs(data, y)
    print(X, y)
    X       = X.astype("float32")
    X   /= 255
    X    = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3] , 1)
    onet = model_fin()
    onet.compile(loss = contrastive_loss, optimizer = "adam")
    onet.fit([X[:, 0], X[:, 1]], y, batch_size = 32, epochs = 20)
    onet.save_weights("weights.h5")

if __name__ == "__main__":
    train()
