import keras
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Input, Lambda, merge
from keras.models import Sequential, Model
import random
import numpy as np
import keras.backend as K
def base_model(input_dim):
    onet = Sequential()
    onet.add(Conv2D(64, (10, 10), activation = 'relu', input_shape = input_dim))
    onet.add(MaxPooling2D(pool_size = (2, 2)))
    onet.add(Conv2D(128, (7, 7), activation = 'relu'))
    onet.add(MaxPooling2D(pool_size = (2, 2)))
    onet.add(Conv2D(128, (4, 4), activation = 'relu'))
    onet.add(MaxPooling2D(pool_size = (2, 2)))
    onet.add(Conv2D(128, (4, 4)))
    onet.add(Flatten())
    onet.add(Dense(4096, activation = 'sigmoid'))
    return onet

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def create_pairs(data, y):
    pairs  = []
    labels = []
    min_cnt = 3
    for i in range(0, len(y)):
        for j in range(0, min_cnt-1):
            pairs += [[data[y[i]][j], data[y[i]][j+1]]]
            inc    = random.randrange(0, len(y)-1)
            dn     = (j + inc)%len(y)
            pairs += [[data[y[i]][j], data[y[dn]][j]]]
            labels+= [0, 1]
    return np.array(pairs), np.array(labels)

def model_fin():
    input_a     = (105, 105, 1)
    input_b     = (105, 105, 1)
    model       = base_model(input_a)
    input_mdl_a = Input(input_a)
    input_mdl_b = Input(input_b)
    processed_a = model(input_mdl_a)
    processed_b = model(input_mdl_b)
    L1_distance = lambda x: keras.backend.abs(x[0]-x[1])
    merge_step  = merge([processed_a,processed_b], mode = L1_distance, output_shape=lambda x: x[0])
    predict     = Dense(1, activation = 'sigmoid')(merge_step)
    final_model = Model(input=[input_mdl_a, input_mdl_b], output = predict)
    return final_model

if __name__ == "__main__":
    onet =  base_model((105, 105, 1))
    onet.summary()
