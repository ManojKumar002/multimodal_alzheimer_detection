import os
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import pickle5 as pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Dense,Dropout,MaxPooling2D, Flatten, Conv2D



def main():
    
    with open("images_train/img_train.pkl", "rb") as fh:
        data = pickle.load(fh)
    X_train_ = pd.DataFrame(data)["img_array"] 
    
    with open("images_train/img_test.pkl", "rb") as fh:
        data = pickle.load(fh)
    X_test_ = pd.DataFrame(data)["img_array"]
    
    with open("images_train/img_y_train.pkl", "rb") as fh:
        data = pickle.load(fh)
    y_train = np.array(pd.DataFrame(data)["label"].values.astype(np.float32)).flatten()
    
    with open("images_train/img_y_test.pkl", "rb") as fh:
        data = pickle.load(fh)
    y_test = np.array(pd.DataFrame(data)["label"].values.astype(np.float32)).flatten()
    

    y_test[y_test == 2] = -1
    y_test[y_test == 1] = 2
    y_test[y_test == -1] = 1
    
    y_train[y_train == 2] = -1
    y_train[y_train == 1] = 2
    y_train[y_train == -1] = 1
    

    X_train = []
    X_test = []
    
    for i in range(len(X_train_)):
        X_train.append(X_train_.values[i])
        
    for i in range(len(X_test_)):
        X_test.append(X_test_.values[i])
    
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)


    # (3909, 1, 72, 72, 3) //original shape
    X_train= X_train.reshape(-1,72,72,3)
    #modified shape (3909,72, 72, 3)

    # (205, 1, 72, 72, 3) //original shape
    X_test= X_test.reshape(-1,72,72,3)
    #modified shape (205,72, 72, 3)


    acc = []
    f1 = []
    precision = []
    recall = []
    model = Sequential()
    model.add(Conv2D(100, (3, 3),  activation='relu', input_shape=(72, 72, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(50, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(3, activation = "softmax"))
    
    
    model.compile(Adam(learning_rate = 0.00001), "sparse_categorical_crossentropy", metrics = ["sparse_categorical_accuracy"])
    
    model.summary()
    model.fit(X_train, y_train, epochs=1500, batch_size=32,validation_split=0.1, verbose=1) 
    model.save("train_images_model.hdf5")
        
    
if __name__ == '__main__':
    main()
    
    
