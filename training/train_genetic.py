
import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential

    
def main():
    
        # X_train = pd.read_pickle("X_train_vcf.pkl")
        # y_train = pd.read_pickle("y_train_vcf.pkl")

        # X_test = pd.read_pickle("X_test_vcf.pkl")
        # y_test = pd.read_pickle("y_test_vcf.pkl")


        with open("genetics_train/X_train_vcf.pkl",'rb') as f:
            X_train=np.load(f,allow_pickle=True)
        with open("genetics_train/y_train_vcf.pkl",'rb') as f:
            y_train=np.load(f,allow_pickle=True)


        acc = []
        f1 = []
        precision = []
        recall = []
        model = Sequential()
        # model.add(Dense(128, input_shape = (15965,), activation = "relu")) 
        # model.add(Dense(128, input_shape = (18975,), activation = "relu"))
        model.add(Dense(128, input_shape = (212444,), activation = "relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation = "relu"))
        model.add(Dropout(0.5))

        model.add(Dense(32, activation = "relu"))
        model.add(Dropout(0.3))

        model.add(Dense(32, activation = "relu"))
        model.add(Dropout(0.3))


        model.add(Dense(3, activation = "softmax"))

        model.compile(Adam(learning_rate = 0.000001), "sparse_categorical_crossentropy", metrics = ["sparse_categorical_accuracy"])


        model.fit(X_train, y_train,epochs=50,batch_size=32,validation_split = 0.1, verbose=1) 
        model.save("train_genetic_model.hdf5")
    

if __name__ == '__main__':
    main()
    
