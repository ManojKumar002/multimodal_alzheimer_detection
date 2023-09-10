
import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model


acc = []
f1 = []
precision = []
recall = []

def set_str(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if isinstance(dataset[i][j],str):
                dataset[i][j]=0

with open("clinical_train/X_train_c.pkl",'rb') as f:
    X_train=np.load(f,allow_pickle=True)
with open("clinical_train/y_train_c.pkl",'rb') as f:
    y_train=np.load(f,allow_pickle=True)
with open("clinical_train/X_test_c.pkl",'rb') as f:
    X_test=np.load(f,allow_pickle=True)
with open("clinical_train/y_test_c.pkl",'rb') as f:
    y_test=np.load(f,allow_pickle=True)

set_str(X_train)
set_str(X_test)

X_train=np.asarray(X_train).astype(np.float32)
y_train=np.asarray(y_train).astype(np.float32)
X_test=np.asarray(X_test).astype(np.float32)
y_test=np.asarray(y_test).astype(np.float32)

model =load_model("./saved_models/train_clinical_model_base.hdf5")

score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
acc.append(score[1])

test_predictions = model.predict(X_test)
test_label = to_categorical(y_test,3)

true_label= np.argmax(test_label, axis =1)

predicted_label= np.argmax(test_predictions, axis =1)

cr = classification_report(true_label, predicted_label, output_dict=True)
precision.append(cr["macro avg"]["precision"])
recall.append(cr["macro avg"]["recall"])
f1.append(cr["macro avg"]["f1-score"])


print("Accuracy: " + str(np.array(acc).mean()))
print("Precision: " + str(np.array(precision).mean()))
print("Recall: " + str(np.array(recall).mean()))
print("f1: " + str(np.array(f1).mean()))