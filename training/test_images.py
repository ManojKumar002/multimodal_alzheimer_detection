import tensorflow as tf
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle5 as pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report



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

model =load_model("./train_images_model.hdf5")
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

print("Avg accuracy: " + str(np.array(acc).mean()))
print("Avg precision: " + str(np.array(precision).mean()))
print("Avg recall: " + str(np.array(recall).mean()))
print("Avg f1: " + str(np.array(f1).mean()))
print("Std accuracy: " + str(np.array(acc).std()))
print("Std precision: " + str(np.array(precision).std()))
print("Std recall: " + str(np.array(recall).std()))
print("Std f1: " + str(np.array(f1).std()))
print(acc)
print(precision)
print(recall)
print(f1)