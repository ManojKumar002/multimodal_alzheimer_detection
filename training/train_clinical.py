import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential



def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


#extra code added to clear the string(date) which is there in between
def set_str(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if isinstance(dataset[i][j],str):
                dataset[i][j]=0



def main():
    #this is created in the clinical preprocess jupyter notebook
    # X_train = pd.read_pickle("clinical_train/X_train_c.pkl")
    # y_train = pd.read_pickle("clinical_train/y_train_c.pkl")

    # X_test = pd.read_pickle("clinical_train/X_test_c.pkl")
    # y_test = pd.read_pickle("clinical_train/y_test_c.pkl")


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
    # cnt=0
    # for i in range(len(X_train)):
    #     for j in range(len(X_train[0])):
    #         if isinstance(X_train[i][j],str):
    #             cnt+=1
    # print(cnt)
            

    X_train=np.asarray(X_train).astype(np.float32)
    y_train=np.asarray(y_train).astype(np.float32)
    X_test=np.asarray(X_test).astype(np.float32)
    y_test=np.asarray(y_test).astype(np.float32)

    acc = []
    f1 = []
    precision = []
    recall = []
    # seeds = random.sample(range(1, 200), 1)
    # for seed in seeds:
    # reset_random_seeds(seed)
    model = Sequential()
    # model.add(Dense(128, input_shape = (185,), activation = "relu"))
    model.add(Dense(128, input_shape = (len(X_test[0]),), activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(50, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(3, activation = "softmax"))
    
    model.compile(Adam(learning_rate = 0.001), "sparse_categorical_crossentropy", metrics = ["sparse_categorical_accuracy"])
    
    model.summary()
    
    
    model.fit(X_train, y_train,  epochs=1000, validation_split=0.1, batch_size=32,verbose=1) 
    model.save("train_clinical_model1.hdf5")

        # score = model.evaluate(X_test, y_test, verbose=0)
        # print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
        # acc.append(score[1])
        
        # test_predictions = model.predict(X_test)
        # test_label = to_categorical(y_test,3)

        # true_label= np.argmax(test_label, axis =1)

        # predicted_label= np.argmax(test_predictions, axis =1)
        
        # cr = classification_report(true_label, predicted_label, output_dict=True)
        # precision.append(cr["macro avg"]["precision"])
        # recall.append(cr["macro avg"]["recall"])
        # f1.append(cr["macro avg"]["f1-score"])
    
    # print("Avg accuracy: " + str(np.array(acc).mean()))
    # print("Avg precision: " + str(np.array(precision).mean()))
    # print("Avg recall: " + str(np.array(recall).mean()))
    # print("Avg f1: " + str(np.array(f1).mean()))
    # print("Std accuracy: " + str(np.array(acc).std()))
    # print("Std precision: " + str(np.array(precision).std()))
    # print("Std recall: " + str(np.array(recall).std()))
    # print("Std f1: " + str(np.array(f1).std()))
    # print(acc)
    # print(precision)
    # print(recall)
    # print(f1)
    
    
    """
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.savefig('snp_loss.png')
    plt.show()
    """


if __name__ == '__main__':
    main()



