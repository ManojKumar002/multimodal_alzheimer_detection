import os
import random
import pickle
import gc, numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import compute_class_weight
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Dense, Dropout,Flatten, BatchNormalization, Conv2D, MultiHeadAttention, concatenate
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential,load_model
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)


def make_img(t_img):
    img = pd.read_pickle(t_img)
    img_l = []
    for i in range(len(img)):
        img_l.append(img.values[i][0])
    return np.array(img_l)


def calc_confusion_matrix(result, test_label,mode, learning_rate, batch_size, epochs):
    test_label = to_categorical(test_label,3)
    true_label= np.argmax(test_label, axis =1)
    predicted_label= np.argmax(result, axis =1)
    
    n_classes = 3
    precision = dict()
    recall = dict()
    thres = dict()
    for i in range(n_classes):
        precision[i], recall[i], thres[i] = precision_recall_curve(test_label[:, i],
                                                            result[:, i])


    print ("Classification Report :") 
    print (classification_report(true_label, predicted_label))
    cr = classification_report(true_label, predicted_label, output_dict=True)
    return cr, precision, recall, thres


def test_model(mode, batch_size, epochs, learning_rate,):
    #?clinical
    test_clinical= pd.read_pickle("X_test_clinical.pkl").fillna(0).replace(r'[^0-9]',0,regex=True).astype("float32").values

    #?Genetic
    test_snp = pd.read_pickle("X_test_snp_M.pkl").fillna(0).replace(r'[^0-9]',0,regex=True).astype("float32").values

    #?MRI image
    test_img= make_img("X_test_img.pkl")

    test_img= test_img.reshape(-1,72,72,3)
    test_img= np.asarray(test_img).astype('float32')

    #?PET image
    test_pet= make_img("X_test_pet.pkl")

    test_pet= test_pet.reshape(-1,72,72,3)
    test_pet= np.asarray(test_pet).astype('float32')


    #?overlap
    test_label= pd.read_pickle("y_test.pkl").values.astype("float32").flatten()


    model =load_model("./train_all.hdf5")
    score = model.evaluate([test_clinical, test_snp, test_img,test_pet], test_label)
        
    acc = score[1] 
    test_predictions = model.predict([test_clinical, test_snp, test_img,test_pet])
    cr, precision_d, recall_d, thres = calc_confusion_matrix(test_predictions, test_label, mode, learning_rate, batch_size, epochs)

    # release gpu memory #
    K.clear_session()
    del model
    gc.collect()
        
        
    print ('Mode: ', mode)
    print ('Batch size:  ', batch_size)
    print ('Learning rate: ', learning_rate)
    print ('Epochs:  ', epochs)
    print ('Test Accuracy: {} '.format(acc))
    print ('-'*55)
    return acc, batch_size, learning_rate, epochs



if __name__=="__main__":
    m_a={}
    acc, bs_, lr_, e_ = test_model('MM_SA_BA', 32, 10, 0.001)
    m_a[acc] = ('MM_SA_BA', acc, bs_, lr_, e_)
    print(m_a)
    print ('-'*55)
    max_acc = max(m_a, key=float)
    print("Highest accuracy of: " + str(max_acc) + " with parameters: " + str(m_a[max_acc]))
