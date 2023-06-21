import os
import random
import gc, numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout,Flatten, BatchNormalization, Conv2D, concatenate
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from tensorflow.python.keras.layers.multi_head_attention import MultiHeadAttention

# tf.get_logger().setLevel('ERROR')

import sys

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

def make_img(t_img):
    img = pd.read_pickle(t_img)
    img_l = []
    for i in range(len(img)):
        img_l.append(img.values[i][0])
    return np.array(img_l)

               
def create_model_snp():
    
    model = Sequential()
    model.add(Dense(200,  activation = "relu")) 
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(50, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    return model

def create_model_clinical():
    
    model = Sequential()
    model.add(Dense(200,  activation = "relu")) 
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(50, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))    
    return model

def create_model_img():

    model = Sequential()
    model.add(Conv2D(72, (3, 3), activation='relu')) 
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))   
    return model


def create_model_pet():
    model = Sequential()
    model.add(Conv2D(72, (3, 3), activation='relu')) 
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))   
    return model


def plot_classification_report(y_tru, y_prd, mode, learning_rate, batch_size,epochs, figsize=(7, 7), ax=None):

    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = ["Control", "Moderate", "Alzheimer's" ] 
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep,
                annot=True, 
                cbar=False, 
                xticklabels=xticks, 
                yticklabels=yticks,
                ax=ax, cmap = "Blues")
    
    plt.savefig('report_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'_' + str(epochs)+'.png')
    


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



def cross_modal_attention(x, y):
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=1)
    a1 = MultiHeadAttention(num_heads = 4,key_dim=50)(x, y)
    a2 = MultiHeadAttention(num_heads = 4,key_dim=50)(y, x)
    a1 = a1[:,0,:]
    a2 = a2[:,0,:]
    return concatenate([a1, a2])


def self_attention(x):
    x = tf.expand_dims(x, axis=1)
    attention = MultiHeadAttention(num_heads = 4, key_dim=50)(x, x)
    attention = attention[:,0,:]
    return attention
    

def multi_modal_model(mode, train_clinical, train_snp, train_img,train_pet):

    in_clinical = Input(shape=(train_clinical.shape[1]))
    
    in_snp = Input(shape=(train_snp.shape[1]))
    
    # (3119, 1, 72, 72, 3) numpy array shape
    in_img = Input(shape=(train_img.shape[1], train_img.shape[2], train_img.shape[3]))
    # in_img = Input(shape=(train_img.shape[2], train_img.shape[3], train_img.shape[4]))
    
    in_pet = Input(shape=(train_pet.shape[1], train_pet.shape[2], train_pet.shape[3]))

    dense_clinical = create_model_clinical()(in_clinical)
    dense_snp = create_model_snp()(in_snp) 
    dense_img = create_model_img()(in_img) 
    dense_pet = create_model_img()(in_pet) 
    
 
        
    ########### Attention Layer ############
        
    ## Cross Modal Bi-directional Attention ##

    if mode == 'MM_BA':
            
        ci_att = cross_modal_attention(dense_clinical,dense_img,)
        gi_att = cross_modal_attention(dense_snp, dense_img)
        cg_att = cross_modal_attention(dense_clinical, dense_snp)
        cp_att = cross_modal_attention(dense_clinical, dense_pet)
        gp_att = cross_modal_attention(dense_snp, dense_pet)
        ip_att = cross_modal_attention(dense_img, dense_pet)
                
        merged = concatenate([ci_att, gi_att, cg_att, cp_att, gp_att, ip_att, dense_img, dense_snp, dense_clinical,dense_pet])
                 
   
        
        
    ## Self Attention ##
    elif mode == 'MM_SA':
            
        ii_att = self_attention(dense_img)
        cc_att = self_attention(dense_clinical)
        gg_att = self_attention(dense_snp)
        pp_att = self_attention(dense_pet)
            
        merged = concatenate([ii_att, cc_att, gg_att, pp_att, dense_img, dense_snp, dense_clinical, dense_pet])
        
    ## Self Attention and Cross Modal Bi-directional Attention##
    elif mode == 'MM_SA_BA':
            
        ii_att = self_attention(dense_img)
        cc_att = self_attention(dense_clinical)
        gg_att = self_attention(dense_snp)
        pp_att = self_attention(dense_pet)
        
        ci_att = cross_modal_attention(cc_att,ii_att)
        gi_att = cross_modal_attention(gg_att,ii_att)
        cg_att = cross_modal_attention(cc_att,gg_att)
        cp_att = cross_modal_attention(cc_att,pp_att)
        gp_att = cross_modal_attention(gg_att,pp_att)
        ip_att = cross_modal_attention(ii_att,pp_att)
            
        merged = concatenate([ci_att,gi_att,cg_att,cp_att,gp_att,ip_att, dense_img, dense_snp, dense_clinical,dense_pet])
            
        
    ## No Attention ##    
    elif mode == 'None':
            
        merged = concatenate([dense_img, dense_snp, dense_clinical,dense_pet])
                
    else:
        print ("Mode must be one of 'MM_SA', 'MM_BA', 'MU_SA_BA' or 'None'.")
        return
                
        
    ########### Output Layer ############
        
    output = Dense(3, activation='softmax')(merged)
    model = Model([in_clinical, in_snp, in_img,in_pet], output)        
        
    return model



def train(mode, batch_size, epochs, learning_rate):
    
    #?clinical
    train_clinical = pd.read_pickle("overlap_train/X_train_clinical.pkl").fillna(0).replace(r'[^0-9]',0,regex=True).astype("float32").values
    # test_clinical= pd.read_pickle("overlap_train/X_test_clinical.pkl").fillna(0).replace(r'[^0-9]',0,regex=True).astype("float32").values

    #?Genetic
    train_snp = pd.read_pickle("overlap_train/X_train_snp_M.pkl").fillna(0).replace(r'[^0-9]',0,regex=True).astype("float32").values
    # test_snp = pd.read_pickle("overlap_train/X_test_snp.pkl").fillna(0).replace(r'[^0-9]',0,regex=True).astype("float32").values

    #?MRI image
    train_img= make_img("overlap_train/X_train_img.pkl")
    # test_img= make_img("overlap_train/X_test_img.pkl")

    train_img= train_img.reshape(-1,72,72,3)
    # test_img= test_img.reshape(-1,72,72,3)
    train_img= np.asarray(train_img).astype('float32')
    # test_img= np.asarray(test_img).astype('float32')

    #?PET image
    train_pet= make_img("overlap_train/X_train_pet.pkl")
    # test_pet= make_img("overlap_train/X_test_pet.pkl")

    train_pet= train_pet.reshape(-1,72,72,3)
    # test_pet= test_pet.reshape(-1,72,72,3)
    train_pet= np.asarray(train_pet).astype('float32')
    # test_pet= np.asarray(test_pet).astype('float32')


    #?overlap
    train_label= pd.read_pickle("overlap_train/y_train.pkl").values.astype("float32").flatten()
    # test_label= pd.read_pickle("overlap_train/y_test.pkl").values.astype("float32").flatten()


    # train_clinical= np.asarray(train_clinical).astype('float32')
    # test_clinical= np.asarray(test_clinical).astype('float32')
    
    # train_snp= np.asarray(train_snp).astype('float32')
    # test_snp= np.asarray(test_snp).astype('float32')


    # train_label= np.asarray(train_label).astype('float32')
    # test_label= np.asarray(test_label).astype('float32')

    class_weights = compute_class_weight(class_weight = 'balanced',classes = np.unique(train_label),y = train_label)
    d_class_weights = dict(enumerate(class_weights))
    
    # compile model #
    model = multi_modal_model(mode, train_clinical, train_snp, train_img,train_pet)
    model.compile(optimizer=Adam(learning_rate = learning_rate), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    

    # summarize results
    model.fit([train_clinical,
                    train_snp,
                    train_img,train_pet],
                    train_label,
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=d_class_weights,
                    validation_split=0.1,
                    verbose=1)
    
    model.save("train_all.hdf5")

    # score = model.evaluate([test_clinical, test_snp, test_img,test_pet], test_label)
    
    # acc = score[1] 
    # test_predictions = model.predict([test_clinical, test_snp, test_img,test_pet])
    # cr, precision_d, recall_d, thres = calc_confusion_matrix(test_predictions, test_label, mode, learning_rate, batch_size, epochs)
    
    
    """
    plt.clf()
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('accuracy_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('loss_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'.png')
    plt.clf()
    """
    
 
    
    # release gpu memory #
    # K.clear_session()
    # del model, history
    # gc.collect()
        
        
    # print ('Mode: ', mode)
    # print ('Batch size:  ', batch_size)
    # print ('Learning rate: ', learning_rate)
    # print ('Epochs:  ', epochs)
    # print ('Test Accuracy: {} '.format(acc))
    # print ('-'*55)
    
    # return acc, batch_size, learning_rate, epochs
    
    
if __name__=="__main__":
    tf.keras.backend.clear_session()
    m_a = {}
    #epoch is 3rd argument here
    train('None',32, 3, 0.00001)