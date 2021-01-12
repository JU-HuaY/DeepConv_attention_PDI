
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:45:12 2020

@author: 华阳
"""
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Embedding, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D, SpatialDropout1D,Flatten,Multiply
from keras.layers import Concatenate,Bidirectional,LSTM,GRU,SimpleRNN
from keras.optimizers import Adam
from keras.regularizers import l2,l1
from keras.preprocessing import sequence
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve, auc, roc_curve, f1_score,cohen_kappa_score


class Protein_Drug_Prediction(object):
    def Conv(self, size, filters, activation, initializer, regularizer_param):
        def f(inputs):
            model_yuan = inputs
            model_p = Convolution1D(filters=filters, kernel_size=size, padding='same', kernel_initializer=initializer, kernel_regularizer=l2(regularizer_param))(inputs)
            model_p = BatchNormalization()(model_p)
            model_p = Activation(activation)(model_p)
            model_p = Concatenate(axis=2)([model_yuan,model_p])
            return GlobalMaxPooling1D()(model_p)
        return f
    
    def attention(self,input1,input2):
        # 如果上一层是LSTM，需要return_sequences=True
        V = input1
        QK = Dense(128,bias=True)(input2)
        QK = Activation("softmax")(QK)
        MV = Multiply()([V, QK])
        
        return MV

     
    def Att(self,att_dim,inputs):
        V = inputs
        QK = Dense(att_dim,bias=None)(inputs)
        QK = Activation("softmax")(QK)
        MV = Multiply()([V, QK])
        return MV
    
    def model(self, dropout, filters, prot_len=2500, activation='relu', initializer="glorot_normal", drug_len1=2048,drug_len2=300):
                
        input_d1 = Input(shape=(drug_len1,))
        input_d2 = Input(shape=(drug_len2,))
        input_p = Input(shape=(prot_len,))
        params_dic = {"kernel_initializer": initializer,
                      "kernel_regularizer": l2(0.001),
        }
        
        model_p = Embedding(7,16, embeddings_initializer=initializer,embeddings_regularizer=l2(0.001))(input_p)
        #model_p = Lambda(lambda x:K.expand_dims(x,axis=1))(input_p)
        model_p = SpatialDropout1D(0.2)(model_p)
        model_ps = [self.Conv(stride_size, filters, activation, initializer, 0.0001)(model_p) for stride_size in (5,10,15)]
        if len(model_ps)!=1:
            model_p = Concatenate(axis=1)(model_ps)
        else:
            model_p = model_ps[0]
      
        model_p = Dense(128, **params_dic)(model_p)
        model_p = BatchNormalization()(model_p)
        model_p = Activation(activation)(model_p)
        model_p = Dropout(dropout)(model_p)

        model_d1 = Lambda(lambda x:K.expand_dims(x,axis=1))(input_d1)
        input_layer_d1 = Bidirectional(GRU(512))(model_d1)
        model_d2 = Lambda(lambda x:K.expand_dims(x,axis=1))(input_d2)
        input_layer_d2 = Bidirectional(GRU(64))(model_d2)
        
        model_d = Concatenate(axis=1)([input_layer_d1,input_layer_d2])
        
        model_d = Lambda(lambda x:K.expand_dims(x,axis=1))(model_d)
        model_d = SpatialDropout1D(0.2)(model_d)
        model_ds = [self.Conv(stride_size, filters, activation, initializer, 0.0001)(model_d) for stride_size in (3,3)]
        if len(model_ds)!=1:
            model_d = Concatenate(axis=1)(model_ds)
        else:
            model_d = model_ds[0]
        
        model_d = Dense(128, **params_dic)(model_d)
        model_d = BatchNormalization()(model_d)
        model_d = Activation(activation)(model_d)
        
        model_p = self.attention(model_p,model_d)
        
        model_t = Concatenate(axis=1)([model_d,model_p])
        model_t = self.Att(256,model_t)
        model_t = Lambda(lambda x:K.expand_dims(x,axis=1))(model_t)
        model_t = Bidirectional(GRU(128))(model_t)
        model_t = BatchNormalization()(model_t)
        model_t = Activation(activation)(model_t)
        #model_t = LSTM(256)(model_t)
        
        model_t = Dense(1, activation='tanh', activity_regularizer=l2(0.0001),**params_dic)(model_t)
        model_t = Lambda(lambda x: (x+1.)/2.)(model_t)
        
        model_f = Model(inputs=[input_d1,input_d2, input_p], outputs = model_t)

        return model_f
    
    def __init__(self, dropout=0.2,filters=128,decay=0.0, prot_len=2500, activation="relu",drug_len1=2048,drug_len2=300):
        self.__dropout = dropout
        self.__filters = filters
        self.__prot_len = prot_len
        self.__drug_len1 = drug_len1
        self.__drug_len2 = drug_len2
        self.__activation = activation
        self.__decay = decay
        self.__model_t = self.model(self.__dropout, self.__filters, prot_len=self.__prot_len, activation=self.__activation,drug_len1=self.__drug_len1,drug_len2=self.__drug_len2)
        opt = Adam(lr=0.0001, decay=self.__decay)
        self.__model_t.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        K.get_session().run(tf.global_variables_initializer())
        
    def fit(self, drug_feature1,drug_feature2,protein_feature, label,test_drug1,test_drug2,test_pro,test_label,n_epoch=10, batch_size=32):
        for _ in range(n_epoch):
            history = self.__model_t.fit([drug_feature1,drug_feature2,protein_feature],label, epochs=_+1, batch_size=batch_size, validation_data=([test_drug1,test_drug2,test_pro],test_label),shuffle=True, verbose=1,initial_epoch=_)
        return self.__model_t
    
    def summary(self):
        self.__model_t.summary()
        
    def validation(self, drug_feature1,drug_feature2, protein_feature, label, test_drug1,test_drug2,test_pro,test_label,output_file=False, n_epoch=50, batch_size=32, **kwargs):
        AUC_best = 0
        for _ in range(n_epoch):
            history = self.__model_t.fit([drug_feature1,drug_feature2,protein_feature],label, epochs=_+1, batch_size=batch_size, validation_data=([test_drug1,test_drug2,test_pro],test_label),shuffle=True, verbose=1,initial_epoch=_)
            for dataset in kwargs:
                print("\tPredction of " + dataset)
                test_p = kwargs[dataset]["protein_feature"]
                test_d1 = kwargs[dataset]["drug_feature1"]
                test_d2 = kwargs[dataset]["drug_feature2"]
                #test_d3 = kwargs[dataset]["drug_feature3"]
                test_label = kwargs[dataset]["label"]
                prediction = self.__model_t.predict([test_d1,test_d2,test_p])
                fpr, tpr, thresholds_AUC = roc_curve(test_label, prediction)
                AUC = auc(fpr, tpr)
                precision, recall, thresholds = precision_recall_curve(test_label,prediction)
                AUPR = auc(recall,precision)
                F1_score = f1_score(test_label,(prediction+0.5).astype(int))
                kappa = cohen_kappa_score(test_label.astype(int),(prediction+0.5).astype(int))
                #print(prediction)
                print("\tKappa: %0.3f" % kappa)
                print("\tArea Under ROC Curve(AUC): %0.3f" % AUC)
                print("\tArea Under PR Curve(AUPR): %0.3f" % AUPR)
                print("\tF1_score: %0.3f" % F1_score)
                if AUC > AUC_best:
                    AUC_best = AUC
                    self.__model_t.save('model/my_model_weights.h5')
                    print("保存该epoch模型")
                #if AUC > 0.951:
                    #np.save("predic.npy",prediction)
                    #np.save("label.npy",test_label)
                    #print("保存成功")
                
    def predict(self, **kwargs):
        results_dic = {}
        
        for dataset in kwargs:
            result_dic = {}
            test_p = kwargs[dataset]["protein_feature"]
            test_d = kwargs[dataset]["drug_feature"]
            result_dic["label"] = kwargs[dataset]["label"]
            result_dic["predicted"] = self.__model_t.predict([test_d, test_p])
            results_dic[dataset] = result_dic
        return results_dic

   
    def save(self, output_file):
        self.__model_t.save(output_file)
        
        

train_protein = np.load("kinase/train/proteins.npy")
train_drug1 = np.load("kinase/train/morgan.npy")
train_drug2 = np.load("kinase/rain/mol2vec.npy")
train_label = np.load("kinase/train/interactions.npy")
test_protein = np.load("kinase/test/proteins.npy")
test_drug1 = np.load("kinase/test/morgan.npy")
test_drug2 = np.load("kinase/test/mol2vec.npy")
test_label = np.load("kinase/test/interactions.npy")

test_dic1 = {"protein_feature": test_protein, "drug_feature1": test_drug1,"drug_feature2": test_drug2, "label": test_label}
test_dic = { "predict" : test_dic1}

model_params = {
        "decay": 0.0001,
        "activation": "elu" ,
        "filters": 128,
        "dropout": 0.05
    }



dti_prediction_model = Protein_Drug_Prediction(**model_params)
dti_prediction_model.summary()
#dti_prediction_model.fit(**train_dic)

dti_prediction_model.validation(train_drug1,train_drug2,train_protein,train_label,test_drug1,test_drug2,test_protein,test_label,**test_dic)
