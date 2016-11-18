# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
__author__ = 'hqj'
import numpy as np;
np.random.seed(1337)  # for reproducibility
import theano;
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers import Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
import pickle
from sklearn import metrics;
from keras.utils import np_utils, generic_utils
import csv;
from sklearn.ensemble import RandomForestClassifier;
from PIL import Image;
import readTrainData as RD;



nb_classes = 34

nb_epoch =30

batch_size = 128









def load_valdata(path):
    with open(path, 'rb') as train_file:
        val_X=pickle.load(train_file);
        val_y=pickle.load(train_file);
        test_X = pickle.load(train_file);
        test_y = pickle.load(train_file)

    val_y = np_utils.to_categorical(val_y, nb_classes)
    test_y = np_utils.to_categorical(test_y, nb_classes)

    with open('E:\Image\Newdata\OldAndAlldata_TestSet.pickle','rb') as otherFile:
        other_X = pickle.load(otherFile);
        other_y = pickle.load(otherFile)
    other_y = np_utils.to_categorical(other_y, nb_classes)

    return (val_X,val_y),(test_X,test_y),(other_X,other_y);


def readTrainData(trainDataPath):
    with open(trainDataPath, 'rb') as train_file:
        X=pickle.load(train_file);
        y=pickle.load(train_file)
        y = np_utils.to_categorical(y, nb_classes)
        return(X,y)




def Net_model(layer1,hidden1,region,rows,cols,nb_classes,lr=0.01 ,decay=1e-6,momentum=0.9):


    model = Sequential()

    model.add(Convolution2D(layer1, region, region,
                                    border_mode='valid',
                                    input_shape=(1, rows, cols)))


    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # 平铺


    model.add(Dense(hidden1)) #Full connection 1:  1000
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))


    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])



    return model

def train_model(iteration,model,X_train,Y_train,layer1,hidden1,region,savePath,lr):

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,shuffle=True,verbose=1,validation_split=0)

    model.save_weights(savePath+'iteration'+str(iteration)+'_model_weights_'+str(layer1)+'-'+str(hidden1)+'_region'+str(region)+'_lr'+str(lr)+'_firstCNN_final.h5',overwrite=True)
    return model









def tramsform(num):
    if num <10:
        return str(num);
    else:
        if num>=24:
            num+=1;
        if num>=31:
            num+=1;
        return chr(ord('A')+num-10)




def test_model(model,X_test,Y_test,layer1,hidden1,region,datapath):
    predicted=model.predict_classes(X_test,verbose=0)
    y = np.zeros((len(predicted),),dtype="uint8")
    for i in range(0,len(Y_test)):
        maxIndex=0;
        for j in range(1,len(Y_test[i,:])):
            if Y_test[i,j]>Y_test[i,maxIndex]:
                maxIndex=j;
        y[i]=maxIndex;
    index=[];
    badcase={};
    for i in range(0, len(y)):
        if (predicted[i]==1 or predicted[i]==18) and (y[i]==1 or y[i]==18):
            predicted[i]=y[i]
        if predicted[i] != y[i]:
            ch1=tramsform(y[i]);
            ch2=tramsform(predicted[i]);
            string = ch1 + ',' + ch2;
            string2 = ch2 + ',' + ch1;
            if badcase.has_key(string):
                badcase[string] = badcase[string] + 1;
            elif badcase.has_key(string2):
                badcase[string2] = badcase[string2] + 1;
            else:
                badcase[string] = 1;
            index.append(i);
    test_accuracy = np.mean(np.equal(predicted, y))
    graterThan5=0;
    graterThan10=0
    for key,value in badcase.items():
        temp=key.split(',');
        original=temp[0];
        predict=temp[1];
        if value>=5:
            graterThan5+=1;
        if value>=10:
            graterThan10+=1;

    return (test_accuracy,graterThan5,graterThan10)


# layer1=[20,25,32]

lr=[0.05,0.01,0.005]
layer1=10;
hidden1=40;
region=3;
trainData='trainData.pickle'
dataPath='E:\Image\Newdata\\TrainSet_trainAndVal_testSet.pickle';
savePath='E:\Image\parameter\model\TrainAs_goodBadAndTrain\data3\\'
recodePath='E:\Image\parameter\model\TrainAs_goodBadAndTrain\data3\\recode.csv'

csvfile = file(recodePath, 'wb')
spamwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
spamwriter.writerow(['迭代','lr', '验证准确率', '验证混淆>5', '验证混淆>10','测试准确率', '测试混淆>5', '测试混淆>10','应用准确率', '应用混淆>5', '应用混淆>10']);
csvfile.close()


iteration=0;
(X_val,y_val),(X_test, y_test),(X_other,y_other) = load_valdata(dataPath)
while(True):
    (X,y)=readTrainData(trainData)
    for i in range(0,len(lr)):
        iteration+=1;
        model=Net_model(layer1,hidden1,region,len(X[0,0,:,0]),len(X[0,0,0,:]),nb_classes=34,lr=lr[i]);
        model = train_model(iteration,model, X, y,layer1,hidden1,region,savePath,lr[i])
        (val_accuracy, val5, val10) = test_model(model, X_val, y_val, layer1, hidden1, region, dataPath)
        (test_accuracy,test5,test10)=test_model(model,X_test,y_test,layer1,hidden1,region,dataPath)
        (other_accuracy, other5, other10) = test_model(model, X_other, y_other, layer1, hidden1, region, dataPath)
        csvfile = file(recodePath, 'a')
        spamwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([iteration,lr[i],val_accuracy,val5,val10,test_accuracy,test5,test10,other_accuracy,other5,other10]);
        csvfile.close()


