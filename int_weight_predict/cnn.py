# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

__author__ = 'hqj'
import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import pickle
import os

nb_classes = 34

nb_epoch = 30

batch_size = 128


def load_valdata(dir_path):
    # 读取验证数据、测试数据
    with open(os.path.join(dir_path, 'valSet&TestSet.pickle'), 'rb') as train_file:
        val_X = pickle.load(train_file)
        val_y = pickle.load(train_file)
        test_X = pickle.load(train_file)
        test_y = pickle.load(train_file)

    # val_y = np_utils.to_categorical(val_y, nb_classes)
    # test_y = np_utils.to_categorical(test_y, nb_classes)

    with open(os.path.join(dir_path, 'Olddata_TestSet.pickle'), 'rb') as otherFile:
        other_X = pickle.load(otherFile)
        other_y = pickle.load(otherFile)
    # other_y = np_utils.to_categorical(other_y, nb_classes)

    return (val_X, val_y), (test_X, test_y), (other_X, other_y)


def Net_model(layer1, hidden1, region, rows, cols, nb_classes, lr=0.01, decay=1e-6, momentum=0.9):
    model = Sequential()

    model.add(Convolution2D(layer1, region, region,
                            border_mode='valid',
                            input_shape=(1, rows, cols)))

    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # 平铺

    model.add(Dense(hidden1))  # Full connection 1:  1000
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

    return model


def tramsform(num):
    if num < 10:
        return str(num)
    else:
        if num >= 24:
            num += 1
        if num >= 31:
            num += 1
        return chr(ord('A') + num - 10)


def test_model(model, X_test, Y_test):
    # 预测
    predicted = model.predict_classes(X_test, verbose=0)
    # 统计混淆集
    badcase = {}
    for i in range(0, len(Y_test)):
        if (tramsform(predicted[i]) in ['1', 'I'] and tramsform(Y_test[i]) in ['1', 'I']):
            # 1跟I区分，只要 是 测试 成1或I，而实际值是 1或I都算对
            predicted[i] = Y_test[i]
        if predicted[i] != Y_test[i]:
            ch1 = tramsform(Y_test[i])
            ch2 = tramsform(predicted[i])
            string = ','.join([ch1, ch2])

            if badcase.has_key(string):
                badcase[string] += 1
            else:
                badcase[string] = 1

    # 计算测试准确率
    test_accuracy = np.mean(np.equal(predicted, Y_test))
    graterThan5 = 0
    graterThan10 = 0
    for key, value in badcase.items():
        if value >= 5:
            graterThan5 += 1
        if value >= 10:
            graterThan10 += 1

    return (test_accuracy, graterThan5, graterThan10)



image_higth, image_width = 15, 15
lr = [0.05, 0.01, 0.005]
layer1 = 10
hidden1 = 40
region = 3
# 数据集根目录
data_root_path = '/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/modelAndData/Data/'
# 模型权重根目录
model_root_path = '/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/modelAndData/model'

# region 读取数据集：验证数据(64369个)、测试数据(64381个)、其他应用数据集(243391个)
(X_val, y_val), (X_test, y_test), (X_other, y_other) = load_valdata(data_root_path)
# endregion


model_file_list_path = os.listdir(model_root_path)
for i in model_file_list_path:
    print(i)
    # iteration += 1

    # 加载模型架构
    # 这里的 lr 设置什么不影响
    model = Net_model(layer1, hidden1, region, image_higth, image_width, nb_classes=34, lr=0)
    model.summary()
    quit()
    model.load_weights(os.path.join(model_root_path, i))
    model.get_weights()
    (val_accuracy, val5, val10) = test_model(model, X_val, y_val)
    print('验证集：%f,%d,%d' % (val_accuracy, val5, val10))
    (test_accuracy, test5, test10) = test_model(model, X_test, y_test)
    print('测试集：%f,%d,%d' % (test_accuracy, test5, test10))
    (other_accuracy, other5, other10) = test_model(model, X_other, y_other)
    print('other集：%f,%d,%d' % (other_accuracy, other5, other10))

