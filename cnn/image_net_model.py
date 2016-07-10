#encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-07-11'
    Email:   '383287471@qq.com'
    Describe: 
"""
from __future__ import print_function

import numpy as np
import pandas as pd
import logging
import timeit
from keras.layers import Dense,Convolution2D,Activation,MaxPooling2D,Flatten,Dropout,ZeroPadding2D
from keras.optimizers import SGD
from keras.models import Sequential
from sklearn.metrics import f1_score


class AlexNet(object):
    def __init__(self,
                 num_labels = None,
                 nb_epoch = 50,
                 ):


        super(AlexNet, self).__init__()

        self.num_labels = num_labels
        self.nb_epoch = nb_epoch


    def build_alexnet(self):
        '''
            build AlexNet 架构

        :return:
        '''
        model = Sequential()

        # Layer 1
        model.add(Convolution2D(96, 11, 11, input_shape=(1, 15, 15), border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 2
        model.add(Convolution2D(256, 5, 5, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 3
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, border_mode='same'))
        model.add(Activation('relu'))

        # Layer 4
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(1024, 3, 3, border_mode='same'))
        model.add(Activation('relu'))

        # Layer 5
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(1024, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 6
        model.add(Flatten())
        model.add(Dense(3072, init='glorot_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Layer 7
        model.add(Dense(4096, init='glorot_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Layer 8
        model.add(Dense(34, init='glorot_normal'))
        model.add(Activation('softmax'))
        return model


    def build_simple_cnn(self):
        model = Sequential()
        win_shape = 3
        model.add(ZeroPadding2D((1, 1), input_shape=(1, 15, 15)))
        model.add(Convolution2D(32, win_shape, win_shape,
                                border_mode='valid',
                                ))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, win_shape, win_shape,
                                border_mode='valid'
                                ))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, win_shape, win_shape,
                                border_mode='valid'
                                ))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(output_dim=100, init="glorot_uniform"))
        model.add(Activation("relu"))
        model.add(Dropout(p=0.5))
        model.add(Dense(output_dim=50, init="glorot_uniform"))
        model.add(Activation("relu"))
        model.add(Dropout(p=0.5))
        model.add(Dense(output_dim=self.num_labels, init="glorot_uniform"))
        model.add(Activation("softmax"))
        return model

    def build_model(self):

        model = self.build_alexnet()
        print(model.summary())
        # quit()
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
        self.model = model

    def fit(self,train_data,validate_data):
        train_X, train_y = train_data
        validate_X, validate_y = validate_data
        start_time = timeit.default_timer()
        self.model.fit(train_X,
                       train_y,
                       nb_epoch=self.nb_epoch,
                       verbose=1,
                       validation_data=(validate_X, validate_y),
                       # validation_split=0.1,
                       shuffle=True,
                       batch_size=100)
        # print model.get_weights()
        end_time = timeit.default_timer()
        print('train time : %f' % (end_time - start_time))

    def accuracy(self,test_data):
        test_X, test_y = test_data
        test_X = np.asarray(test_X)
        y_pred = self.model.predict(test_X)
        is_correct = y_pred == test_y
        logging.debug('正确的个数:%d' % (sum(is_correct)))
        print('正确的个数:%d' % (sum(is_correct)))
        accu = sum(is_correct) / (1.0 * len(test_y))
        logging.debug('准确率为:%f' % (accu))
        print('准确率为:%f' % (accu))

        f1 = f1_score(test_y, y_pred.tolist(), average=None)
        logging.debug('F1为：%s' % (str(f1)))
        print('F1为：%s' % (str(f1)))


if __name__ == '__main__':
    pass