#encoding=utf8
from dataProcessing.read_data import load_pix
import logging
import timeit
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Convolution2D,Activation,MaxPooling2D,Flatten,Dropout,ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )
NB_EPOCH = 32
num_train = 1000
num_test = 100
char_set = 3
num_class = 34

train_file_path = '/home/jdwang/PycharmProjects/digitRecognition/train_test_data/20160426/' \
                  'train_%dcharset_%d.csv' % (char_set, num_train)
test_file_path = '/home/jdwang/PycharmProjects/digitRecognition/train_test_data/20160426/' \
                 'test_%dcharset_%d.csv' % (char_set, num_test)

image_shape = (15,15)
# train_pix,train_y,train_label,train_im_name = load_pix(train_file_path,
#                                            shape=image_shape,
#                                            char_set = char_set
#                                            )
# # print train_y
# # quit()
# test_pix,test_y,test_label,test_im_name = load_pix(test_file_path,
#                                                    shape=image_shape,
#                                                    char_set=char_set
#                                                    )
# label = test_y
# train_pix = train_pix.reshape(train_pix.shape[0],
#                              1,
#                              train_pix.shape[1],
#                              train_pix.shape[2])
# test_pix = test_pix.reshape(test_pix.shape[0],
#                              1,
#                              test_pix.shape[1],
#                              test_pix.shape[2])

# character_name = sorted(list(set('0123456789ABCDEFGHIJKLMNPQRSTUWXYZ')))
# print character_name
# 将y转换成one-hot编码
# train_y = [character_name.index(item) for item in train_y]

from train_test_data.dataset_20160801.data_util import DataUtil

dutil = DataUtil()


charset='8B'

(train_pix,train_y),(test_pix,test_y) = dutil.load_train_test_data(
    version=1,
    charset=charset
)

train_y = np_utils.to_categorical(train_y,num_class)
# test_y = [character_name.index(item) for item in test_y]
test_y = np_utils.to_categorical(test_y,num_class)
# print train_y.shape
# print test_y.shape
# quit()

logging.debug( 'the shape of train sample:%d,%d,%d,%d'%(train_pix.shape))
logging.debug( 'the shape of test sample:%d,%d,%d,%d'%(test_pix.shape))
model = Sequential()

# Layer 1
model.add(Convolution2D(96, 11, 11, input_shape = (1,15,15), border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Convolution2D(256, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, border_mode='same'))
model.add(Activation('relu'))

# Layer 4
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(1024, 3, 3, border_mode='same'))
model.add(Activation('relu'))

# Layer 5
model.add(ZeroPadding2D((1,1)))
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


print model.summary()
# quit()
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
nb_epoch = NB_EPOCH
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
start_time = timeit.default_timer()
model.fit(train_pix,
          train_y,
          nb_epoch=nb_epoch,
          verbose=1,
          validation_data=(test_pix,test_y),
          validation_split=0.1,
          shuffle=True,
          batch_size=100)
# print model.get_weights()
end_time = timeit.default_timer()
print 'train time : %f'%(end_time-start_time)

# 保存模型
json_string = model.to_json()
# print json_string
cnn_model_architecture = '/home/jdwang/PycharmProjects/digitRecognition/cnn_model/model/' \
                         'cnn_model_architecture_%dtrain_%dwin_%depoch.json' \
                         % (num_train,win_shape, nb_epoch)
open(cnn_model_architecture, 'w').write(json_string)
logging.info('模型架构保存到：%s'%cnn_model_architecture)
cnn_model_weights = '/home/jdwang/PycharmProjects/digitRecognition/cnn_model/model/' \
                    'cnn_model_weights_%dtrain_%dwin_%depoch.h5' \
                    % (num_train,win_shape, nb_epoch)
model.save_weights(cnn_model_weights,overwrite=True)
logging.info('模型权重保存到：%s'%cnn_model_weights)


