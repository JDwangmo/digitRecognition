# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-11-18'; 'last updated date: 2016-11-18'
    Email:   '383287471@qq.com'
    Describe: 整形权重的预测, 5-6 二分类器 ，包括 56二分类器的训练，测试，最后 用 56二分类器修正 34分类结果
"""
from __future__ import print_function
import numpy as np
import struct

import time
import pickle
import os

np.random.seed(1337)  # for reproducibility
nb_classes = 2
nb_epoch = 10
batch_size = 128

character_name = list('ABCDEFGHIJKLMN0PQRSTUWXYZ')

image_higth, image_width = 15, 15
# lr = [0.05, 0.01, 0.005]
layer1 = 10
hidden1 = 20
region = 3

# 模型权重根目录
model_root_path = '/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/modelAndData1214/'


# 过滤数字，得到字母，并按0-24编码A-Z,注意没有V
def filterData(X, y):
    y_letter = [];
    index = []
    for i in range(0, len(y)):
        if y[i] >= 10:
            index.append(i)
            label = y[i]
            if label >= 24:
                label += 1
            y_letter.append(label - 10)
        elif y[i] == 0:
            index.append(i)
            y_letter.append(14)
    X_letter = X[index]
    return (X_letter, y_letter)


def load_valdata(version='1214'):
    """读取不同版本的测试集
    1122 - 对应 /home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/modelAndData1122/Data/

    :param dir_path: str
    :param version: str
    :return:
    """

    # 数据集根目录
    data_root_path = '/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/'
    other_X_new, other_y_new = None, None

    # 读取验证数据、测试数据
    if version == '1122':
        val_file_path = os.path.join(data_root_path, 'modelAndData1122/Data/', 'TrainSet_trainAndVal_testSet.pickle')
        other_file_path = os.path.join(data_root_path, 'modelAndData1122/Data/', 'Olddata_TestSet.pickle')
        with open(val_file_path, 'rb') as train_file:
            train_X = pickle.load(train_file)
            train_y = pickle.load(train_file)
            val_X, val_y = None, None
            test_X = np.asarray(pickle.load(train_file))
            test_y = np.asarray(pickle.load(train_file))
            with open(other_file_path, 'rb') as otherFile:
                other_X = pickle.load(otherFile)
                other_y = pickle.load(otherFile)

    elif version == '1214':
        val_file_path = os.path.join(data_root_path, 'modelAndData1214/Data/', 'TrainSet_trainAndVal_testSet.pickle')
        other_file_path = os.path.join(data_root_path, 'modelAndData1214/Data/', 'OldAndAlldata_TestSet.pickle')
        new_other_file_path = os.path.join(data_root_path, 'modelAndData1214/Data/', '20161202am.pickle')
        with open(val_file_path, 'rb') as train_file:
            val_X = pickle.load(train_file)
            val_y = pickle.load(train_file)
            test_X = pickle.load(train_file)
            test_y = pickle.load(train_file)
        (test_X, test_y) = filterData(test_X, test_y)
        test_y = np.asarray(test_y)
        with open(other_file_path, 'rb') as otherFile:
            other_X = pickle.load(otherFile)
            other_y = pickle.load(otherFile)
        (other_X, other_y) = filterData(other_X, other_y)
        other_y = np.asarray(other_y)

        with open(new_other_file_path, 'rb') as otherFile:
            other_X_new = pickle.load(otherFile)
            other_y_new = pickle.load(otherFile)
        (other_X_new, other_y_new) = filterData(other_X_new, other_y_new)
        other_y_new = np.asarray(other_y_new)

    else:
        raise NotImplementedError

    return (val_X, val_y), (test_X, test_y), (other_X, other_y), (other_X_new, other_y_new)


def Net_model(layer1, hidden1, region, rows, cols, nb_classes, lr=0.01, decay=1e-6, momentum=0.9):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.optimizers import SGD
    from keras import backend as K

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
    # 获取CNN的中间结果
    mid_output = K.function(inputs=[
        model.layers[0].input,
        K.learning_phase(),
    ],
        outputs=[
            model.layers[-9].output,
            model.layers[-8].output,
            model.layers[-7].output,
            model.layers[-6].output,
            model.layers[-5].output,
            model.layers[-4].output,
            model.layers[-3].output,
            model.layers[-2].output,
            model.layers[-1].output,
        ]
    )

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

    return model, mid_output


def test_model(model_file, X_test, y_test, other_X, other_y, type='0D'):
    if type == '0D':
        X_test = X_test[(y_test == char_to_index('0')) | (y_test == char_to_index('D'))][:, :, :, :8]
        y_test = y_test[(y_test == char_to_index('0')) | (y_test == char_to_index('D'))]

        other_X = other_X[(other_y == char_to_index('0')) | (other_y == char_to_index('D'))][:, :, :, :8]
        other_y = other_y[(other_y == char_to_index('0')) | (other_y == char_to_index('D'))]
        # 左半边
        image_higth = 15
        image_width = 8
    elif type == '0Q':
        X_test = X_test[(y_test == char_to_index('0')) | (y_test == char_to_index('Q'))][:, :, 8:, :]
        y_test = y_test[(y_test == char_to_index('0')) | (y_test == char_to_index('Q'))]

        other_X = other_X[(other_y == char_to_index('0')) | (other_y == char_to_index('Q'))][:, :, 8:, :]
        other_y = other_y[(other_y == char_to_index('0')) | (other_y == char_to_index('Q'))]
        # 下半边
        image_higth = 7
        image_width = 15
    elif type == '0G':
        X_test = X_test[(y_test == char_to_index('0')) | (y_test == char_to_index('G'))][:, :, :, 8:]
        y_test = y_test[(y_test == char_to_index('0')) | (y_test == char_to_index('G'))]

        other_X = other_X[(other_y == char_to_index('0')) | (other_y == char_to_index('G'))][:, :, :, 8:]
        other_y = other_y[(other_y == char_to_index('0')) | (other_y == char_to_index('G'))]
        # 右半边
        image_higth = 15
        image_width = 7
    else:
        raise NotImplementedError

    # 加载模型架构
    # 这里的 lr 设置什么不影响
    model, mid_output = Net_model(layer1, hidden1, region, image_higth, image_width, nb_classes=2, lr=0.01)
    model.load_weights(model_file)
    model.summary()
    # quit()


    # 预测
    predicted = model.predict_classes(X_test, verbose=0)

    if type == '0D':
        predicted[predicted == 0] = char_to_index('0')
        predicted[predicted == 1] = char_to_index('D')
    if type == '0Q':
        predicted[predicted == 0] = char_to_index('0')
        predicted[predicted == 1] = char_to_index('Q')
    if type == '0G':
        predicted[predicted == 0] = char_to_index('0')
        predicted[predicted == 1] = char_to_index('G')

    return count_result(predicted, y_test)


def count_result(predicted, Y_test):
    '''统计预测情况，包括混淆集个数等

    :return:
    '''
    # 统计混淆集
    badcase = {}
    for i in range(0, len(Y_test)):
        if predicted[i] != Y_test[i]:
            ch1 = tramsform(Y_test[i])
            ch2 = tramsform(predicted[i])
            # string = ','.join(sorted([ch1, ch2]))
            string = ','.join([ch1, ch2])
            if badcase.has_key(string):
                badcase[string] += 1
            else:
                badcase[string] = 1

    # 计算测试准确率
    test_accuracy = np.mean(np.equal(predicted, Y_test))
    graterThan2 = 0
    graterThan5 = 0
    graterThan10 = 0
    for key, value in badcase.items():
        if value >= 2:
            graterThan2 += 1
        if value >= 5:
            graterThan5 += 1
        if value >= 10:
            graterThan10 += 1

    return test_accuracy, graterThan2, graterThan5, graterThan10, badcase


def char_to_index(char,type=25):
    if character_name.__contains__(char):
        return character_name.index(char)
    else:
        raise NotImplementedError


def tramsform(num,type=25):
    """ 数字 转换成 字符

    :param num:
    :return:
    """
    return character_name[num]



def conv_pool_operation(img, conv_W, conv_b):
    """CNN 卷积 和 池化操作
        一张图片
    :param img: array-3D
        一张图片，3D，(num_of_channels,img_height,img_width)
    :param conv_W: array-4D
        10*3*3
    :param conv_b:
    :return: array-3D
    """
    filter_row, filter_col = conv_W.shape[2:]
    img_row, img_col = img.shape[1:]
    # convolution
    # 3D
    conv_result = np.zeros((conv_W.shape[0], img_row - filter_row + 1, img_col - filter_col + 1))
    # quit()

    for filter_index in range(conv_W.shape[0]):
        j_1 = conv_W[filter_index, 0].flatten()[-1::-1]
        # j= conv_W[filter_index, 0].flatten()
        for x in range(0, img_row - filter_row + 1):
            for y in range(0, img_col - filter_col + 1):
                conv_result[filter_index,
                            x,
                            y] = tanh_approximate_function(np.dot(img[0, x:x + filter_row, y:y + filter_col].flatten(),
                                                                  j_1
                                                                  # conv_W[filter_index, 0].flatten()[-1::-1]
                                                                  )
                                                           + conv_b[filter_index]
                                                           )
                # print(conv_result[filter_index,
                #             x,
                #             y])
                # print(np.dot(img[0, x:x + filter_row, y:y + filter_col].flatten(),j_1)+ conv_b[filter_index])
    # quit()
    # conv_result = tanh_approximate_function(conv_result)
    pool_row, pool_col = 2, 2
    pool_result = np.zeros(
        (conv_result.shape[0], conv_result.shape[1] / pool_row, conv_result.shape[2] / pool_col))
    # max-pooling
    for channel_index in range(pool_result.shape[0]):
        for x in range(pool_result.shape[1]):
            for y in range(pool_result.shape[2]):
                pool_result[
                    channel_index,
                    x,
                    y
                ] = np.max(
                    conv_result[
                    channel_index,
                    x * pool_row: (x + 1) * pool_row,
                    y * pool_col: (y + 1) * pool_col
                    ])
                # print('%d'%(pool_result[
                #     channel_index,
                #     x,
                #     y
                # ]))
    return pool_result


def hidden_operation(feature_vector, W, b, activion='tanh'):
    """ 隐含层操作

    :param activion: str
        激活函数
    :param feature_vector: array-like
        一张 图片的 feature vector，1D
    :param W: array-2D
        权重
    :param b: array-1D
        偏差
        0.001417s
        0.000125s
    :return:
    """
    # start = time.time()

    # temp =np.dot(feature_vector, W) + b
    # temp =map(lambda x:np.dot(feature_vector, x[0])+x[1],zip(W.transpose(),b))
    # temp =map(lambda x:np.dot(feature_vector, x[0])+x[1], zip(W.transpose(),b))
    #
    # print(feature_vector.shape,W.shape)
    # end = time.time()
    # print('hidden_operation time:%fs' % (end - start))
    # quit()
    if activion == 'tanh':
        # return np.asarray([tanh_approximate_function(item) for item in temp])
        return map(lambda x: tanh_approximate_function(np.dot(feature_vector, x[0]) + x[1]), zip(W.transpose(), b))
    elif activion == 'none':
        # return temp
        return map(lambda x: np.dot(feature_vector, x[0]) + x[1], zip(W.transpose(), b))
    else:
        raise NotImplementedError


def tanh_approximate_function(x):
    """
         * tanh x=sinh x / cosh x
         * 其中sinh x=(e^(x)-e^(-x))/2 ，cosh x=(e^x+e^(-x))/2
         * 所以tanhx = (e^(x)-e^(-x)) /(e^x+e^(-x))
    """
    #     return (int)(tanh(x));

    if x > 10:
        return 1
    elif x < -10:
        return -1
    else:
        return (int)((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))


def cnn_batch_predict(X_val, weights):
    '''CNN批量预测

    :param X_val:
    :param weights:
    :return:
    '''
    weights = [(item * 1e5).astype(dtype=int) for item in weights]
    # result = cnn_predict(X_val[0],weights_56)
    result = []
    for index, img in enumerate(X_val):
        if (index + 1) % 1000 == 0:
            print('%d' % (index + 1))
        result.append(cnn_predict(img, weights))
    return np.asarray(result)


def cnn_predict(img, weights):
    '''单张图片的预测

    :param img:
    :param weights:
    :return:
    '''
    # 3D
    # print(img)
    # quit()
    start = time.time()

    imgs_conv_result = conv_pool_operation(img, weights[0], weights[1])
    end = time.time()
    # print('conv time:%fs' % (end - start))

    start = time.time()

    # print('conv over..')
    flatten_result = imgs_conv_result.flatten()
    # print(flatten_result)
    end = time.time()
    # print('flatten time:%fs' % (end - start))

    start = time.time()
    # print('flatten over..')
    # print(flatten_result.shape)
    hidden1_output = hidden_operation(flatten_result, weights[2], weights[3], activion='tanh')
    hidden1_output = hidden_operation(flatten_result, weights[2], weights[3], activion='none')
    # print(hidden1_output)
    # quit()
    end = time.time()
    # print('hidden1 time:%fs' % (end - start))

    start = time.time()
    # print('hidden1 over..')
    hidden2_output = hidden_operation(hidden1_output, weights[4], weights[5], activion='none')
    end = time.time()
    # print('hidden2 time:%fs' % (end - start))
    # print('hidden2 over..')
    # print(hidden2_output.shape)
    start = time.time()
    result = np.argmax(hidden2_output)
    end = time.time()

    # print('max time:%fs' % (end - start))
    return result

def save_cnn_weight_to_bininary_file(model_weights, type='0D'):
    # print(model_weights)
    # print(len(model_weights))
    g_name = 'Letter_%s'%type
    names = [
        'C11_Map_Weight',
        'C11_B_Weight',
        'FC1_Map_Weight',
        'FC1_B_Weight',
        'FC2_Map_Weight',
        'FC2_B_Weight',
    ]
    count = 0
    for weight in model_weights:
        print(weight.shape)
        # print(weight[0][0])
        fout = open(os.path.join(model_root_path, 'int_weights', 'letter%s_int_weight%d.h' % (type,count)), 'w')
        fout.write('//%s\n' % str(weight.shape))
        fout.write('#ifndef DR_%s_%s_H\n'%(g_name,names[count]))
        fout.write('#define DR_%s_%s_H\n'%(g_name,names[count]))

        fout.write('static int %s_%s[]={\n'% (g_name,names[count]))
        weight1 = weight.reshape(weight.shape[0], -1)
        # print(weight1[0])
        if len(weight.shape) == 4:
            # conv weight,reverse
            weight1 = np.asarray([item[-1::-1] for item in weight1])
        if len(weight.shape) == 2:
            # fc1 weight,reverse
            weight1 = np.transpose(weight1)
            # print(weight1[0])

        np.savetxt(fout,
                   weight1 * 1e5,
                   fmt='%i',
                   delimiter=',',
                   newline=',\n')
        fout.write('};\n')
        fout.write('#endif\n')
        count += 1
        # for s in weight.shape:
        #     fout.write(struct.pack('i', s))
        #
        # for item in weight.flatten():
        #     fout.write(struct.pack('f', item))
        fout.close()


def save_img_to_bininary_file(test_X, test_y, name='val'):
    # 将图片保存成 二进制 形式
    with open(os.path.join(model_root_path, 'Data/images_%sdata.mat' % name), 'wb') as fout:
        print(test_X.shape)
        fout.write(struct.pack('i', len(test_X)))

        for in_data in test_X:
            # print(weight[0][0])
            # quit()
            for item in in_data.flatten():
                # print(item,ord(chr(item)))
                fout.write(struct.pack('c', chr(item)))
                # quit()

    with open(os.path.join(model_root_path, 'Data/labels_%sdata.mat' % name),
              'wb') as fout:
        fout.write(struct.pack('i', len(test_y)))
        for item in test_y:
            fout.write(struct.pack('c', chr(item)))


def save_model_file_to_pickle(type='0D'):
    '''
    将模型权重读取出来，并保存
    :return:
    '''

    if type == '0D':
        filename = '0Dbinary_model_weight.pkl'
        # model_file = 'OD_iteration860_model_weights_10-20_region3_lr0.0005.h5'
        model_file = 'OD_iteration1772_model_weights_10-20_region3_lr0.0005.h5'
        image_higth = 15
        image_width = 8
    elif type == '0Q':
        filename = '0Qbinary_model_weight.pkl'
        model_file = 'OQ_iteration70_model_weights_10-20_region3_lr0.005.h5'
        image_higth = 7
        image_width = 15
    elif type == '0G':
        filename = '0Gbinary_model_weight.pkl'
        model_file = 'OG_iteration1_model_weights_10-20_region3_lr0.01.h5'
        image_higth = 15
        image_width = 7
    else:
        raise NotImplementedError

    with open(os.path.join(model_root_path, 'letterModel', filename), 'w') as fout:

        # 加载模型架构
        # 这里的 lr 设置什么不影响
        # print(layer1, hidden1, region, image_higth, image_width)
        model, mid_output = Net_model(layer1, hidden1, region, image_higth, image_width, nb_classes=nb_classes, lr=0)
        model.summary()

        # quit()
        model.load_weights(os.path.join(model_root_path, 'letterModel', model_file))
        pickle.dump(model.get_weights(), fout)


def train_CNN_model(train_X, train_y, X_test, y_test, other_X, other_y):
    '''
    训练56二分类器
    :param train_X:
    :param train_y:
    :param X_test:
    :param y_test:
    :param other_X:
    :param other_y:
    :return:
    '''
    from keras.utils import np_utils
    model, mid_output = Net_model(layer1, hidden1, region, image_higth, image_width, nb_classes=nb_classes, lr=0.01)

    train_X = train_X[(train_y == 5) + (train_y == 6)]
    train_y = train_y[(train_y == 5) + (train_y == 6)]
    # 将5改为0
    # 将6改为1
    # 以便于CNN做2分类
    train_y[train_y == 5] = 0
    train_y[train_y == 6] = 1

    X_test = X_test[(y_test == 5) + (y_test == 6)]
    y_test = y_test[(y_test == 5) + (y_test == 6)]

    other_X = other_X[(other_y == 5) + (other_y == 6)]
    other_y = other_y[(other_y == 5) + (other_y == 6)]

    train_y = np_utils.to_categorical(train_y, 2)

    model.fit(
        train_X,
        train_y,
        nb_epoch=nb_epoch,
        shuffle=True,
        batch_size=32,
        verbose=1,
    )

    fout = open(os.path.join(model_root_path, '56binary_model_weight.pkl'), 'w')
    pickle.dump(model.get_weights(), fout)

    # model.save_weights(os.path.join(model_root_path,'56binary_model_weight.pkl'))

    predicted = model.predict_classes(X_test, verbose=0)
    # 恢复标签
    predicted[predicted == 0] = 5
    predicted[predicted == 1] = 6

    print(np.mean(predicted == y_test))

    predicted = model.predict_classes(other_X, verbose=0)
    # 恢复标签
    predicted[predicted == 0] = 5
    predicted[predicted == 1] = 6

    print(np.mean(predicted == other_y))


def binary_class_test():
    '''
    二进制分类的测试
    :return:
    '''
    # 开始位置
    start = 0
    with open(os.path.join(model_root_path, '56binary_model_weight.pkl'), 'r') as fin:
        for index in range(1, len(model_file_list_path) + 1):
            # 从 模型1 开始，依次往后
            # 找到对应模型文件

            if index < start:
                weights = pickle.load(fin)
                continue

            weights = pickle.load(fin)

            # save_cnn_weight_to_bininary_file(weights_56)
            # quit()
            # print(np.mean(predicted == y_val))
            #
            # print('OK')
            start = time.time()
            int_predict = cnn_batch_predict(X_test, weights)
            int_predict[int_predict == 0] = 5
            int_predict[int_predict == 1] = 6
            print(int_predict)
            print(index, count_result(int_predict, y_test))
            # 将结果保存
            # pickle.dump(int_predict,predict_result_34class_file)
            end = time.time()
            print('time:%ds' % (end - start))

            # if (np.mean(int_predict == predicted)) != 1.0:
            #     print(index, model_file)
            #     print(predicted)
            #     print(int_predict)
            #     print(np.mean(int_predict == predicted))
            #
            # # if (index + 1) % 10 == 0:
            # #     print(index)
            # # assert (np.mean(int_predict == predicted)) == 1.0
            #
            # (val_accuracy, val5, val10) = test_model(model, X_val, y_val)
            # print('验证集：%f,%d,%d' % (val_accuracy, val5, val10))
            break
            # (test_accuracy, test5, test10) = test_model(model, X_test, y_test)
            # print('测试集：%f,%d,%d' % (test_accuracy, test5, test10))
            # (other_accuracy, other5, other10) = test_model(model, X_other, y_other)
            # print('other集：%f,%d,%d' % (other_accuracy, other5, other10))


def generation_badcese(s):
    """将混淆集的结果生成矩阵形式，便于复制到excel

    :param s: str
    :return:

    Example
    ---
    >>> s='''{'3,8': 2, 'X,Y': 1, '3,S': 2, '4,A': 1, '0,D': 1, '0,C': 1, '5,L': 2, '2,Z': 2, 'G,Q': 1, 'H,K': 1}
    >>>     {'A,N': 1, '6,K': 1, '0,Q': 2, 'F,P': 1, 'C,G': 1, 'E,F': 2, '6,8': 2, '0,D': 2, '2,Z': 2}
    >>>     {'0,C': 1, '8,Y': 1, 'F,P': 2, '5,B': 1, '5,S': 1, '1,T': 2, '3,B': 2}'''
    >>> generation_badcese(s)
    """
    # 收集 混淆集 集合
    key_set = []
    for item in s.split('\n'):
        a = eval(item)
        key_set += a.keys()

    key_set = sorted(set(key_set))
    print(key_set)

    result = []
    for item in s.split('\n'):
        a = eval(item)
        for i in key_set:
            if i not in a.keys():
                a[i] = 0
                #     print(a)

        print([item[1] for item in sorted(a.items(), key=lambda x: x[0])])

        result.append(sorted(a.items(), key=lambda x: x[0]))
        #     break

        # print(result)


s = '''{'0,6': 4, '6,B': 1, '1,9': 1, '1,4': 1, '6,8': 9, 'H,K': 2, '3,8': 4, '5,6': 2, 'G,Q': 4, '8,L': 1, '9,S': 4, '7,P': 1, '7,T': 1, '8,B': 6, '4,A': 2, '9,H': 5, '0,Q': 15, '1,T': 14, 'C,G': 4, '6,E': 6, '0,G': 1, 'D,Q': 2, '0,D': 8, '0,C': 5, '2,Z': 20, 'B,G': 2, '1,B': 2}
{'D,Q': 2, '4,Q': 1, '6,E': 1, '0,Q': 15, '1,7': 9, '1,T': 1, '6,B': 2, '5,6': 2, '8,B': 8, '6,G': 4, '6,8': 2, '4,A': 2, '0,D': 8, '9,B': 2, '8,Z': 2, '2,Z': 1}
{'7,Z': 2, 'J,T': 2, '4,Q': 1, 'E,F': 1, '3,8': 4, '9,S': 2, '0,Q': 15, '1,Y': 2, '1,7': 1, '1,T': 2, '5,6': 2, '8,B': 8, '6,8': 1, 'D,Q': 2, '0,D': 8, '0,C': 5, 'F,P': 1, '5,S': 3, '3,B': 2}
{'1,7': 9, '6,8': 5, '3,8': 7, '4,6': 4, 'F,P': 3, '5,6': 2, '3,B': 3, 'G,Q': 1, '9,S': 2, '7,T': 2, '8,B': 10, '4,A': 3, '5,B': 1, '4,M': 1, '0,Q': 15, '1,Y': 3, '1,W': 2, '1,T': 2, 'C,G': 1, 'D,Q': 2, '0,D': 8, '0,C': 2}
{'D,Q': 2, '7,T': 2, '9,P': 2, '0,Q': 15, '6,E': 2, '1,T': 11, '5,6': 2, '8,B': 8, '6,G': 2, 'C,Q': 3, '0,G': 2, 'F,P': 1, '4,A': 5, '0,D': 8, '0,C': 2, 'G,Q': 1, '2,Z': 9, '5,B': 2, '3,8': 3, '1,4': 1}
{'J,T': 6, '0,3': 2, '1,7': 25, '6,8': 2, '2,9': 1, '3,8': 5, '5,6': 2, 'K,Y': 1, '9,S': 5, '7,T': 2, '8,B': 8, '9,B': 2, '9,J': 2, '0,Q': 15, 'C,D': 5, '1,T': 20, 'C,G': 1, '6,E': 3, '6,F': 2, '0,G': 2, '0,D': 6, '0,C': 16, 'E,S': 3}
{'1,7': 3, '6,8': 3, '5,9': 3, 'F,P': 1, '5,6': 2, 'G,Q': 1, 'K,X': 1, '4,Q': 2, '9,S': 2, '7,T': 13, '8,B': 6, '9,E': 8, '5,F': 5, '3,B': 2, '6,E': 3, '0,Q': 15, '1,T': 6, 'E,F': 1, 'B,P': 2, '6,G': 2, 'B,R': 2, '0,G': 4, 'D,Q': 2, '0,D': 8, '0,C': 19, '0,L': 3}
{'6,B': 1, '1,7': 3, '6,8': 1, '5,9': 4, 'F,P': 4, '5,6': 2, 'F,J': 2, '4,M': 1, '8,9': 2, 'N,X': 3, 'M,Y': 7, 'G,U': 2, '7,Z': 1, '9,S': 5, '8,B': 8, '4,A': 1, 'R,Z': 2, '5,B': 1, '5,F': 2, '3,B': 2, '0,Q': 15, '1,T': 3, 'C,G': 1, 'D,Q': 2, '0,D': 8, '0,C': 10, '2,Z': 1, '1,G': 2, '6,S': 2}
{'J,T': 9, 'A,K': 7, '1,7': 6, '6,8': 11, '5,8': 1, '5,9': 4, '3,8': 3, '3,5': 1, '5,6': 2, 'X,Y': 5, 'G,Q': 1, '7,Z': 1, '8,B': 6, '4,A': 7, '9,B': 4, '0,Q': 15, 'C,G': 1, 'B,Q': 1, '0,G': 2, 'D,Q': 2, '0,D': 8, '0,C': 10}
{'1,7': 1, '1,3': 1, '6,8': 11, 'P,R': 2, '5,6': 2, 'G,Q': 4, 'K,Y': 1, '9,S': 7, '7,T': 2, '8,B': 8, '5,K': 7, '4,A': 3, '5,F': 2, '3,B': 2, '0,Q': 15, '1,Y': 11, '1,T': 20, '6,B': 6, '6,G': 2, 'D,Q': 2, '0,D': 8, '0,C': 3, '2,Z': 7}
{'D,Q': 2, 'C,G': 4, '7,T': 5, '0,Q': 15, '0,1': 2, 'J,T': 2, '5,S': 4, '1,T': 3, '9,S': 6, '5,6': 2, '1,3': 2, '8,B': 10, '9,D': 1, '0,D': 6, '0,C': 1, '4,G': 1, '6,S': 2, '1,C': 1, '4,L': 1}
{'D,Q': 2, '0,U': 5, '3,8': 9, '9,S': 2, '0,Q': 15, 'J,T': 2, '7,T': 1, '1,T': 14, '5,6': 2, '8,B': 8, '6,8': 8, '4,A': 4, '0,D': 8, '9,B': 2, '2,Z': 3, '6,S': 2}
{'F,Y': 1, 'D,Q': 2, '6,K': 1, '3,8': 6, '1,7': 1, '0,Q': 15, '5,S': 3, '1,T': 3, '6,E': 3, '8,B': 8, '4,A': 1, '0,D': 8, '0,C': 25, '2,Z': 22, '5,B': 5, 'G,Q': 1, '9,S': 6, '7,T': 1, '1,4': 2, '5,6': 2}
{'4,7': 1, '9,S': 1, '0,Q': 15, '7,T': 15, '1,T': 7, 'C,G': 2, '5,6': 2, '8,B': 8, '6,8': 2, 'D,Q': 2, '0,D': 8, '0,C': 6, '4,Z': 2, 'G,Q': 3, 'C,Q': 2, '3,B': 2}
{'J,T': 2, '1,7': 6, '6,8': 7, '2,3': 1, '3,8': 7, '5,6': 2, '4,W': 1, '8,B': 8, '3,S': 5, '4,A': 15, '9,B': 2, '8,S': 29, '3,B': 5, '0,Q': 15, '1,T': 3, '6,E': 12, 'D,Q': 2, '0,D': 8, '0,C': 10, '2,Z': 2, '1,G': 4, '6,S': 2, 'C,Q': 1}
{'D,Q': 2, '0,3': 2, 'X,Y': 1, '0,Q': 15, '1,7': 7, '1,T': 16, '5,6': 2, '8,B': 8, 'F,P': 1, '6,8': 2, '4,A': 1, '0,D': 6, '0,C': 2, 'B,H': 2, 'G,Q': 1, '2,Z': 3, '5,B': 1, 'A,W': 2, '5,S': 6, '1,4': 1}
{'J,T': 4, '1,7': 2, '6,8': 10, '3,8': 4, '3,9': 1, 'F,P': 5, 'M,N': 5, '8,9': 1, 'M,X': 2, 'K,X': 3, '4,Q': 1, '9,S': 8, '7,T': 2, '8,B': 8, '3,B': 2, '5,6': 2, '0,Q': 15, 'C,D': 3, '1,T': 2, '0,G': 1, 'D,Q': 2, '0,D': 8, '1,J': 3}
{'J,T': 2, '1,7': 4, '3,8': 1, '3,9': 1, '5,6': 2, 'D,P': 7, '9,S': 4, '8,B': 8, '5,T': 4, '4,A': 14, '9,B': 2, '4,F': 1, 'C,L': 3, '0,Q': 15, '1,Y': 1, 'C,D': 3, '1,T': 23, '6,E': 6, '0,G': 2, 'D,Q': 2, '0,D': 8, '0,C': 6, '2,Z': 2, '1,F': 2, '1,D': 2, '1,C': 2}
{'7,T': 3, '0,2': 2, '0,Q': 15, '6,E': 1, '1,T': 21, 'C,G': 2, '5,6': 2, '8,B': 8, 'L,U': 3, '0,G': 1, 'D,Q': 2, '0,D': 6, '0,C': 23, 'X,Y': 1, '2,Z': 19, 'G,Q': 3, '9,S': 4, '3,B': 2, '4,6': 1}
{'J,T': 5, '6,8': 9, '9,S': 12, '5,9': 2, '3,8': 2, 'F,P': 2, '5,6': 2, 'G,Q': 1, '4,Q': 1, '3,Y': 1, '9,Q': 2, '8,E': 3, '7,T': 28, '8,B': 8, '4,A': 7, '3,E': 1, '0,Q': 15, '1,T': 20, 'C,G': 1, '6,E': 1, 'D,Q': 2, '0,D': 8, '0,C': 8, '2,Z': 12, 'B,E': 2}
{'D,Q': 2, '7,T': 5, '3,9': 3, '0,Q': 15, '5,S': 4, '1,T': 3, '6,B': 2, '5,6': 2, '8,B': 8, '6,8': 1, '4,A': 3, '0,D': 8, '4,G': 1, '2,Z': 19, '5,B': 3, 'G,Q': 1, 'K,X': 2, 'K,Y': 1}'''
# generation_badcese(s)
# quit()
# region 读取数据集：验证数据(64369个)、测试数据(64381个)、其他应用数据集(243391个)
(X_val, y_val), (X_test, y_test), (X_other, y_other), (other_X_new, other_y_new) = load_valdata(version='1214')

print(X_test.shape)
print(X_other.shape)
print(other_X_new.shape)

# test_fin = open('/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/Data1129/pm_TestSet.pickle')
# X_test = pickle.load(test_fin)
# y_test = pickle.load(test_fin)
# print([character_name[item] for item in y_test])

# region CNN 模型的训练
# train_CNN_model(X_train, y_train, X_test, y_test, X_other, y_other)
# quit()
# endregion
# region 保存图片到二进制文件
# save_img_to_bininary_file(X_val,y_val,name='val')
# save_img_to_bininary_file(X_test, y_test,name='test')
# save_img_to_bininary_file(X_other, y_other,name='other')
# endregion
# region 将模型权重保存
# save_model_file_to_pickle(type='0D')
# save_model_file_to_pickle(type='0Q')
# save_model_file_to_pickle(type='0G')
# quit()
# endregion
# region 测试模型
# print(
#     test_model(os.path.join(model_root_path, 'letterModel', 'OG_iteration1_model_weights_10-20_region3_lr0.01.h5'),
#                X_test, y_test, X_other, y_other,
#                type='0G',
#                ))
# print(
#     test_model(os.path.join(model_root_path, 'letterModel', 'OQ_iteration70_model_weights_10-20_region3_lr0.005.h5'),
#                X_test, y_test, X_other, y_other,
#                type='0Q',
#                ))
# quit()
# endregion
start = time.time()

predict_result_25class_file = open(os.path.join(model_root_path,'letterModel', 'letter_result.pkl'), 'r')

# 用二分类器修正34分类结果
# 34分类前21个模型的预测结果
# run_id = [129, 141, 245, 249, 270, 287, 300, 311, 375, 425, 509, 543, 630, 758, 864, 875, 890, 905, 975, 1014]
type=0
if type==0:
    test_int_predict_25class = np.asarray(pickle.load(predict_result_25class_file))
elif type ==1:
    test_int_predict_25class = np.asarray(pickle.load(predict_result_25class_file))
    test_int_predict_25class = np.asarray(pickle.load(predict_result_25class_file))
    X_test = X_other
    y_test=y_other
elif type ==2:
    test_int_predict_25class = np.asarray(pickle.load(predict_result_25class_file))
    test_int_predict_25class = np.asarray(pickle.load(predict_result_25class_file))
    test_int_predict_25class = np.asarray(pickle.load(predict_result_25class_file))
    X_test = other_X_new
    y_test=other_y_new
else:
    raise NotImplementedError

print(count_result(test_int_predict_25class, y_test))

# 读取2分类器的权重 - 0-D
weights_0D = pickle.load(open(os.path.join(model_root_path, 'letterModel','0Dbinary_model_weight.pkl'), 'r'))
# save_cnn_weight_to_bininary_file(weights_0D,type='0D')
# quit()
weights_0Q = pickle.load(open(os.path.join(model_root_path, 'letterModel','0Qbinary_model_weight.pkl'), 'r'))
# save_cnn_weight_to_bininary_file(weights_0Q, type='0Q')

# 读取2分类器的权重 - 0-D
weights_0G = pickle.load(open(os.path.join(model_root_path, 'letterModel','0Gbinary_model_weight.pkl'), 'r'))
# save_cnn_weight_to_bininary_file(weights_0G,type='0G')

# quit()
# region 被预测成0D的数据 ----> 左半边
idx_predicted_0D = (test_int_predict_25class == char_to_index('0')) + (test_int_predict_25class == char_to_index('D'))
print('预测成0-D的个数:%d' % sum(idx_predicted_0D))

binary_result_0D = cnn_batch_predict(X_test[idx_predicted_0D][:, :, :, :8], weights_0D)
binary_result_0D[binary_result_0D == 0] = char_to_index('0')
binary_result_0D[binary_result_0D == 1] = char_to_index('D')
test_int_predict_25class[idx_predicted_0D] = binary_result_0D
print(count_result(test_int_predict_25class, y_test))

# model, mid_output = Net_model(layer1, hidden1, region, 15, 8, nb_classes=2, lr=0.01)
# model.set_weights(weights_0D)
# binary_result_0D = model.predict_classes(X_test[idx_predicted_0D][:, :, :, :8], verbose=0)
# binary_result_0D[binary_result_0D == 0] = char_to_index('0')
# binary_result_0D[binary_result_0D == 1] = char_to_index('D')
# # print(count_result(binary_result_0D, y_test[idx_predicted_0D]))
#
# test_int_predict_25class[idx_predicted_0D] = binary_result_0D
# print(count_result(test_int_predict_25class, y_test))

# endregion

# region 被预测成0Q的数据 ---> 下半边
idx_predicted_0Q = (test_int_predict_25class == char_to_index('0')) + (test_int_predict_25class == char_to_index('Q'))
print('预测成0-Q的个数:%d' % sum(idx_predicted_0Q))
print(X_test[idx_predicted_0Q][0][0])
binary_result_0Q = cnn_batch_predict(X_test[idx_predicted_0Q][:, :, 8:, :], weights_0Q)
binary_result_0Q[binary_result_0Q == 0] = char_to_index('0')
binary_result_0Q[binary_result_0Q == 1] = char_to_index('Q')
test_int_predict_25class[idx_predicted_0Q] = binary_result_0Q
print(count_result(test_int_predict_25class, y_test))

# model, mid_output = Net_model(layer1, hidden1, region, 7, 15, nb_classes=2, lr=0.01)
# model.set_weights(weights_0Q)
# binary_result_0Q = model.predict_classes(X_test[idx_predicted_0Q][:, :, 8:, :], verbose=0)
# binary_result_0Q[binary_result_0Q == 0] = char_to_index('0')
# binary_result_0Q[binary_result_0Q == 1] = char_to_index('Q')
# # print(count_result(binary_result_0Q, y_test[idx_predicted_0Q]))
#
# test_int_predict_25class[idx_predicted_0Q] = binary_result_0Q
# print(count_result(test_int_predict_25class, y_test))

# endregion

# region 被预测成0G的数据 ----> 右半边
idx_predicted_0G = (test_int_predict_25class == char_to_index('0')) + (test_int_predict_25class == char_to_index('G'))
print('预测成0-G的个数:%d' % sum(idx_predicted_0G))
print(X_test[idx_predicted_0G][0][0])
binary_result_0G = cnn_batch_predict(X_test[idx_predicted_0G][:, :, :, 8:], weights_0G)
binary_result_0G[binary_result_0G == 0] = char_to_index('0')
binary_result_0G[binary_result_0G == 1] = char_to_index('G')
# print(count_result(binary_result_0G, y_test[idx_predicted_0G]))
test_int_predict_25class[idx_predicted_0G] = binary_result_0G

print(count_result(test_int_predict_25class, y_test))

# model, mid_output = Net_model(layer1, hidden1, region, 15, 7, nb_classes=2, lr=0.01)
# model.set_weights(weights_0G)
# binary_result_0G = model.predict_classes(X_test[idx_predicted_0G][:, :, :, 8:], verbose=0)
# binary_result_0G[binary_result_0G == 0] = char_to_index('0')
# binary_result_0G[binary_result_0G == 1] = char_to_index('G')
# # print(count_result(binary_result_0G, y_test[idx_predicted_0G]))
#
# test_int_predict_25class[idx_predicted_0G] = binary_result_0G
# print(count_result(test_int_predict_25class, y_test))

# endregion


end = time.time()
print('time:%ds' % (end - start))

# break