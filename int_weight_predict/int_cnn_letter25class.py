# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-11-18'; 'last updated date: 2016-11-18'
    Email:   '383287471@qq.com'
    Describe: 整形权重的预测,34分类
"""
from __future__ import print_function
import numpy as np
import struct
# from data_processing_util.data_util import DataUtil
import time
import pickle
import os

character_name = list('ABCDEFGHIJKLMN0PQRSTUWXYZ')
project_path = '/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/modelAndData1214/'
np.random.seed(1337)  # for reproducibility
nb_classes = 25
nb_epoch = 30
batch_size = 128

image_higth, image_width = 15, 15
# lr = [0.05, 0.01, 0.005]
layer1 = 10
hidden1 = 40
region = 3
# dutil = DataUtil()
# 模型权重根目录
model_root_path = '/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/modelAndData1214/'
model_file_list_path = os.listdir(os.path.join(model_root_path, 'letterModel'))


# 过滤数字，得到字母，并按0-24编码A-Z,注意没有V
def filterData(X, y):
    y_letter = [];
    index = []
    for i in range(0, len(y)):
        if y[i] >= 10:
            index.append(i)
            label = y[i];
            if label >= 24:
                label += 1;
            y_letter.append(label - 10)
        elif y[i] == 0:
            index.append(i);
            y_letter.append(14)
    X_letter = X[index]
    return (X_letter, y_letter)


def load_valdata(version='1122'):
    """读取不同版本的测试集
    1120 - 对应 /home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/modelAndData1120/Data/
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
            test_X = pickle.load(train_file)
            test_y = pickle.load(train_file)

        with open(other_file_path, 'rb') as otherFile:
            other_X = pickle.load(otherFile)
            other_y = pickle.load(otherFile)

    elif version == '1120':
        val_file_path = os.path.join(data_root_path, 'modelAndData1120/Data/', 'valSet&TestSet.pickle')
        other_file_path = os.path.join(data_root_path, 'modelAndData1120/Data/', 'Olddata_TestSet.pickle')
        with open(val_file_path, 'rb') as train_file:
            val_X = pickle.load(train_file)
            val_y = pickle.load(train_file)
            test_X = pickle.load(train_file)
            test_y = pickle.load(train_file)

        with open(other_file_path, 'rb') as otherFile:
            other_X = pickle.load(otherFile)
            other_y = pickle.load(otherFile)

    elif version == '1128':
        val_file_path = os.path.join(data_root_path, '9905_1128/', 'TestSet.pickle')
        with open(val_file_path, 'rb') as train_file:
            val_X = None
            val_y = None
            test_X = pickle.load(train_file)
            test_y = pickle.load(train_file)
            other_X, other_y = None, None
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

        with open(other_file_path, 'rb') as otherFile:
            other_X = pickle.load(otherFile)
            other_y = pickle.load(otherFile)

        (other_X, other_y) = filterData(other_X, other_y)

        with open(new_other_file_path, 'rb') as otherFile:
            other_X_new = pickle.load(otherFile)
            other_y_new = pickle.load(otherFile)
        (other_X_new, other_y_new) = filterData(other_X_new, other_y_new)

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


def test_model(model_file, X_test, Y_test):
    # 加载模型架构
    # 这里的 lr 设置什么不影响
    model, mid_output = Net_model(layer1, hidden1, region, image_higth, image_width, nb_classes=34, lr=0)
    model.load_weights(model_file)

    # 预测
    predicted = model.predict_classes(X_test, verbose=0)

    return count_result(predicted, Y_test)


def count_result(predicted, Y_test):
    '''统计预测情况，包括混淆集个数等

    :return:
    '''
    # 统计混淆集
    badcase = {}
    for i in range(0, len(Y_test)):
        # if (tramsform(predicted[i]) in ['1', 'I'] and tramsform(Y_test[i]) in ['1', 'I']):
        # 1跟I区分，只要 是 测试 成1或I，而实际值是 1或I都算对
        # predicted[i] = Y_test[i]
        if predicted[i] != Y_test[i]:
            ch1 = tramsform(Y_test[i])
            ch2 = tramsform(predicted[i])
            string = ','.join(sorted([ch1, ch2]))

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


def tramsform(num):
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
                # print(np.dot(img[0, x:x + filter_row, y:y + filter_col].flatten(),j_1)+ conv_b[filter_index])

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
    # start = time.time()

    imgs_conv_result = conv_pool_operation(img, weights[0], weights[1])
    # end = time.time()
    # print('conv time:%fs' % (end - start))

    # start = time.time()

    # print('conv over..')
    flatten_result = imgs_conv_result.flatten()

    # end = time.time()
    # print('flatten time:%fs' % (end - start))

    # start = time.time()
    # print('flatten over..')
    # print(flatten_result.shape)
    hidden1_output = hidden_operation(flatten_result, weights[2], weights[3], activion='tanh')
    # end = time.time()
    # print('hidden1 time:%fs' % (end - start))

    # start = time.time()
    # print('hidden1 over..')
    hidden2_output = hidden_operation(hidden1_output, weights[4], weights[5], activion='none')
    # end = time.time()
    # print('hidden2 time:%fs' % (end - start))
    # print('hidden2 over..')
    # print(hidden2_output.shape)
    # start = time.time()
    result = np.argmax(hidden2_output)
    # end = time.time()
    # print('max time:%fs' % (end - start))
    return result


def save_cnn_weight_to_bininary_file(model_weights):
    # print(model_weights)
    # print(len(model_weights))
    g_name = 'Letter'
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
        fout = open(os.path.join(model_root_path, 'int_weights', 'letter25_int_weight%d.h' % count), 'w')
        fout.write('//%s\n' % str(weight.shape))
        fout.write('#ifndef DR_%s_%s_H\n' % (g_name, names[count]))
        fout.write('#define DR_%s_%s_H\n' % (g_name, names[count]))

        fout.write('static int %s_%s[]={\n' % (g_name, names[count]))
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
    with open(os.path.join(model_root_path, 'Data/images_letter25_%sdata.mat' % name), 'wb') as fout:
        print(test_X.shape)
        fout.write(struct.pack('i', len(test_X)))

        for in_data in test_X:
            # print(weight[0][0])
            # quit()
            for item in in_data.flatten():
                # print(item,ord(chr(item)))
                fout.write(struct.pack('c', chr(item)))
                # quit()

    with open(os.path.join(model_root_path, 'Data/labels_letter25_%sdata.mat' % name),
              'wb') as fout:
        fout.write(struct.pack('i', len(test_y)))
        for item in test_y:
            fout.write(struct.pack('c', chr(item)))


def save_model_file_to_pickle(type=1):
    '''
    将模型权重读取出来，并保存
    :return:
    '''
    if type == 1:
        with open(os.path.join(model_root_path, 'model_weight.pkl'), 'w') as fout:
            for index in range(1, len(model_file_list_path) + 1):
                # 从 模型1 开始，依次往后
                # 找到对应模型文件
                for item in model_file_list_path:
                    if item.__contains__('iteration%d_model_weights_10-40_region3_' % index):
                        model_file = item
                        break

                if (index + 1) % 10 == 0:
                    print(index + 1)

                # 加载模型架构
                # 这里的 lr 设置什么不影响
                model, mid_output = Net_model(layer1, hidden1, region, image_higth, image_width, nb_classes=34, lr=0)
                # model.summary()

                # quit()
                model.load_weights(os.path.join(model_root_path, 'model', model_file))
                pickle.dump(model.get_weights(), fout)

                # print(model_file)
    elif type == 2:
        with open(
                '/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/modelAndData1214/letterModel/25class_model_weight.pkl',
                'w') as fout:
            model, mid_output = Net_model(layer1, hidden1, region, image_higth, image_width, nb_classes=nb_classes,
                                          lr=0)
            model.summary()

            # quit()
            model.load_weights(
                '/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/modelAndData1214/letterModel/iteration73_model_weights_10-40_region3_lr0.01_firstCNN_final.h5')
            pickle.dump(model.get_weights(), fout)


def predict(images):
    """使用 141 模型预测

    :param images:
    :return:
    """
    start = 141
    with open(os.path.join(model_root_path, '34class_model_weight.pkl'), 'r') as fin:
        for index in range(1, len(model_file_list_path) + 1):
            # 从 模型1 开始，依次往后
            # 找到对应模型文件
            weights = pickle.load(fin)

            if index != start:
                continue

            # save_cnn_weight_to_bininary_file(weights)
            # quit()
            # print(np.mean(predicted == y_val))
            #
            # print('OK')
            start = time.time()
            int_predict = cnn_batch_predict(images, weights)
            # print(int_predict, character_name[int_predict])
            return character_name[int_predict]


# region 读取数据集：验证数据(64369个)、测试数据(64381个)、其他应用数据集(243391个)
(X_val, y_val), (X_test, y_test), (X_other, y_other), (other_X_new, other_y_new) = load_valdata(version='1214')

# quit()
# 真实值              ['0', '0', '0', '0', '0', '0', '0', 'C']
# 34分类              ['Q', 'D', '0', '0', '0', '0', 'Q', 'Q']
# 加0-D-Q分类器        ['D', 'D', '0', 'D', '0', '0', 'Q', 'Q']

# 真实值              ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '6', '6', 'G', 'G']
# 34分类              ['0', '0', '0', '0', 'U', '0', 'C', '0', 'Q', '0', 'C', '0', '0', '6', '6', 'D', 'C']
# 加分类器             ['Q', 'Q', '0', 'Q', 'U', '0', 'C', '0', '0', '0', 'C', 'D', '0', '5', '5', 'Q', 'C']

# test_fin = open('/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/Data1129/pm_TestSet.pickle')
# X_test = pickle.load(test_fin)
# y_test = pickle.load(test_fin)
# images = dutil.load_image_from_file('/home/jdwang/Desktop/误识别 20161129Pm/0-4/201611291428255649.bmp')
# X_test = images.reshape(1,1,15,15)
# y_test = [0]
# print([character_name[item] for item in y_test])
# print(X_test.shape)
# save_img_to_bininary_file(X_test,y_test,name='test1129pm01')
# quit()
# print(X_other.shape)

# region 预测一张图片
# image_root = '/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/9905_1128/data/E'
# for image_path in sorted(os.listdir(image_root)):
#     images = dutil.load_image_from_file(os.path.join(image_root,image_path))
#     # print(images)
#     images = images.reshape((1,1,15,15))
#     print(image_path,predict(images))
#
# quit()
# endregion



# save_img_to_bininary_file(X_val,y_val,name='val')
save_img_to_bininary_file(X_test, y_test,name='test')
# save_img_to_bininary_file(X_other, y_other,name='other')
# 将模型权重保存
# save_model_file_to_pickle(type=2)
quit()
# endregion
# 测试模型
# print(test_model(os.path.join(model_root_path,'model', 'iteration141_model_weights_10-40_region3_lr0.005_firstCNN_final.h5'),
#            X_test,
#            y_test))
# quit()


# 开始位置
start = 1
result_fout = open(
    '/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/modelAndData1214/letterModel/letter_result.csv',
    'w')

with open(os.path.join(project_path, 'letterModel', '25class_model_weight.pkl'), 'r') as fin:
    for index in range(1, len(model_file_list_path) + 1):
        # 从 模型1 开始，依次往后
        # 找到对应模型文件
        weights = pickle.load(fin)

        if index != start:
            continue

        save_cnn_weight_to_bininary_file(weights)
        quit()
        # print(np.mean(predicted == y_val))
        #
        # print('OK')
        # X_test=X_test
        # y_test=y_test

        start = time.time()
        int_predict = cnn_batch_predict(X_test, weights)
        print(int_predict)
        print([character_name[item] for item in int_predict])
        print(index, count_result(int_predict, y_test))
        pickle.dump(int_predict, result_fout)
        end = time.time()
        print('time:%ds' % (end - start))

        start = time.time()
        int_predict = cnn_batch_predict(X_other, weights)
        print(int_predict)
        print([character_name[item] for item in int_predict])
        print(index, count_result(int_predict, y_other))
        pickle.dump(int_predict, result_fout)

        end = time.time()
        print('time:%ds' % (end - start))

        start = time.time()
        int_predict = cnn_batch_predict(other_X_new, weights)
        print(int_predict)
        print([character_name[item] for item in int_predict])
        print(index, count_result(int_predict, other_y_new))

        pickle.dump(int_predict, result_fout)
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
