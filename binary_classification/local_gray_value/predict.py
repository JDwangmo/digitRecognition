# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-01-01'; 'last updated date: 2017-01-03'
    Email:   '383287471@qq.com'
    Describe: 字符的二分类方法，对于 A 是 B的局部的字符，通过比对 差异处的 灰度值大小来判定 字符类别归属
        - R-P :  8:14,8:14
        - I-T : 对区域 [1:5,1:6]，[1:5,9:14] 求灰度和  I：0-44   T：2676-6481  临界值：1316
        - T-J : 只对上面预测为T的进行修改， 对区域 [7:14,1:14] 进行二值化，取这个小区域的左下角 [5::,0:2]，如果这个区域有黑点，就修正为 J
        - E-F : 对区域 [7:,::] 进行二值化，然后取小区域的  [3:7,6:14]    F:：0-1，E：7-24  临界值：3
        - 7-Z : 对区域 [7:14,1:14] 进行二值化，然后取这个小区域的左下角（4*4，3::,0:4）和右下角（4*4，3::,9::），选择点数最小的这个区域来计算，7:0-1   Z：3-12
        - 3-8：区域 [1:14,1:6]， 对区域进行二值化， 并判断是否有环
        - 8-6：区域 [1:8,8:14]， 对区域进行二值化， 并判断是否有环，单向 8 修正为 6
        - 0-C：区域 [1:14,9:14], 对区域进行二值化， 并判断是否有环,单向 C 修正为 0
        - P-F：区域 [1:9,8:14] , 对区域进行二值化， 并判断是否有环
        - 6-E：区域 [5:14,9:14], 对区域进行二值化， 并判断是否有环
        - 5-6：区域 [6:14,1:7]， 对区域进行二值化， 并判断是否有环
"""
import struct

import numpy as np
from PIL import Image
import matplotlib.image as IMG
import os
import pickle

nb_classes = 2

nb_epoch = 30

batch_size = 64

character_name = list('0123456789ABCDEFGHIJKLMNPQRSTUWXYZ')


def load_valdata(version='1214'):
    """读取不同版本的测试集

    :param dir_path: str
    :param version: str
    :return:
    """

    # 数据集根目录
    data_root_path = '/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/'

    # 读取验证数据、测试数据
    if version == '1214':
        val_file_path = os.path.join(data_root_path, 'modelAndData1214/Data/', 'TrainSet_trainAndVal_testSet.pickle')
        other_file_path = os.path.join(data_root_path, 'modelAndData1214/Data/', 'OldAndAlldata_TestSet.pickle')
        new_other_file_path = os.path.join(data_root_path, 'modelAndData1214/Data/', '20161202am.pickle')
        test20170103_file_path = os.path.join(data_root_path, 'modelAndData1214/Data/', '20170103TestSet.pickle')
        with open(val_file_path, 'rb') as train_file:
            train_X = pickle.load(train_file)
            train_y = pickle.load(train_file)
            test_X = pickle.load(train_file)
            test_y = pickle.load(train_file)
        test_y = np.asarray(test_y)
        with open(other_file_path, 'rb') as otherFile:
            other_X = pickle.load(otherFile)
            other_y = pickle.load(otherFile)
        other_y = np.asarray(other_y)

        with open(new_other_file_path, 'rb') as otherFile:
            other_X_new = pickle.load(otherFile)
            other_y_new = pickle.load(otherFile)
        other_y_new = np.asarray(other_y_new)

        with open(test20170103_file_path, 'rb') as otherFile:
            test20170103_X_new = pickle.load(otherFile)
            test20170103_y_new = pickle.load(otherFile)
        test20170103_y_new = np.asarray(test20170103_y_new)
    else:
        raise NotImplementedError

    return (train_X, train_y), (test_X, test_y), \
           (other_X, other_y), (other_X_new, other_y_new), \
           (test20170103_X_new, test20170103_y_new)


def char_to_index(char):
    if character_name.__contains__(char):
        return character_name.index(char)
    else:
        raise NotImplementedError


def save_img_to_bininary_file(test_X, test_y, name='val'):
    model_root_path = '/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/modelAndData1214'
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


def getGray(img, height, width):
    """

    :param img:
    :param height:
    :param width:
    :return:
    """
    numGray = [0 for i in range(256)]
    for h in range(height):
        for w in range(width):
            numGray[int(img[h, w])] += 1
    return numGray


def getThres(X):
    """ 寻找 图片 最优 二值化 阈值

    :param X:
    :return:
    """
    # 统计 灰度值
    gray = getGray(X, len(X[:, 0]), len(X[0, :]))

    maxV = 0
    bestTh = 0
    # 像素点累计个数
    w = [0 for i in range(len(gray))]
    # 像素值 累计大小
    px = [0 for i in range(len(gray))]
    w[0] = gray[0]
    px[0] = 0
    for m in range(1, len(gray)):
        w[m] = w[m - 1] + gray[m]
        px[m] = px[m - 1] + gray[m] * m
    for th in range(len(gray)):
        w1 = w[th]
        w2 = w[len(gray) - 1] - w1
        if (w1 * w2 == 0):
            continue
        u1 = px[th] / w1
        u2 = (px[len(gray) - 1] - px[th]) / w2
        v = w1 * w2 * (u1 - u2) * (u1 - u2)
        if v > maxV:
            maxV = v
            bestTh = th
    return bestTh


def binaryzation(X):
    """将图片 0-1 二值化
    :return:
    """

    Thread = getThres(X)
    # 白色（=1）是前景，黑色（=0）是背景
    X = (X <= Thread) * 1
    return X


def judgeCircle(X):
    black = False
    i = 0
    while i < len(X):
        if sum(X[i]) > 0:
            break
        i += 1
    i += 1
    while i < len(X):
        if sum(X[i]) == 0:
            black = True
        if black and sum(X[i]) > 0:
            return False
        i += 1
    return True


def predict(X):
    binary_X = binaryzation(X[1:9, 8:14])  # 得到二值化图片
    circle = judgeCircle(binary_X)  # 判断是否有环，有环返回True,无环返回False
    if circle:
        # print('p')
        return 'P'
    else:
        # print('F')
        return 'F'


(train_X, train_y), (test_X, test_y), (other_X, other_y), (other_X_new, other_y_new), (
test20170103_X_new, test20170103_y_new) = load_valdata(version='1214')

# region 保存数据
# save_img_to_bininary_file(test_X, test_y,name='test_')
# save_img_to_bininary_file(other_X, other_y,name='other_')
# save_img_to_bininary_file(other_X_new, other_y_new,name='other_new_')
# save_img_to_bininary_file(test20170103_X_new, test20170103_y_new,name='test20170103_')
# endregion

from data_processing_util.data_util import DataUtil
dutil = DataUtil()
dutil.show_image_from_array(test20170103_X_new[5][0])
# X = other_X_new[other_y_new == char_to_index('Q')]
# dutil.show_image_from_array(X[0][0])

# print(other_X_new[149][0])
quit()
binary_X = binaryzation(test_X[27724][0][7:14, 1:14])  # 得到二值化图片
print(binary_X)
print(binary_X[3::, 0:4])
print(binary_X[3::, 9::])

result = []
for x in X:
    # print(x[0])
    # quit()
    result.append(predict(x[0]))
# predict(X)  # 预测
# print(np.asarray(result))
print(np.mean(np.asarray(result) == 'F'))

# quit()
#
# imgPath = 'C:\Users\hqj\Desktop\\badcase\\20161219CNN5\\20161219CNN\\P-F 2.bmp'
#
# img = IMG.imread(imgPath)
# img = np.asarray(img)
# predict(img)  # 预测
