# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-01-01'; 'last updated date: 2017-01-01'
    Email:   '383287471@qq.com'
    Describe: 字符的二分类方法，对于 A 是 B的局部的字符，通过比对 差异处的 灰度值大小来判定 字符类别归属
        - P-R :  8:14,8:14
        - E-F : 对区域 [7:,::] 进行二值化，然后取小区域的  [3:7,6:14]    F:：0-1，E：7-24
        - I-T : 对区域 [1:5,1:6]，[1:5,9:14] 求灰度和  I：0-44   T：2676-6481  临界值：1316
        - 对 T进行修正，取区域 [7:14,1:14] 进行二值化，取这个小区域的左下角 [5::,0:2]，如果这个区域有黑点，就修正为 J
        - 7-Z : 对区域 [7:14,1:14] 进行二值化，然后取这个小区域的左下角（4*4，3::,0:4）和右下角（4*4，3::,9::），选择点数最小的这个区域来计算，7:0-1   Z：3-12
        - 5-6：区域 [6:14,1:7]， 对区域进行二值化， 并判断是否有环
        - 3-8：区域 [1:14,1:6]， 对区域进行二值化， 并判断是否有环
        - C-0：区域 [1:14,9:14], 对区域进行二值化， 并判断是否有环
        - 6-E：区域 [5:14,9:14], 对区域进行二值化， 并判断是否有环
        - P-F：区域 [1:9,8:14] , 对区域进行二值化， 并判断是否有环
"""

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
    else:
        raise NotImplementedError

    return (train_X, train_y), (test_X, test_y), (other_X, other_y), (other_X_new, other_y_new)


def char_to_index(char):
    if character_name.__contains__(char):
        return character_name.index(char)
    else:
        raise NotImplementedError


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


(train_X, train_y), (test_X, test_y), (other_X, other_y), (other_X_new, other_y_new) = load_valdata(version='1214')

X = other_X_new[other_y_new == char_to_index('F')]

# print(X)
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
