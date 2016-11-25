# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-11-01'; 'last updated date: 2016-11-01'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
from binary_classification.dataset.data_util import DataUtil
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer, StandardScaler
import numpy as np


class KnnBinaryClassifier(object):
    def __init__(self):
        pass

    def fit(self,
            train_data=None,
            val_data=None,
            test_data=None,
            preprocessing_option=1,
            ):
        model = KNeighborsClassifier(n_neighbors=3,
                                     weights='distance',
                                     algorithm='kd_tree',
                                     leaf_size=30,
                                     p=2,
                                     metric='minkowski',
                                     metric_params=None,
                                     n_jobs=4
                                     )

        if preprocessing_option == 0:
            preprocessing = None
        elif preprocessing_option == 1:
            preprocessing = 'norm'
            preprocessing_tool = Normalizer()
        elif preprocessing_option == 2:
            preprocessing = 'scale'
            preprocessing_tool = StandardScaler()
        else:
            raise NotImplementedError

        train_X, train_y = train_data
        train_X = train_X.reshape(len(train_X), -1)

        if preprocessing is not None:
            train_X = preprocessing_tool.fit_transform(train_X)
        # print(X_train[0])
        # print(X_train[1])
        # quit()
        val_X, val_y = val_data
        val_X = val_X.reshape(len(val_X), -1)
        if preprocessing is not None:
            val_X = preprocessing_tool.transform(val_X)

        test_X, test_y = test_data
        test_X = test_X.reshape(len(test_X), -1)
        if preprocessing is not None:
            test_X = preprocessing_tool.transform(test_X)

        model.fit(train_X, train_y)

        print(model.score(train_X, train_y))
        print(model.score(val_X, val_y))
        print(model.score(test_X, test_y))

        test_pred = model.predict(test_X)
        val_pred = model.predict(val_X)

        # print(test_pred)
        # print(test_y)
        print('test：%d'%sum(test_pred != test_y))
        print('val:%d'%sum(val_pred != val_y))
        print('test+val:%d'%((sum(test_pred != test_y))+sum(val_pred != val_y)))

        print('test错误的index')
        print(np.arange(len(test_pred))[test_pred != test_y])
        print('真实值')
        print(test_y[test_pred != test_y])
        print('预测值')
        print(test_pred[test_pred != test_y])


def binary_classifier_0D(dataset='3-1'):
    """预测二分类 0-D 的分类情况
        取局部特征预测 —— 取图片左半边，右半边基本一样
    :return: None
    """
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = DataUtil.load_train_test_data(option=dataset,
                                                                                         binary_classes='0D')
    # DataUtil.show_image(test_X[-1,0])

    # 取图片左半边，右半边基本一样
    train_X = train_X[:, :, :, :8]
    val_X = val_X[:, :, :, :8]
    test_X = test_X[:, :, :, :8]

    print(len(train_X),len(val_X),len(test_X))

    model = KnnBinaryClassifier()
    model.fit(
        (train_X, train_y),
        (val_X, val_y),
        (test_X, test_y),
        preprocessing_option=0,
    )


def binary_classifier_1I(dataset='2-1'):
    """预测二分类 1-I 的分类情况
        取局部特征预测 —— 取图片上半边，下半边基本一样
    :return: None
    """
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = DataUtil.load_train_test_data(option=dataset,
                                                                                         binary_classes='1I')
    # DataUtil.show_image(test_X[-1,0])
    # DataUtil.show_image(test_X[0,0])
    DataUtil.show_image(test_X[81, 0])
    # print(test_X[-1,0])

    # 取图片上半边，下半边基本一样
    train_X = train_X[:, :, :8, :10]
    val_X = val_X[:, :, :8, :10]
    test_X = test_X[:, :, :8, :10]

    # DataUtil.show_image(test_X[-1,0])
    # DataUtil.show_image(test_X[11,0])
    # print(test_X[-1,0])

    model = KnnBinaryClassifier()
    model.fit(
        (train_X, train_y),
        (val_X, val_y),
        (test_X, test_y),
        preprocessing_option=1,
    )


def binary_classifier_4A(dataset='2-1'):
    """预测二分类 1-I 的分类情况
        取局部特征预测 —— 取图片上半边，下半边基本一样
    :return: None
    """
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = DataUtil.load_train_test_data(option=dataset,
                                                                                         binary_classes='4A')
    # DataUtil.show_image(test_X[-1,0])
    # DataUtil.show_image(test_X[0,0])
    # DataUtil.show_image(test_X[3113,0])
    # print(test_X[-1,0])

    # 取图片上半边，下半边基本一样
    train_X = train_X[:, :, :, :]
    val_X = val_X[:, :, :, :]
    test_X = test_X[:, :, :, :]

    # DataUtil.show_image(test_X[-1,0])
    DataUtil.show_image(test_X[3113,0])
    # print(test_X[-1,0])

    model = KnnBinaryClassifier()
    model.fit(
        (train_X, train_y),
        (val_X, val_y),
        (test_X, test_y),
        preprocessing_option=1,
    )

def binary_classifier_8B(dataset='3-1'):
    """预测二分类 8-B 的分类情况
        取局部特征预测 —— 取图片左半边，右半边基本一样
    :return: None
    """
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = DataUtil.load_train_test_data(option=dataset,
                                                                                         binary_classes='8B')
    # DataUtil.show_image(test_X[-1,0])
    # DataUtil.show_image(test_X[0,0])
    # DataUtil.show_image(test_X[3113,0])
    # print(test_X[-1,0])

    # 取图片上半边，下半边基本一样
    train_X = train_X[:, :, :, :]
    val_X = val_X[:, :, :, :]
    test_X = test_X[:, :, :, :]

    # DataUtil.show_image(test_X[-1,0])
    DataUtil.show_image(test_X[3113,0])
    # print(test_X[-1,0])

    model = KnnBinaryClassifier()
    model.fit(
        (train_X, train_y),
        (val_X, val_y),
        (test_X, test_y),
        preprocessing_option=1,
    )


def binary_classifer(dataset='3-1', option='0D'):
    if option == '0D':
        binary_classifier_0D(dataset=dataset)
    elif option == '1I':
        binary_classifier_1I(dataset=dataset)
    elif option == '4A':
        binary_classifier_4A(dataset=dataset)
    elif option == '8B':
        binary_classifier_8B(dataset=dataset)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    binary_classifer(dataset='3-3', option='0D')
