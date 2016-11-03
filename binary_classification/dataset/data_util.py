# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-11-01'; 'last updated date: 2016-11-01'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
import pickle
import os
import numpy as np

class DataUtil:
    def __init__(self):
        pass

    @staticmethod
    def load_train_test_data(option='1-1', binary_classes = '0D'):
        """ 加载训练数据、验证数据和测试数据

        :param option: str
            数据集版本
        :param binary_classes: int
            待分类字符的数据集
        :return:
            (train_X, train_y), (val_X, val_y), (test_X, test_y)
        """
        if option == '1-1':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161101'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_data1.pickle')
        elif option == '1-2':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161101'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_data2.pickle')
        elif option == '1-3':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161101'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_data3.pickle')
        elif option == '1-4':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161101'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_data4.pickle')
        elif option == '1-5':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161101'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_data5.pickle')
        elif option == '2-1':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161102'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_avgValandTest_data1.pickle')
        elif option == '2-2':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161102'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_avgValandTest_data2.pickle')
        elif option == '2-3':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161102'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_avgValandTest_data3.pickle')
        elif option == '2-4':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161102'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_avgValandTest_data4.pickle')
        elif option == '2-5':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161102'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_avgValandTest_data5.pickle')
        else:
            raise NotImplementedError

        with open(path, 'rb') as train_file:
            train_X = pickle.load(train_file)
            train_y = np.asarray(pickle.load(train_file))
            val_X = pickle.load(train_file)
            val_y = np.asarray(pickle.load(train_file))
            test_X = pickle.load(train_file)
            test_y = np.asarray(pickle.load(train_file))

        character_name = list('0123456789ABCDEFGHIJKLMNPQRSTUWXYZ')

        selected_samples = np.array(map(lambda x:character_name[x] in list(binary_classes),train_y))
        train_X = train_X[selected_samples]
        train_y = train_y[selected_samples]

        selected_samples = np.array(map(lambda x:character_name[x] in list(binary_classes),val_y))
        val_X = val_X[selected_samples]
        val_y = val_y[selected_samples]

        selected_samples = np.array(map(lambda x:character_name[x] in list(binary_classes),test_y))
        test_X = test_X[selected_samples]
        test_y = test_y[selected_samples]

        # train_y = np_utils.to_categorical(train_y, nb_classes)
        # val_y = np_utils.to_categorical(val_y, nb_classes)
        # test_y = np_utils.to_categorical(test_y, nb_classes)

        return (train_X, train_y), (val_X, val_y), (test_X, test_y)
    @staticmethod
    def show_image(image_array):
        """ 输入二维数组，显示图片

        :param image_array: np.array
            2D 数组
        :return: None
        """
        import Image
        image = Image.fromarray(image_array)
        image.show()

if __name__ == '__main__':
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = DataUtil.load_train_test_data(option=1)
