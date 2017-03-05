# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-03-05'; 'last updated date: 2017-03-05'
    Email:   '383287471@qq.com'
    Describe: 数据处理工具类
                1. 读取数据 load_valdata() ;
                2. 保存数据成C程序能调用的二进制格式: save_img_to_bininary_file()
"""
from __future__ import print_function
import pickle
import os
import numpy as np
import struct

__version__ = '1.0'

# 数据集根目录
data_root_path = '/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/'


class DataUtil(object):
    def load_valdata(self, version='1122'):
        """读取不同版本的测试集
        1120 - 对应 /home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/modelAndData1120/Data/
        1122 - 对应 /home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/modelAndData1122/Data/

        :param dir_path: str
        :param version: str
        :return:
        """
        other_X_new, other_y_new = None, None
        other_X, other_y = None, None
        train_X, train_y = None, None
        val_X, val_y = None, None

        # 读取验证数据、测试数据
        if version == '1122':
            val_file_path = os.path.join(data_root_path, 'modelAndData1122/Data/',
                                         'TrainSet_trainAndVal_testSet.pickle')
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
            val_file_path = os.path.join(data_root_path, 'modelAndData1214/Data/',
                                         'TrainSet_trainAndVal_testSet.pickle')
            other_file_path = os.path.join(data_root_path, 'modelAndData1214/Data/', 'OldAndAlldata_TestSet.pickle')
            new_other_file_path = os.path.join(data_root_path, 'modelAndData1214/Data/', '20161202am.pickle')
            with open(val_file_path, 'rb') as train_file:
                val_X = pickle.load(train_file)
                val_y = pickle.load(train_file)
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
        elif version == '20170218':

            train_file_path = os.path.join(data_root_path, 'modelAndData20170218/Data/',
                                           '20170218_train_test_ignore.pickle')
            with open(train_file_path, 'rb') as train_file:
                train_X = pickle.load(train_file)
                train_y = pickle.load(train_file)
                test_X = pickle.load(train_file)
                test_y = pickle.load(train_file)
            train_y = np.asarray(train_y)
            test_y = np.asarray(test_y)


        else:
            raise NotImplementedError

        return (train_X, train_y), (val_X, val_y), (test_X, test_y), (other_X, other_y), (other_X_new, other_y_new)

    def save_img_to_bininary_file(self, test_X, test_y, name='val'):
        # 将图片保存成 二进制 形式
        with open(os.path.join(data_root_path, 'modelAndData20170218/Data/images_%sdata.mat' % name), 'wb') as fout:
            print(test_X.shape)
            fout.write(struct.pack('i', len(test_X)))

            for in_data in test_X:
                # print(weight[0][0])
                # quit()
                for item in in_data.flatten():
                    # print(item,ord(chr(item)))
                    fout.write(struct.pack('c', chr(item)))
                    # quit()

        with open(os.path.join(data_root_path, 'modelAndData20170218/Data/labels_%sdata.mat' % name),
                  'wb') as fout:
            fout.write(struct.pack('i', len(test_y)))
            for item in test_y:
                fout.write(struct.pack('c', chr(item)))


if __name__ == '__main__':
    dutil = DataUtil()
    (train_X, train_y), (val_X, val_y), (test_X, test_y), (other_X, other_y), (
    other_X_new, other_y_new) = dutil.load_valdata(version='20170218')

    X = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    print(X.shape)
    print(y.shape)

    print(train_X.shape)
    print(train_y.shape)
    print(test_X.shape)
    print(test_y.shape)

    dutil.save_img_to_bininary_file(X, y, name='train_test20170218')
