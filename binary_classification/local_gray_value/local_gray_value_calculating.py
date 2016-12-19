# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-12-19'; 'last updated date: 2016-12-19'
    Email:   '383287471@qq.com'
    Describe: 字符的二分类方法，对于 A 是 B的局部的字符，通过比对 差异处的 灰度值大小来判定 字符类别归属
        - P-R : 8:14,8:14
        - E-F : 10:14,5:14
"""
from __future__ import print_function
import pickle
import os
import numpy as np

character_name = list('0123456789ABCDEFGHIJKLMNPQRSTUWXYZ')
from data_processing_util.data_util import DataUtil

dutil = DataUtil()


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


def calculating_local_region_gray_values(images, row_index, cols_index):
    images_local_region = images[:, 0, row_index[0]:row_index[1], cols_index[0]:cols_index[1]]
    gray_values = []
    for index, item in enumerate(images_local_region):
        # 反转像素值，白色变黑色，黑色变白色
        local_region = 255 - item
        local_region_min_value = np.min(local_region)
        local_region_sum = np.sum(local_region - local_region_min_value)
        # if local_region_sum == 638:
        #     print(index)
        #     print(255-images[index,0])
        #     dutil.show_image_from_array(images[index,0])
        #     print(item)
        #     print(local_region)
        #     print(local_region - local_region_min_value)
        #     quit()

        gray_values.append(local_region_sum)

    print('局部灰度值范围为：%d～%d' % (min(gray_values), max(gray_values)))
    return min(gray_values), max(gray_values)


(train_X, train_y), (test_X, test_y), (other_X, other_y), (other_X_new, other_y_new) = load_valdata(version='1214')

# region E - F: 10:14, 5:14
print('------------ E-F ------------- ')
# E
idx_is_E = train_y == char_to_index('E')
print('训练集合是E的个数:%d' % sum(idx_is_E))
E_min, E_max = calculating_local_region_gray_values(train_X[idx_is_E], [10, 14], [5, 14])
# quit()
# F
idx_is_F = train_y == char_to_index('F')
print('训练集合是F的个数:%d' % sum(idx_is_F))
F_min, F_max = calculating_local_region_gray_values(train_X[idx_is_F], [10, 14], [5, 14])

print('E-F取得的边界点是：%d' % ((E_min + F_max) / 2))
# endregion

# region P - R: 8:14, 8:14
print('------------ R-P ------------- ')
# R
idx_is_R = train_y == char_to_index('R')
print('训练集合是R的个数:%d' % sum(idx_is_R))
R_min, R_max = calculating_local_region_gray_values(train_X[idx_is_R], [8, 14], [8, 14])
# P
idx_is_P = train_y == char_to_index('P')
print('训练集合是P的个数:%d' % sum(idx_is_P))
P_min, P_max = calculating_local_region_gray_values(train_X[idx_is_P], [8, 14], [8, 14])
# quit()
print('P-R取得的边界点是：%d' % ((R_min + P_max) / 2))
# endregion

# region I - T: 1:5, 1:14
print('------------ T-I ------------- ')
# T
idx_is_T = train_y == char_to_index('T')
print('训练集合是T的个数:%d' % sum(idx_is_T))
T_min, T_max = calculating_local_region_gray_values(train_X[idx_is_T], [1, 5], [1, 14])

# quit()
# I
idx_is_I = train_y == char_to_index('I')
print('训练集合是I的个数:%d' % sum(idx_is_I))
I_min, I_max = calculating_local_region_gray_values(train_X[idx_is_I], [1, 5], [1, 14])
print('T-I取得的边界点是：%d' % ((T_min + I_max) / 2))
# endregion
