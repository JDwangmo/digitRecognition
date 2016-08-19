# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-18'
    Email:   '383287471@qq.com'
    Describe: 将 badcase文件中的badcase 向量转为图片，并保存
"""

from train_test_data.dataset_20160801.data_util import DataUtil

dutil = DataUtil()

import pickle

charset = '8B'
badcase_file_path = '/home/jdwang/PycharmProjects/digitRecognition/cnn/result/badcase_%s.pickle'%charset

with open(badcase_file_path) as fin:
    test_X = pickle.load(fin)
    test_y = pickle.load(fin)
    predict_result = pickle.load(fin)
    dutil.outPutImage(
        root_dir='/home/jdwang/PycharmProjects/digitRecognition/cnn/result/badcase/',
        badcase=(test_X,test_y,predict_result),
        charset = charset
    )