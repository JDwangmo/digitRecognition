# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-15'
    Email:   '383287471@qq.com'
    Describe:
"""

from __future__ import print_function
from train_test_data.dataset_20160801.data_util import DataUtil
import sys
dutil = DataUtil()
log_output_file=open('M3_log.txt','w')
# log_output_file=sys.stdout

charset='8B'
version=2
image_feature_shape = (15,15)
badcase_file_path = '/home/jdwang/PycharmProjects/digitRecognition/cnn/result/badcase_%s.pickle'%charset
result_file_path = '/home/jdwang/PycharmProjects/digitRecognition/cnn/result/result_%s.pickle'%charset
print('version:%d,charset:%s'%(version,charset),file = log_output_file)
print('图片取大小：%s'%str(image_feature_shape),file = log_output_file )

train_data,test_data = dutil.load_train_test_data(
    version=version,
    charset=charset
)


from cnn_model.example.simple_m3_cnn import ImageCNN


ImageCNN.cross_validation(
    train_data=train_data,
    test_data=test_data,
    cv_data=None,
    output_shape=image_feature_shape,
    # 设置输出badcase
    output_badcase=False,
    output_result=False,
    badcase_file_path = badcase_file_path,
    result_file_path = result_file_path,
    log_output_file=log_output_file,
    num_labels = 2,
    # num_filter_list=[10,20,32,64,100,128],
    num_filter1_list=[10,20,32,64,100,128],
    num_filter2_list=[10,20,32,64,100,128],
    hidden1_list=[10,100,300,600,500,1000],
    # hidden1_list=[600],
    # filter1_list=[1,2,3],
    filter1_list=[2],
    # filter2_list=[4,5,6],
    filter2_list=[2],
    # filter3_list=[8,9],
    filter3_list=[2],
    verbose=0,
)