# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-15'
    Email:   '383287471@qq.com'
    Describe:
"""

from train_test_data.dataset_20160801.data_util import DataUtil

dutil = DataUtil()


charset='8B'
version=2
image_feature_shape = (15,8)
badcase_file_path = '/home/jdwang/PycharmProjects/digitRecognition/cnn/result/badcase_%s.pickle'%charset

print('version:%d,charset:%s'%(version,charset))
print('图片取大小：%s'%str(image_feature_shape))

train_data,test_data = dutil.load_train_test_data(
    version=version,
    charset=charset
)


from cnn_model.example.one_conv_simple_cnn import ImageCNNWithOneConv


ImageCNNWithOneConv.cross_validation(
    train_data=train_data,
    test_data=test_data,
    cv_data=None,
    output_shape=image_feature_shape,
    # 设置输出badcase
    output_badcase=False,
    badcase_file_path = badcase_file_path,
    num_labels = 2,
    # num_filter_list=[10,20,32,64,100,128],
    num_filter_list=[32],
    # hidden1_list=[10,100,300,500,600,1000],
    hidden1_list=[600],
    # filter1_list=[1,2,3],
    filter1_list=[3],
    # filter2_list=[4,5,6],
    filter2_list=[5],
    filter3_list=[8,9],
    # filter3_list=[7],
    verbose=0,
)