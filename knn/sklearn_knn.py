#encoding=utf8

from sklearn.neighbors import KNeighborsClassifier
from dataProcessing.read_data import load_pix
import logging
import numpy as np
import pandas as pd
import timeit

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )
num_train = 100
num_test = 1000
# 设置训练数据和测试数据的路径
char_set = 1
train_file_path = '/home/jdwang/PycharmProjects/digitRecognition/train_test_data/' \
                  '20160426/train_%dcharset_%d.csv'%(char_set,num_train)
test_file_path = '/home/jdwang/PycharmProjects/digitRecognition/train_test_data/' \
                 '20160426/test_%dcharset_%d.csv'%(char_set,num_test)

train_pix, train_y, train_label, train_im_name = load_pix(train_file_path,
                                                          shape=(1,15*15),
                                                          char_set=char_set
                                                          )

test_pix, test_y, test_label, test_im_name = load_pix(test_file_path,
                                                      shape=(1, 15 * 15),
                                                      char_set=char_set
                                                      )

logging.debug( 'the shape of train sample:%d,%d'%(train_pix.shape))
logging.debug( 'the shape of test sample:%d,%d'%(test_pix.shape))

start = timeit.default_timer()

model = KNeighborsClassifier(n_neighbors=3,
                             weights='distance',
                             algorithm='kd_tree',
                             leaf_size=30,
                             p=2,
                             metric='minkowski',
                             metric_params=None,
                             n_jobs=4
                             )

model.fit(train_pix,train_y)

pred_result = model.predict(test_pix)

is_correct = (pred_result==test_y)
print '正确的个数：%d'%(sum(is_correct))
print '正确率：%f'%(sum(is_correct)/(len(test_y)*1.0))

test_result = pd.DataFrame({
            'label':test_y,
            'pred':pred_result,
            'is_correct':is_correct,
            'image_id':test_im_name
            })

test_result_path = '/home/jdwang/PycharmProjects/digitRecognition/knn/output/20160426/' \
                   'sklearn_knn_result_%d_%d.csv'%(num_train,num_test)
test_result.to_csv(test_result_path,sep='\t')

end = timeit.default_timer()
logging.debug('总共运行时间:%ds' % (end-start))
