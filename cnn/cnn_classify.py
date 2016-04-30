#encoding=utf8
from dataProcessing.read_data import load_pix
import logging
import timeit
import pandas as pd
from keras.models import Sequential,model_from_json
from keras.layers import Dense,Convolution2D,Activation,MaxPooling2D,Flatten,Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )

num_train = 50
num_test = 1000


test_file_path = '/home/jdwang/PycharmProjects/digitRecognition/train_test_data/' \
                 '20160426/test_%d.csv'%(num_test)
image_shape = (15,15)

test_pix,test_y,test_im_name = load_pix(test_file_path,
                    shape=image_shape
                    )
label = test_y

test_pix = test_pix.reshape(test_pix.shape[0],
                             1,
                             test_pix.shape[1],
                             test_pix.shape[2])

character_name = sorted(list(set('0123456789ABCDEFGHIJKLMNPQRSTUWXYZ')))
# print character_name
# 将y转换成one-hot编码
test_y = [character_name.index(item) for item in test_y]
test_y = np_utils.to_categorical(test_y,34)
# print train_y.shape
# print test_y.shape

logging.debug( 'the shape of test sample:%d,%d,%d,%d'%(test_pix.shape))

start_time = timeit.default_timer()
win_shape = 2
nb_epoch = 15
model = model_from_json(open('/home/jdwang/PycharmProjects/digitRecognition/cnn/model/'
                             'cnn_model_architecture_%dwin_%depoch.json'%(win_shape,nb_epoch)).read())
model.load_weights('/home/jdwang/PycharmProjects/digitRecognition/cnn/model/'
                   'cnn_model_weights_%dwin_%depoch.h5'%(win_shape,nb_epoch))
end_time = timeit.default_timer()

print 'train time : %f'%(end_time-start_time)
classes = model.predict_classes([test_pix], batch_size=32)
print classes.shape
pred_result = [character_name[item] for item in classes]
# print pred_result
# print label
is_correct = pred_result==label
print sum(is_correct)
print sum(is_correct)/(len(label)*1.0)
test_result = pd.DataFrame({
    'label': label,
    'pred': pred_result,
    'is_correct': is_correct,
    'image_id': test_im_name
})
# 保存结果
test_result_path = '/home/jdwang/PycharmProjects/digitRecognition/cnn/result/20160426/' \
                   'cnn_result_%d_%d.csv' % (num_train, num_test)
test_result.to_csv(test_result_path, sep='\t')
# 保存模型

