#encoding=utf8
from dataProcessing.read_data import load_pix
import logging
import timeit
import pandas as pd
from keras.models import Sequential,model_from_json
from keras.optimizers import SGD
from keras.utils import np_utils

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )
# 设置参数
# 设置迭代的次数
NB_EPOCH = 50
NUM_TRAIN = 100
NUM_TEST = 1000

logging.info('选择迭代次数：%d;训练sample%d(单个字符)的模型。'%(NB_EPOCH,NUM_TRAIN))
logging.info('进行测试，测试sample个数(单个字符)：%d...'%(NUM_TEST))


num_train = NUM_TRAIN
num_test = NUM_TEST
test_file_path = '/home/jdwang/PycharmProjects/digitRecognition/train_test_data/' \
                 '20160426/test_%d.csv'%(num_test)
image_shape = (15,15)

test_pix,test_y,test_im_name = load_pix(test_file_path,
                    shape=image_shape,
                    shuffle=False
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
nb_epoch = NB_EPOCH
cnn_model_architecture = '/home/jdwang/PycharmProjects/digitRecognition/cnn_model/model/' \
                         'cnn_model_architecture_%dtrain_%dwin_%depoch.json' \
                         % (num_train,win_shape, nb_epoch)
logging.info('加载模型架构（%s）...'%cnn_model_architecture)

model = model_from_json(open(cnn_model_architecture,'r').read())
cnn_model_weights = '/home/jdwang/PycharmProjects/digitRecognition/cnn_model/model/' \
                    'cnn_model_weights_%dtrain_%dwin_%depoch.h5' \
                    % (num_train,win_shape, nb_epoch)

logging.info('加载模型权重（%s）...'%cnn_model_weights)
model.load_weights(cnn_model_weights)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile( optimizer=sgd,loss='mse')


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
test_result_path = '/home/jdwang/PycharmProjects/digitRecognition/cnn_model/output/20160426/' \
                   'cnn_result_%depoch_%d_%d.csv' % (nb_epoch,num_train, num_test)
test_result.to_csv(test_result_path, sep='\t')
# 保存模型

