#encoding=utf8
'''
该处理程序主要将图片数据转换成灰度值的矩阵数据，并分别储存到训练文件和训练数据;
可以根据需要设置需要处理的字符、训练、测试sample的个数;
冠字号有字母和数字组成，总共34种字符，除去字母O和V;
'''

from PIL import Image
import numpy as np
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )
# 设置随机抽取的训练集中，每个字符数量的大小
NUM_TRAIN = 10000
# 设置随机抽取的测试集中，每个字符数量的大小
NUM_TEST = 100

char_set = 3
# 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
if char_set == 1:
    character_name = sorted(list(set('0123456789')))
elif char_set == 2:
    character_name = sorted(list(set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')))
else:
    character_name = sorted(list(set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')))

logging.debug('需要处理的字符长度为：%d，有：%s'%(len(character_name),str(character_name)))

data_dir = '/home/jdwang/PycharmProjects/digitRecognition/image_data/20160426/'
sub_dir_list = sorted(os.listdir(data_dir))
logging.debug('子文件个数为：%d，有：%s'%(len(sub_dir_list),str(sub_dir_list)))
rand = np.random.RandomState(0)

logging.debug('生成训练数据和测试数据...')
num_train = NUM_TRAIN
num_test = NUM_TEST
logging.debug('每个字符分别随机选取训练和测试sample个数分别为：%d,%d'%(num_train,num_test))

train_data_file = '/home/jdwang/PycharmProjects/digitRecognition/train_test_data/20160426/' \
                  'train_%dcharset_%d.csv'%(char_set,num_train)
test_data_file = '/home/jdwang/PycharmProjects/digitRecognition/train_test_data/20160426/' \
                  'test_%dcharset_%d.csv'%(char_set,num_test)
logging.debug('训练文件为：%s'%(train_data_file))
logging.debug('测试文件为：%s'%(test_data_file))
train_data_out = open(train_data_file,'w')
test_data_out = open(test_data_file,'w')

for sub_dir in sub_dir_list:
    if sub_dir in character_name:
        logging.debug('正在处理字符：%s'%(sub_dir))
        file_list = os.listdir(data_dir+sub_dir)
        # 过滤非.bmp结尾的
        file_list = [item for item in file_list if item.endswith('.bmp')]
        logging.debug('字符(%s)的sample个数为：%d'%(sub_dir,len(file_list)))
        # print '字符(%s)的sample个数为：%d'%(sub_dir,len(file_list))
        rand_train_list = rand.permutation(file_list)[:num_train]
        if num_train + num_test > len(file_list):
            rand_test_list = rand.permutation(file_list)[num_train:-1]
        else:
            rand_test_list = rand.permutation(file_list)[num_train:num_train+num_test]
        # print rand_train_list
        # print rand_test_list
        for train_file in rand_train_list:
            with open(data_dir+sub_dir+'/'+train_file) as fin:
                pic = Image.open(fin)
                # print pic.size,np.prod(pic.size)
                pix = np.asarray(pic)
                # print pix
                # reshape the matrix'size to (1,225)
                pix = pix.flatten()
                # print pix
                pic_grey = ','.join([str(item) for item in pix])
                train_data_out.write(sub_dir+'/'+train_file+'\t'+pic_grey+'\n')
                # print pic_grey
                # print pix.shape

        for test_file in rand_test_list:
            with open(data_dir+sub_dir+'/'+test_file) as fin:
                pic = Image.open(fin)
                # print pic.size,np.prod(pic.size)
                pix = np.asarray(pic)
                # print pix
                # reshape the matrix'size to (1,225)
                pix = pix.flatten()
                pic_grey = ','.join([str(item) for item in pix])
                test_data_out.write(sub_dir+'/'+test_file+'\t'+pic_grey+'\n')
                # print pic_grey
                # print pix.shape

train_data_out.close()
test_data_out.close()
