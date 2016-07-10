#encoding=utf8
'''
该文件用于加载字符的矩阵数据
'''
import pandas as pd
import numpy as np
from PIL import Image
import logging
import timeit



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )

def load_pix(file_path,shape = (15,15),shuffle=True,normalize = True, char_set = 1):
    '''

    :param file_path:
    :param shape:
    :param shuffle:bool，是否打乱数据
    :return:
    '''
    logging.debug('开始加载文件:%s....'%(file_path))
    data = pd.read_csv(file_path,sep='\t',header=None)
    # print train_data
    # print train_data[1]
    if char_set == 1:
        character_name = sorted(list(set('0123456789')))
    elif char_set == 2:
        character_name = sorted(list(set('ABCDEFGHIJKLMNPQRSTUWXYZ')))
    else:
        character_name = sorted(list(set('0123456789ABCDEFGHIJKLMNPQRSTUWXYZ')))

    label = data[0].apply(lambda x:x[0]).as_matrix()
    y = np.asarray([character_name.index(item) for item in label])
    im_name = data[0].apply(lambda x:x.split('/')[1]).as_matrix()
    # print y
    if shape[0] == 1:
        train_pix = np.array([np.array(item.split(','),np.uint8)
                              for item in data[1]])
    else:
        train_pix = np.array([np.array(item.split(','),np.uint8).reshape(shape)
                              for item in data[1]])
    if normalize:
        train_pix = train_pix/255.0
    # print train_pix[0]
    # 打乱数据
    if shuffle:
        logging.debug('随机打乱数据...')
        rand = np.random.RandomState(0)
        rand_list = rand.permutation(len(y))
        # print y[rand_list]
        y = y[rand_list]
        label = label[rand_list]
        im_name = im_name[rand_list]
        train_pix = train_pix[rand_list]
    # pic = Image.fromarray(train_pix[0])
    # pic.save('test.bmp','bmp')
    logging.debug('完成加载文件并转换成矩阵，总共有%d个图片！'%(len(train_pix)))
    y = y-1
    return train_pix,y,label,im_name

if __name__=='__main__':

    train_file_path = '/home/jdwang/PycharmProjects/digitRecognition/train_test_data/' \
                      '20160426/train_5.csv'
    test_file_path = '/home/jdwang/PycharmProjects/digitRecognition/train_test_data/' \
                     '/20160426/test_10.csv'

    start = timeit.default_timer()

    train_pix,train_y,train_im_name = load_pix(train_file_path,
                         shape=(1, 15 * 15)
                         )

    test_pix,test_y,test_im_name = load_pix(test_file_path,
                        shape=(1, 15 * 15)
                        )
    logging.debug('the shape of train sample:%d,%d' % (train_pix.shape))
    logging.debug('the shape of test sample:%d,%d' % (test_pix.shape))
    end = timeit.default_timer()
    logging.debug('总共运行时间:%ds' % (end-start))
