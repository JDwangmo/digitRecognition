#encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-07-11'
    Email:   '383287471@qq.com'
    Describe: 
"""
from __future__ import print_function

import Image
import io
import numpy as np
import pandas as pd
import logging
import timeit
import os

class DataUtil(object):

    def img_to_vector(self,
                      data_dir='/home/jdwang/PycharmProjects/digitRecognition/image_data/20160426/'):
        '''
            将15*15的图片转为 255*1 的向量，并保存成csv格式

        :param data_dir: 图片的文件夹
        :type data_dir: str
        :return:
        '''
        output_file_path = './output/image_data.csv'

        character_name = sorted(list(set('0123456789ABCDEFGHIJKLMNPQRSTUWXYZ')))
        sub_dir_list = sorted(os.listdir(data_dir))
        logging.debug('子文件个数为：%d，有：%s' % (len(sub_dir_list), str(sub_dir_list)))
        print('子文件个数为：%d，有：%s' % (len(sub_dir_list), str(sub_dir_list)))
        rand = np.random.RandomState(0)
        with io.open(output_file_path,'w',encoding='utf8') as fout:
            fout.write(u'LABEL\tFILE_NAME\tPIC\n')
            for sub_dir in sub_dir_list:
                if sub_dir in character_name:
                    logging.debug('正在处理字符：%s' % (sub_dir))
                    file_list = os.listdir(data_dir + sub_dir)
                    # 过滤非.bmp结尾的
                    file_list = [item for item in file_list if item.endswith('.bmp')]
                    logging.debug('字符(%s)的sample个数为：%d' % (sub_dir, len(file_list)))
                    print('字符(%s)的sample个数为：%d' % (sub_dir, len(file_list)))
                    # print '字符(%s)的sample个数为：%d'%(sub_dir,len(file_list))
                    # print rand_train_list
                    # print rand_test_list
                    for train_file in file_list:
                        with open(data_dir + sub_dir + '/' + train_file) as fin:
                            pic = Image.open(fin)
                            # print pic.size,np.prod(pic.size)
                            pix = np.asarray(pic)
                            # print pix
                            # reshape the matrix'size to (1,225)
                            pix = pix.flatten()
                            # print pix
                            pic_grey = ','.join([str(item) for item in pix])
                            # print(sub_dir+'\t'+sub_dir + '/' + train_file + '\t' + pic_grey)
                            # quit()
                            fout.write(u'%s\t%s\t%s\n'%(sub_dir,sub_dir + '/' + train_file, pic_grey))
                            # print pic_grey
                            # print pix.shape
                        # print pic_grey
                        # print pix.shape



    def load_pic(self,path):
        data_pic = pd.read_csv(path,
                               sep='\t',
                               encoding='utf8',
                               header=0,
                               )
        return data_pic

if __name__ == '__main__':
    dutil = DataUtil()
    # dutil.img_to_vector()
    data_pic = dutil.load_pic('/home/jdwang/PycharmProjects/digitRecognition/dataProcessing/output/image_data.csv')
    print(data_pic.shape)
    # print(data_pic.head(2))
    print('|'.join(data_pic['LABEL'].value_counts().sort_index().index))
    print('|'.join(str(item) for item in data_pic['LABEL'].value_counts().sort_index().values))