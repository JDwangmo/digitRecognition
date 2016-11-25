#encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-07-11'
    Email:   '383287471@qq.com'
    Describe: 
"""
from __future__ import print_function
from keras.utils import np_utils

import Image
import io
import numpy as np
import pandas as pd
import logging
import timeit
import os



# index_to_character = sorted(list(set('7T')))



class DataUtil(object):



    def __init__(self,
                 charset_type = 0,
                 ):
        '''
            初始化参数

        :param charset_type: 选择何种字符集合，可以输入字符串，也可以输入整形。分别有：
                            0: 0123456789ABCDEFGHIJKLMNPQRSTUWXYZ
                            1: 0123456789
                            2: ABCDEFGHIJKLMNPQRSTUWXYZ


        :type charset_type: int/str
        '''
        super(DataUtil, self).__init__()

        if type(charset_type) == int:
            charset = self.transform_chartype(charset_type)
        elif type(charset_type) == str:
            charset = charset_type
        else:
            assert False,'charset_type 字符集类型错误！'
        # print(charset)
        self.charset_type = charset_type
        self.index_to_character = charset
        self.character_to_index = {j:i for i,j in enumerate(self.index_to_character)}



    def transform_chartype(self,charset_type):
        '''
            将整形数字转成charset集合
        :param chartype:
        :type chartype: int
        :return:
        '''

        if charset_type == 0:
            charset = sorted(list(set('0123456789ABCDEFGHIJKLMNPQRSTUWXYZ')))
        elif charset_type == 1:
            charset = sorted(list(set('0123456789')))
        elif charset_type == 2:
            charset = sorted(list(set('ABCDEFGHIJKLMNPQRSTUWXYZ')))
        else:
            charset = sorted(list(set('0123456789ABCDEFGHIJKLMNPQRSTUWXYZ')))

        return charset

    def img_to_vector(self,
                      data_dir='/home/jdwang/PycharmProjects/digitRecognition/image_data/20160426_modify/'):
        '''
            将15*15的图片转为 255*1 的向量，并保存成csv格式
            输出到文件：./output/image_data.csv

        :param data_dir: 图片的文件夹，默认路径为：/image_data/20160426_modify/
        :type data_dir: str
        :return:
        '''
        output_file_path = './output/image_data.csv'
        print(output_file_path)
        sub_dir_list = sorted(os.listdir(data_dir))
        logging.debug('子文件个数为：%d，有：%s' % (len(sub_dir_list), str(sub_dir_list)))
        print('子文件个数为：%d，有：%s' % (len(sub_dir_list), str(sub_dir_list)))
        rand = np.random.RandomState(0)
        with io.open(output_file_path,'w',encoding='utf8') as fout:
            fout.write(u'LABEL\tFILE_NAME\tPIC\n')
            for sub_dir in sub_dir_list:
                if sub_dir in self.transform_chartype(0):
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

    def save_pic(self,data,path):
        data.to_csv(path,
                    sep='\t',
                    encoding='utf8',
                    index=False,
                    )

    def split_train_test(self,data):
        num_batch = 3
        batch_size = 100
        train_batchs = [pd.DataFrame(columns=[u'LABEL',u'FILE_NAME',u'PIC'])]*num_batch
        test_batchs = [pd.DataFrame(columns=[u'LABEL',u'FILE_NAME',u'PIC'])]*num_batch
        rand = np.random.RandomState(1337)
        for label,group in data.groupby(by=['LABEL']):
            # print(label,len(group))
            if label not in self.index_to_character:
                print(label)
                continue
            num_data = len(group)
            rand_index = rand.permutation(num_data)
            if num_data>=num_batch*batch_size:
                # print(rand_index[:5])

                for batch_index in range(num_batch):
                    train_index = rand_index[batch_index*batch_size:(batch_index+1)*batch_size]
                    test_index = list(rand_index[:batch_index*batch_size])+list(rand_index[(batch_index+1)*batch_size:])
                    train = (group.iloc[train_index])
                    test = (group.iloc[test_index])

                    train_batchs[batch_index] = pd.concat((train_batchs[batch_index],train),axis=0)
                    test_batchs[batch_index] = pd.concat((test_batchs[batch_index],test),axis=0)
                    # print(len(train_batchs[batch_index]))


            else:

                a_part_size = round(num_data/(num_batch*1.0))


                for batch_index in range(num_batch):

                    train_index = rand_index[batch_index*a_part_size:(batch_index+1)*a_part_size]
                    test_index = list(rand_index[:batch_index*a_part_size])+list(rand_index[(batch_index+1)*a_part_size:])


                    need_padding_mul = int(batch_size/len(train_index))
                    need_padding_add = int(batch_size%len(train_index))
                    # print(num_data,len(train_index),need_padding_mul,need_padding_add)

                    train_index = (list(train_index)*need_padding_mul) + list(train_index[:need_padding_add])

                    # print(len(train_batchs[batch_index]))
                    # print(len(train_index))

                    train = group.iloc[train_index]
                    test = group.iloc[test_index]

                    train_batchs[batch_index] = pd.concat((train_batchs[batch_index],train),axis=0)
                    test_batchs[batch_index] = pd.concat((test_batchs[batch_index],test),axis=0)

                # quit()



                # quit()


        # print(len(train_batchs[0]),len(test_batchs[0]))
        # print(len(train_batchs[1]),len(test_batchs[1]))
        # print(len(train_batchs[2]),len(test_batchs[2]))
        self.save_pic(train_batchs[0],'output/train0_%stype.csv'%(self.charset_type))
        self.save_pic(train_batchs[1],'output/train1_%stype.csv'%(self.charset_type))
        self.save_pic(train_batchs[2],'output/train2_%stype.csv'%(self.charset_type))
        self.save_pic(test_batchs[0],'output/test0_%stype.csv'%(self.charset_type))
        self.save_pic(test_batchs[1],'output/test1_%stype.csv'%(self.charset_type))
        self.save_pic(test_batchs[2],'output/test2_%stype.csv'%(self.charset_type))
        return train_batchs,test_batchs

    def get_train_test(self):
        num_batch = 3
        character_to_index = {j:i for i,j in enumerate(self.index_to_character)}
        for batch_index in range(num_batch):
            train_data = self.load_pic('/home/jdwang/PycharmProjects/digitRecognition/dataProcessing/output/train%d_%stype.csv'%(batch_index,self.charset_type))
            train_X = train_data[u'PIC'].as_matrix()
            to_pic = lambda x : [int(item) for item in x.split(',')]
            train_X = np.asarray(map(to_pic,train_X))

            train_X = train_X.reshape(train_X.shape[0],
                                      1,
                                      15,
                                      15)
            # print(X_train.shape)
            # quit()

            train_y = np.asarray(train_data[u'LABEL'].astype(dtype='str').map(character_to_index).as_matrix())

            test_data = self.load_pic('/home/jdwang/PycharmProjects/digitRecognition/dataProcessing/output/test%d_%stype.csv'%(batch_index,self.charset_type))
            test_X = test_data[u'PIC'].as_matrix()

            test_X = np.asarray(map(to_pic,test_X))
            test_X = test_X.reshape(test_X.shape[0],
                                      1,
                                      15,
                                      15)

            print(len(train_y))
            print(len(test_X))
            test_y = np.asarray(test_data[u'LABEL'].astype(dtype='str').map(character_to_index).as_matrix())
            # test_y = np_utils.to_categorical(test_y, num_class)

            yield (train_X,train_y),(test_X,test_y)

if __name__ == '__main__':
    dutil = DataUtil(charset_type='0DQ')
    # dutil.img_to_vector()
    # quit()
    data_pic = dutil.load_pic('/home/jdwang/PycharmProjects/digitRecognition/train_test_data/20160426_modify/image_data.csv')
    # print(data_pic.shape)
    # print(data_pic.head(2))
    # print('|'.join(data_pic['LABEL'].value_counts().sort_index().index))
    # print('|'.join(str(item) for item in data_pic['LABEL'].value_counts().sort_index().values))
    # quit()
    dutil.split_train_test(data_pic)
    # quit()
    for item in dutil.get_train_test():
        # print(len(item))
        pass
