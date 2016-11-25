# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-15'
    Email:   '383287471@qq.com'
    Describe:
"""

import pickle
import Image
import numpy as np
import os

class DataUtil(object):
    def __init__(self):
        character_name = list('0123456789ABCDEFGHIJKLMNPQRSTUWXYZ')
        self.char_to_index = lambda x:character_name.index(x)
        self.index_to_char = lambda x:character_name[x]

    def get_data(self,data ,char_index):
        '''
            返回某个 char 的数据

        :param char_index:
        :return:
        '''
        train_X,train_y = data

        selected_index = train_y == char_index
        return train_X[selected_index],train_y[selected_index]

    def img_to_vector(
            self,
            data_dir=None,
            save_vector=True
    ):
        '''
            将图片转为 array-like 向量形式

        :param save_vector: 是否将 向量 保存到本地
        :return:
        '''

        output_file_path = 'data_vector.pickle'

        # print(output_file_path)
        sub_dir_list = sorted(os.listdir(data_dir))

        print('子文件个数为：%d，有：%s' % (len(sub_dir_list), str(sub_dir_list)))
        image_id = []
        X = []
        y = []
        for sub_dir in sub_dir_list:
            if sub_dir in list('0123456789ABCDEFGHIJKLMNPQRSTUWXYZ'):
                file_list = os.listdir(data_dir + sub_dir)
                # 过滤非.bmp结尾的
                file_list = [item for item in file_list if item.endswith('.bmp')]
                print('字符(%s)的sample个数为：%d' % (sub_dir, len(file_list)))
                # print '字符(%s)的sample个数为：%d'%(sub_dir,len(file_list))
                # print rand_train_list
                # print rand_test_list
                for train_file in file_list:
                    with open(data_dir + sub_dir + '/' + train_file) as fin:
                        pic = Image.open(fin)
                        pix = np.asarray(pic)
                        # print pix
                        # reshape the matrix'size to (1,225)
                        X.append(pix)
                        y.append(self.char_to_index(sub_dir))
                        image_id.append(sub_dir + '/' + train_file)
                        # pic_grey = ','.join([str(item) for item in pix])
                        # print(sub_dir+'\t'+sub_dir + '/' + train_file + '\t' + pic_grey)
                        # quit()
                        # fout.write( u'%s\t%s\t%s\n' %(sub_dir ,sub_dir + '/' + train_file, pic_grey))

        print('测试个数：%d'(len(X)))
        X = np.asarray(X).reshape(-1, 1, 15, 15)
        y = np.asarray(y, dtype=int)
        # print(X.shape)
        # print(y)
        if save_vector:
            with open(output_file_path, 'wb') as fout:
                pickle.dump(X, fout)
                pickle.dump(y, fout)
        return X, y

    def load_train_test_data(self,version=1,charset='all'):
        '''
            加载训练集和测试集合，分为了两组，通过version来选择不同的组，

        :param version: 选择不同的组
        :type version: int
        :param charset: 字符集,比如输入'8B',则返回 8 和 B 的所有数据集
        :type charset: str
        :return: (X_train,y_train),(test_X,test_y)
        '''

        root_path = '/home/jdwang/PycharmProjects/digitRecognition/train_test_data/dataset_20160801/'
        if version == 1:
            data_file_path = root_path + 'TrainSet&TestSet_3.pickle'
        elif version==2:
            data_file_path = root_path + 'TrainSet&TestSet_4.pickle'
        else:
            raise NotImplementedError


        with open(data_file_path,'rb') as fin:
            train_X = np.asarray(pickle.load(fin))
            train_y = np.asarray(pickle.load(fin))
            test_X = np.asarray(pickle.load(fin))
            test_y = np.asarray(pickle.load(fin))

        if charset=='all':
            pass
        else:
            tr_X = []
            tr_y = []
            te_X = []
            te_y = []
            new_index = 0
            for char in charset:
                char_index = self.char_to_index(char)
                x,y = self.get_data((train_X,train_y),char_index)
                tr_X.append(x)
                tr_y.append([new_index]*len(y))

                x,y = self.get_data((test_X,test_y),char_index)
                te_X.append(x)
                te_y.append([new_index]*len(y))
                new_index +=1


            train_X = np.concatenate(tr_X,axis=0)
            train_y = np.concatenate(tr_y,axis=0)
            test_X = np.concatenate(te_X,axis=0)
            test_y = np.concatenate(te_y,axis=0)

        return (train_X,train_y),(test_X,test_y)

    def show_picture(self,array):
        '''
            显示图片，输入一个数组，显示成一个图片

        :param array:
        :return:
        '''
        im = Image.fromarray(array)

        im.show()

    def outPutImage(self,
                    root_dir='./badcase/',
                    badcase=None,
                    charset=None,
                    ):
        '''
            # 将badcase的图片输出到Badcase的文件夹中

        :param root_dir:
        :param badcase: (test_X,test_y,predict_result)
        :param charset:
        :return:
        '''
        X, y,predict = badcase
        for i in range(0, len(X)):
            # print(i)
            img = Image.fromarray(X[i, 0, :, :])
            img.save(root_dir + str(i) + '|' + '本来为' + charset[y[i]] + '|' + '预测为' + charset[predict[i]] + '.jpg')

if __name__ == '__main__':
    dutil = DataUtil()
    (train_X, train_y), (test_X, test_y) = dutil.load_train_test_data(
        version=1,
        charset= '8B'
    )
    print(train_X.shape)
    print(test_X.shape)
    print(train_y.shape)
    print(train_X[0][0])
    print(train_y)
    print(test_y)
    dutil.show_picture(train_X[1][0])
    # dutil.show_picture(test_X[1][0])
    # dutil.show_picture(test_X[-1][0])


