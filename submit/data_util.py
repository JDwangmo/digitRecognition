# encoding=utf8


import os
import io
import Image
import numpy as np
import pickle


def char_to_index(char):
    character_name = list('0123456789ABCDEFGHIJKLMNPQRSTUWXYZ')
    if character_name.__contains__(char):
        return character_name.index(char)
    else:
        raise NotImplementedError

def img_to_vector(
        data_dir='/home/jdwang/PycharmProjects/digitRecognition/submit/image_data/',
save_vector=True):
    output_file_path = 'data_vector.pickle'

    # print(output_file_path)
    sub_dir_list = sorted(os.listdir(data_dir))

    print('子文件个数为：%d，有：%s' % (len(sub_dir_list), str(sub_dir_list)))
    image_id=[]
    X = []
    y =[]
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
                    y.append(char_to_index(sub_dir))
                    image_id.append(sub_dir + '/' + train_file)
                    # pic_grey = ','.join([str(item) for item in pix])
                    # print(sub_dir+'\t'+sub_dir + '/' + train_file + '\t' + pic_grey)
                    # quit()
                    # fout.write( u'%s\t%s\t%s\n' %(sub_dir ,sub_dir + '/' + train_file, pic_grey))

    print('测试个数：%d'(len(X)))
    X = np.asarray(X).reshape(-1,1,15,15)
    y= np.asarray(y,dtype=int)
    # print(X.shape)
    # print(y)
    if save_vector:
        with open(output_file_path,'wb') as fout:
            pickle.dump(X,fout)
            pickle.dump(y,fout)
    return X,y


def load_pic(path):
    with open(path, 'rb') as fin:
        train_X = np.asarray(pickle.load(fin))
        train_y = np.asarray(pickle.load(fin))
    return train_X,train_y

if __name__ == '__main__':
    img_to_vector()