# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-17'
    Email:   '383287471@qq.com'
    Describe:
"""

# 在这里设置变量
# 选择测试的模型
CHOICE = 2
# 设置图片文件夹
# data_dir='./image_data/'
# data_dir='/home/jdwang/PycharmProjects/digitRecognition/image_data/newData_20160801/'
data_dir='/home/jdwang/PycharmProjects/digitRecognition/image_data/20160426_modify/'


import csv
import struct
from keras.layers import Convolution2D,Activation,MaxPooling2D,Flatten,Merge,Dense,Dropout
from keras import backend as K
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import Image

class DigitRecognizationModel(object):


    def __init__(self):
        nkerns1 = [3, 5, 7]
        nkerns2 = [1, 2, 3]
        root_path ='./model/'
        # 34 class
        self.model_all,self.conv1_output = self.load_cnn_model(
            root_path+'25_0_0_400_0_lr0.001_0model_weights_25-400_firstCNN_final.h5',
            nb_classes=34,
            input_shape=(15,15),
            layer1=25,
            hidden1=400,
            nkerns=nkerns1,
        )

        # self.save_cnn_weight_to_bininary_file(self.model_all,file_name='model_all_weight.mat')







        # 0D
        self.model_binary_0D, self.conv1_output = self.load_cnn_model(
            root_path + '32_0_0_300_0_lr0.001_0model_weights_0-13_final.h5',
            nb_classes=2,
            input_shape=(15, 8),
            layer1=32,
            hidden1=300,
            nkerns=nkerns1,
        )

        # self.save_cnn_weight_to_bininary_file(self.model_binary_0D,'model_0D_weight.mat')



        # 1I
        self.model_binary_1I,self.conv1_output = self.load_cnn_model(
            root_path+'32_0_0_300_0_lr0.001_0model_weights_1-18_final.h5',
            nb_classes=2,
            input_shape=(8, 15),
            layer1=32,
            hidden1=300,
            nkerns=nkerns1,
        )

        self.save_cnn_weight_to_bininary_file(self.model_binary_1I,'model_1I_weight.mat')
        # return
        # 2Z
        self.model_binary_2Z,conv1_output = self.load_cnn_model(
            root_path+'32_0_0_800_0_lr0.001_0model_weights_2-33_final.h5',
            nb_classes=2,
            input_shape=(8, 15),
            layer1=32,
            hidden1=800,
            nkerns=nkerns1,
        )
        self.save_cnn_weight_to_bininary_file(self.model_binary_2Z,'model_2Z_weight.mat')

        # 56
        self.model_binary_56,conv1_output = self.load_cnn_model(
            root_path + '32_0_0_600_0_lr0.001_0model_weights_5-6_1&4_final.h5',
            nb_classes=2,
            input_shape=(8, 8),
            layer1=32,
            hidden1=600,
            nkerns=nkerns1,
        )
        self.save_cnn_weight_to_bininary_file(self.model_binary_56,'model_56_weight.mat')



        # 8B
        self.model_binary_8B,conv1_output = self.load_cnn_model(
            root_path + '32_0_0_600_0_lr0.001_0model_weights_8-11_final.h5',
            nb_classes=2,
            input_shape=(15, 8),
            layer1=32,
            hidden1=600,
            nkerns=nkerns1,
        )
        self.save_cnn_weight_to_bininary_file(self.model_binary_8B,'model_8B_weight.mat')

        character_name = list('0123456789ABCDEFGHIJKLMNPQRSTUWXYZ')
        self.index_to_char = lambda x:character_name[x]

        # 4A

        self.model_binary_4A, self.conv1_output = self.load_cnn_model(
            root_path + '25_0_0_200_0_lr0.001_0model_weights_4-10_1&16_final.h5',
            nb_classes=2,
            input_shape=(4, 5),
            layer1=25,
            hidden1=200,
            nkerns=nkerns2,
        )
        # self.save_cnn_weight_to_bininary_file(self.model_binary_4A,'model_4A_weight.mat')
        return

    def load_cnn_model(
            self,
            weights_path=None,
            nb_classes=2,
            input_shape=(15, 15),
            layer1=None,
            hidden1=None,
            nkerns=None
    ):
        lr = 1e-3
        model,conv1_output = self.net_model(layer1,
                               hidden1,
                               input_shape[0],
                               input_shape[1],
                               nkerns=nkerns,
                               nb_classes=nb_classes,
                               lr=lr
                               )
        self.conv1_output = conv1_output
        model.load_weights(weights_path)
        return model,conv1_output

    def save_cnn_weight_to_bininary_file(self,model,file_name):
        model_weights = model.get_weights()
        # print(model_weights)
        # print(len(model_weights))

        with open('/home/jdwang/PycharmProjects/digitRecognition/submit/'+file_name, 'wb') as fout:
            for weight in model_weights:
                print(weight.shape)
                # print(weight[0][0])
                # quit()
                for s in weight.shape:
                    fout.write(struct.pack('i', s))

                for item in weight.flatten():
                    fout.write(struct.pack('f', item))


    def cnn_binary_predict(self,
                           test_X,
                           class1,
                           class2
                           ):

        if class1 == 0 or class2 == 13:
            model = self.model_binary_0D
        elif class1 == 1 or class2 == 18:
            model = self.model_binary_1I
        elif class1 == 2 or class2 == 33:
            model = self.model_binary_2Z
        elif class1 == 5 or class2 == 6:
            model = self.model_binary_56
        elif class1 == 4 or class2 == 10:
            model = self.model_binary_4A
        elif class1 == 8 or class2 == 11:
            model = self.model_binary_8B
        else:
            raise NotImplementedError

        predicted = model.predict_classes([test_X, test_X, test_X], verbose=0)
        for i in range(0, len(predicted)):
            if (predicted[i] == 0):
                predicted[i] = class1
            else:
                predicted[i] = class2
        return predicted

    def outPutImage(self, root_dir='./badcase/',
                    badcase=None):  # 将badcase的图片输出到Badcase的文件夹中
        X,y = badcase
        for i in range(0, len(X)):
            # print(i)
            img = Image.fromarray(X[i, 0, :, :])
            img.save(root_dir +str(i)+'|'+'预测为' +self.index_to_char(y[i]) + '.jpg')

    def batch_predict(self,test_X,test_y,verbose=1):
        # 批量预测
        # 34分类预测
        # test_y = np_utils.to_categorical(test_y,34)
        predicted = self.model_all.predict_classes(
            [test_X, test_X, test_X],
            verbose=0
        )
        # print(predicted)
        index_0_13 = []
        index_1_18 = []
        index_2_33 = []
        index_5_6 = []
        index_4_10 = []
        index_8_11 = []
        # 将符合二分类的挑出，等待进一步二分类
        for i in range(0, len(predicted)):
            if predicted[i] == 0 or predicted[i] == 13:
                index_0_13.append(i)
            elif predicted[i] == 1 or predicted[i] == 18:
                index_1_18.append(i)
            elif predicted[i] == 2 or predicted[i] == 33:
                index_2_33.append(i)
            elif predicted[i] == 5 or predicted[i] == 6:
                index_5_6.append(i)
            elif predicted[i] == 4 or predicted[i] == 10:
                index_4_10.append(i)
            elif predicted[i] == 8 or predicted[i] == 11:
                index_8_11.append(i)
                # 进行二分类
        if len(index_0_13) > 0:
            binary_predicted = self.cnn_binary_predict(
                test_X[index_0_13, :, :, 0:8],
                0,
                13
            )
            predicted[index_0_13] = binary_predicted
        if len(index_1_18) > 0:
            binary_predicted = self.cnn_binary_predict(
                test_X[index_1_18, :, 0:8, :],
                # test_X[6263:6265, :, 0:8, :],
                1,
                18
            )
            predicted[index_1_18] = binary_predicted
        if len(index_2_33) > 0:
            binary_predicted = self.cnn_binary_predict(
                test_X[index_2_33, :, 0:8, :],
                2,
                33
            )
            predicted[index_2_33] = binary_predicted
        if len(index_5_6) > 0:
            binary_predicted = self.cnn_binary_predict(
                test_X[index_5_6, :, 7::, 0:8],
                5,
                6
            )
            predicted[index_5_6] = binary_predicted
        if len(index_4_10) > 0:
            binary_predicted = self.cnn_binary_predict(
                test_X[index_4_10, :, 11::, 0:5],
                4,
                10
            )
            predicted[index_4_10] = binary_predicted
        if len(index_8_11) > 0:
            binary_predicted = self.cnn_binary_predict(
                test_X[index_8_11, :, :, 0:8],
                8,
                11
            )
            predicted[index_8_11] = binary_predicted

        if verbose>0:
            # >0 返回更详细信息
            pass
        else:
            return predicted
        # 得到最终的测试结果
        test_accuracy = np.mean(np.equal(predicted, test_y))

        # 将badcase写入csv文件：ID，PREDICT,ORIGINALClass
        csvfile = file('cnn_test_badcase.csv', 'wb')
        spamwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['ID', 'PREDICT', 'CALSS'])
        index = []
        print('预测结果为：')

        # print([ self.index_to_char(item) for item in predicted])
        for i in range(0, len(test_y)):
            if predicted[i] != test_y[i]:
                spamwriter.writerow([i, self.index_to_char(predicted[i]), self.index_to_char(test_y[i])])
                index.append(i)
        csvfile.close()
        print('准确率为：%f'%(test_accuracy))
        if len(test_X[index])==0:
            print('没有basecase！')
        else:
            print('basecase有:%d个'%(len(test_X[index])))
            print('已经保存到文件中：cnn_test_badcase.csv')
            # 将错误结果的图片输出到Badcase的文件夹中
            self.outPutImage(root_dir='./badcase/',
                             badcase = (test_X[index],predicted[index]))
        return test_accuracy,predicted  # 返回最终测试的准确率

    def predict(self,image_path):
        # 识别一张图片
        pic = Image.open(open(image_path))
        pix = np.asarray(pic).reshape(1,1,15,15)
        # print(pix)
        result = self.batch_predict(pix,
                           None,
                           verbose=0)
        print('预测结果为：%s'%(self.index_to_char(result)))

    def net_model(self, layer1, hidden1, rows, cols, nkerns, nb_classes, lr=0.01, decay=1e-6, momentum=0.9):
        layer1_model1 = Sequential()
        layer1_model1.add(Convolution2D(layer1, nkerns[0], nkerns[0],
                                        border_mode='valid',
                                        input_shape=(1, rows, cols)))
        layer1_model1.add(Activation('tanh'))
        layer1_model1.add(MaxPooling2D(pool_size=(2, 2)))
        layer1_model1.add(Flatten())  # 平铺
        # layer1_model1.get_weights()
        # quit()
        layer1_model2 = Sequential()
        layer1_model2.add(Convolution2D(layer1, nkerns[1], nkerns[1],
                                        border_mode='valid',
                                        input_shape=(1, rows, cols)))
        layer1_model2.add(Activation('tanh'))
        layer1_model2.add(MaxPooling2D(pool_size=(2, 2)))
        layer1_model2.add(Flatten())  # 平铺


        layer1_model3 = Sequential()
        layer1_model3.add(Convolution2D(layer1, nkerns[2], nkerns[2],
                                        border_mode='valid',
                                        input_shape=(1, rows, cols)))
        layer1_model3.add(Activation('tanh'))
        layer1_model3.add(MaxPooling2D(pool_size=(2, 2)))
        layer1_model3.add(Flatten())  # 平铺

        model = Sequential()

        model.add(Merge([layer1_model2, layer1_model1, layer1_model3], mode='concat', concat_axis=1))  # merge
        # print(len(model.layers[0]))
        # quit()

        model.add(Dense(hidden1))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.summary()
        conv1_output = K.function(inputs=[
            layer1_model1.layers[0].input,
            layer1_model2.layers[0].input,
            layer1_model3.layers[0].input,
            K.learning_phase(),
        ],
            outputs=[
                layer1_model1.layers[0].output,
                layer1_model2.layers[0].output,
                layer1_model3.layers[0].output,
                model.layers[-2].output,
                model.layers[-1].output,
            ]
        )




        sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

        return model,conv1_output



from submit.data_util import load_pic,img_to_vector
# 加载模型
model = DigitRecognizationModel()
# quit()
# ==1, 则是单图片测试
# !=1, 批量预测
choice = CHOICE
if choice==1:
    # 单图片测试
    print('逐张图片测试：')
    while True:
        s = raw_input("输入图片的路径（建议使用绝对路径）: ")
        model.predict(s)
else:
    # 批量预测
    # 读取数据
    print('=='*20)
    print('加载数据。。')
    print('=='*20)
    # 设置图片路径
    test_X,test_y = img_to_vector(data_dir=data_dir,
                                  save_vector=False,)
    # print(test_X[0])
    # 将图片保存成 二进制 形式
    with open('/home/jdwang/PycharmProjects/digitRecognition/submit/input_data2.mat', 'wb') as fout:
        print(test_X.shape)
        fout.write(struct.pack('i', len(test_X)))

        for in_data in test_X:
            # print(weight[0][0])
            # quit()
            for item in in_data.flatten():
                fout.write(struct.pack('f', item))

    with open('/home/jdwang/PycharmProjects/digitRecognition/submit/input_data2_label.mat', 'wb') as fout:
        fout.write(struct.pack('i', len(test_y)))
        for item in test_y:
            fout.write(struct.pack('i', item))

    # test_X,test_y = load_pic('./data_vector.pickle')
    # print(test_y)
    # test_Xx = test_X[:,:,11::, 0:5]
    # layer_output = model.conv1_output([test_Xx,test_Xx,test_Xx,0])
    test_accuracy, y_pred = model.batch_predict(test_X,test_y)
    with open('/home/jdwang/PycharmProjects/digitRecognition/submit/input_data_label_pred.mat', 'wb') as fout:
        fout.write(struct.pack('i', len(y_pred)))
        for item in y_pred:
            fout.write(struct.pack('i', item))
    # print(np.mean(y_pred==test_y))
    # print(np.sum(y_pred==test_y))

    # print(layer_output[0])
    # print(layer_output.shape)
    quit()
    print('==' * 20)
    print('预测中。。')
    model.batch_predict(test_X,test_y)

