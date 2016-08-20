# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-19'
    Email:   '383287471@qq.com'
    Describe:
"""

from matplotlib import pyplot as plt
import pickle

charset = '8B'
conv1_output_file_path = '/home/jdwang/PycharmProjects/digitRecognition/cnn/result/result_%s.pickle'%charset

with open(conv1_output_file_path) as fin:
    test_X = pickle.load(fin)
    test_y = pickle.load(fin)
    test_predict = pickle.load(fin)
    predict = test_predict[0]
    conv1_output = test_predict[1]
    print(test_X.shape)
    print(conv1_output.shape)
    # print(test_X[-1])
    # quit()
    image_index = 10
    plt.subplot(6, 1, 1)
    dst = test_X[image_index][0]
    plt.imshow(dst, cmap='gray', interpolation='bicubic')
    # print(dst)
    plt.xticks([]), plt.yticks([])
    for i in range(20):
        plt.subplot(6,4,5+i)
        dst = conv1_output[image_index][i]
        plt.imshow(dst, cmap='gray', interpolation='bicubic')
        # print(dst)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # print(dst*255)

    # plt.subplot(5,5,2)
    # dst = conv1_output[1].reshape(10,8,8)[0]
    # plt.imshow(dst, cmap='gray', interpolation='bicubic')
    # print(dst)
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    plt.clf()
    plt.close()
    # print(conv1_output[0].reshape(1,19,19))
