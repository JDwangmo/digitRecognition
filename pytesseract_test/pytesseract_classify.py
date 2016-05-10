#encoding=utf8
from dataProcessing.read_data import load_pix
import pytesseract
import Image
import matplotlib.pyplot as plt
import numpy as np
num_train = 50
num_test = 1000

# 设置训练数据和测试数据的路径
train_file_path = '/home/jdwang/PycharmProjects/digitRecognition/train_test_data/' \
                  '20160426/train_%d.csv'%(num_train)
test_file_path = '/home/jdwang/PycharmProjects/digitRecognition/train_test_data/' \
                 '20160426/test_%d.csv'%(num_test)

train_pix,train_y,train_im_name = load_pix(train_file_path,
                                           shape=(15,15),
                                           shuffle=True,
                                           normalize=False
                                           )

test_pix,test_y,test_im_name = load_pix(test_file_path,
                                        shape=(15, 15),
                                        shuffle=True,
                                        normalize=False
                                        )

im = Image.fromarray(train_pix[1])

print pytesseract.image_to_string(im)

plt.imshow(im,cmap=plt.cm.gray)
plt.show()

