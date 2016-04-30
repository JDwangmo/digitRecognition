#encoding=utf8
from dataProcessing.read_data import load_pix
import logging
import numpy as np
import pandas as pd
import timeit

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )

num_train = 50
num_test = 1000

train_file_path = '/home/jdwang/PycharmProjects/digitRecognition/train_test_data/' \
                  '20160426/train_%d.csv'%(num_train)
test_file_path = '/home/jdwang/PycharmProjects/digitRecognition/train_test_data/' \
                 '20160426/test_%d.csv'%(num_test)

train_pix,train_y,train_im_name = load_pix(train_file_path,
                     shape=(1,15*15)
                     )

test_pix,test_y,test_im_name = load_pix(test_file_path,
                    shape=(1, 15 * 15)
                    )
logging.debug( 'the shape of train sample:%d,%d'%(train_pix.shape))
logging.debug( 'the shape of test sample:%d,%d'%(test_pix.shape))

def classify(inputPoint,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]	 #已知分类的数据集（训练集）的行数
    #先tile函数将输入点拓展成与训练集相同维数的矩阵，再计算欧氏距离
    diffMat = np.tile(inputPoint,(dataSetSize,1))-dataSet  #样本与训练集的差值矩阵
    sqDiffMat = diffMat ** 2					#差值矩阵平方
    sqDistances = sqDiffMat.sum(axis=1)		 #计算每一行上元素的和
    distances = sqDistances ** 0.5			  #开方得到欧拉距离矩阵
    sortedDistIndicies = distances.argsort()	#按distances中元素进行升序排序后得到的对应下标的列表
    #选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[ sortedDistIndicies[i] ]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #按classCount字典的第2个元素（即类别出现的次数）从大到小排序
    sortedClassCount = sorted(classCount.items(), key = lambda x:x[1], reverse = True)
    return sortedClassCount[0][0]

start = timeit.default_timer()
logging.debug('开始分类...')
pred_result = []
for index,test in enumerate(test_pix):
    if (index+1)%100 == 0:
        logging.debug('已处理%d张图片！'%(index+1))
    # print test
    result = classify(test,train_pix,train_y,3)
    pred_result.append(result)


is_correct = (pred_result==test_y)
print '正确的个数：%d'%(sum(is_correct))
print '正确率：%f'%(sum(is_correct)/(len(test_y)*1.0))

test_result = pd.DataFrame({
            'label':test_y,
            'pred':pred_result,
            'is_correct':is_correct,
            'image_id':test_im_name
            })

test_result_path = '/home/jdwang/PycharmProjects/digitRecognition/knn/result/20160426/' \
                   'knn_result_%d_%d.csv'%(num_train,num_test)
test_result.to_csv(test_result_path,sep='\t')

end = timeit.default_timer()
logging.debug('总共运行时间:%ds' % (end-start))