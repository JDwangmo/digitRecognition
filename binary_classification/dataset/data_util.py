# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-11-01'; 'last updated date: 2016-11-01'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
import pickle
import os
import numpy as np
from sklearn.cluster import KMeans
import Image
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class DataUtil:
    Character_Name = list('0123456789ABCDEFGHIJKLMNPQRSTUWXYZ')

    def __init__(self):
        pass

    @staticmethod
    def load_train_test_data(option='1-1', binary_classes='0D'):
        """ 加载训练数据、验证数据和测试数据

        :param option: str
            数据集版本
        :param binary_classes: int
            待分类字符的数据集
        :return:
            (train_X, train_y), (val_X, val_y), (test_X, test_y)
        """
        if option == '1-1':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161101'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_data1.pickle')
        elif option == '1-2':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161101'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_data2.pickle')
        elif option == '1-3':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161101'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_data3.pickle')
        elif option == '1-4':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161101'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_data4.pickle')
        elif option == '1-5':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161101'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_data5.pickle')
        elif option == '2-1':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161102'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_avgValandTest_data1.pickle')
        elif option == '2-2':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161102'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_avgValandTest_data2.pickle')
        elif option == '2-3':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161102'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_avgValandTest_data3.pickle')
        elif option == '2-4':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161102'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_avgValandTest_data4.pickle')
        elif option == '2-5':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161102'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_avgValandTest_data5.pickle')
        elif option == '3-1':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161104'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_avgValandTest_data1.pickle')
        elif option == '3-2':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161104'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_avgValandTest_data2.pickle')
        elif option == '3-3':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161104'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_avgValandTest_data3.pickle')
        elif option == '3-4':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161104'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_avgValandTest_data4.pickle')
        elif option == '3-5':
            root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/dataset20161104'
            path = os.path.join(root_path, 'TrainSet&TestSet_3400_avgValandTest_data5.pickle')
        else:
            raise NotImplementedError

        with open(path, 'rb') as train_file:
            train_X = pickle.load(train_file)
            train_y = np.asarray(pickle.load(train_file))
            val_X = pickle.load(train_file)
            val_y = np.asarray(pickle.load(train_file))
            test_X = pickle.load(train_file)
            test_y = np.asarray(pickle.load(train_file))

        selected_samples = np.array(map(lambda x: DataUtil.Character_Name[x] in list(binary_classes), train_y))
        train_X = train_X[selected_samples]
        train_y = train_y[selected_samples]

        selected_samples = np.array(map(lambda x: DataUtil.Character_Name[x] in list(binary_classes), val_y))
        val_X = val_X[selected_samples]
        val_y = val_y[selected_samples]

        selected_samples = np.array(map(lambda x: DataUtil.Character_Name[x] in list(binary_classes), test_y))
        test_X = test_X[selected_samples]
        test_y = test_y[selected_samples]

        # train_y = np_utils.to_categorical(train_y, nb_classes)
        # val_y = np_utils.to_categorical(val_y, nb_classes)
        # test_y = np_utils.to_categorical(test_y, nb_classes)

        return (train_X, train_y), (val_X, val_y), (test_X, test_y)

    @staticmethod
    def show_image(image_array):
        """ 输入二维数组，显示图片

        :param image_array: np.array
            2D 数组
        :return: None
        """
        image = Image.fromarray(image_array)
        image.show()

    @staticmethod
    def save_image(image_array, file_name):
        """ 保存图片

        :param file_name: str
            保存的图片名
        :param image_array: np.array
            2D 数组
        :return: None
        """
        # 二维数组转为图片对象
        image = Image.fromarray(image_array)
        # 文件名增加后缀
        if not file_name.endswith('.bmp'):
            file_name += '.bmp'
        # 保存图片
        image.save(file_name)

    @staticmethod
    def outlier_detection(X, label):
        """ 检测 标为 label 的数据集中，是否有异常点
            1、所有图片PCA降维并显示到图片上，观察图像
            2、聚类 ——> 将聚类中心绘制出来，观察

        :param X:
        :param label: str
            被标记的 label
        :return: None
        """

        X_vector = X.reshape(len(X), -1)

        # region 将所用样例进行聚类
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=10, n_jobs=4)
        class_arr = kmeans.fit_predict(X_vector)

        # endregion

        root_path = '/home/jdwang/PycharmProjects/digitRecognition/binary_classification/dataset/badcase/'

        for cluster_id in range(n_clusters):
            # 该 聚类的 个数
            print(cluster_id, sum(class_arr == cluster_id))

            img_dir = root_path + '%s/%d(%d)' % (label, cluster_id, sum(class_arr == cluster_id))
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            for index, img in enumerate(X[class_arr == cluster_id]):
                DataUtil.save_image(img[0], os.path.join(img_dir, str(index)))

        # region PCA 可视化
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)

        cluster_centers_2dim = pca.fit_transform(kmeans.cluster_centers_)
        X_vector_2dim = pca.fit_transform(X_vector)
        # print(cluster_centers_2dim)

        fig = plt.figure()
        # 聚类中心
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(cluster_centers_2dim[:, 0], cluster_centers_2dim[:, 1], cluster_centers_2dim[:, 2])  # scatter绘制散点
        for txt in range(len(cluster_centers_2dim)):
            ax1.text(cluster_centers_2dim[txt, 0], cluster_centers_2dim[txt, 1], cluster_centers_2dim[txt, 2], txt)
        # 每个样例
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(X_vector_2dim[:, 0], X_vector_2dim[:, 1], X_vector_2dim[:, 2])  # scatter绘制散点
        for txt in range(len(X_vector_2dim)):
            ax2.text(X_vector_2dim[txt, 0], X_vector_2dim[txt, 1], X_vector_2dim[txt, 2], txt)
        # 增加网格
        plt.grid()
        # plt.show()
        plt.savefig(root_path + '%s' % label, dpi=200)
        # 清理内存
        fig.clf()
        fig.clear()
        # endregion


def batch_outlier_detection():
    """批量 数据集 的异常点检测
        分别对每个字符数据集 进行异常点检测

    :return: None
    """
    for char in DataUtil.Character_Name:
        (train_X, train_y), (val_X, val_y), (test_X, test_y) = DataUtil.load_train_test_data(option='3-1',
                                                                                             binary_classes=char)

        X = np.concatenate((train_X, val_X, test_X), axis=0)
        y = np.concatenate((train_y, val_y, test_y), axis=0)

        print(y, len(X))

        DataUtil.outlier_detection(X, char)


if __name__ == '__main__':
    batch_outlier_detection()
