# digitRecognition
###Digit Recognition

### summary:
1. 

### 环境:
- Ubuntu 14.04 / Linux mint 17.03
- Python: 2.7.6版本.


### 数据：
1. `train_test_data/20160426/image_data.csv`：20160426版本所有图片的灰度值向量数据，总共41934条数据，包括34种字符（24个字母（不包括V和O）+10个数字）,分布如下：

    |0|1|2|3|4|5|6|7|8|9|A|B|C|D|E|F|G|H|I|J|K|L|M|N|P|Q|R|S|T|U|W|X|Y|Z|
    |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
    |3971|3193|2745|4688|3087|2784|3128|2797|3738|3120|305|353|786|480|174|175|171|110|172|416|679|127|170|160|527|314|172|139|159|1088|525|329|847|303|
    
2. `train_test_data/20160426_modify/image_data.csv`：20160426版本所有图片的灰度值向量数据，总共41932条数据，包括34种字符（24个字母（不包括V和O）+10个数字）,分布如下：

    |0|1|2|3|4|5|6|7|8|9|A|B|C|D|E|F|G|H|I|J|K|L|M|N|P|Q|R|S|T|U|W|X|Y|Z|
    |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
    |3970|3191|2745|4688|3087|2784|3128|2789|3738|3120|305|353|786|480|174|175|171|110|172|416|679|127|170|160|527|314|172|139|159|1088|525|329|847|240|


## 项目架构
明天计划出这些结果供参考：
（1）随机取多几次100个，比如取5次，跑出来的结果列一下（除了总正确率，细分34个类统计正确率，0和O，U和V不用区分）。
（2）放大到比如200个（估计还是有的，能多点就更好，比如500），试一下有没有提升。
（3）随机取5次100个跑SVM（其他都作为每一次的验证数据），6个特征，按照，选高斯核，gamm和C按这个配置7*7=49次 跑，看一下平均正确率。取正确率最高的10个参数组合（C和gamma），各跑10次随机搜。看看这个验证过程最后能取得多高的正确率。

（4）你那边的网络说降低一点层次，看看明晚能不能出来一个结果。

cnn_train: 



|metric| 0|1|2|3|4|5|6|7|8|9|A|B|C|D|E|F|G|H|I|J|K|L|M|N|P|Q|R|S|T|U|W|X|Y|Z  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|F1 | 0.996631|0.991070|0.994870|0.999673|0.999330|0.992329|0.997849|0.990874|0.991690|0.999503|0.985294|0.894831|0.992690|0.970551|1.000000|0.983193|0.995595|1.000000|0.838951|0.993671|0.999136|0.934066|0.977578|0.981651|0.995294|0.974828|0.991379|0.857143|0.836066|0.988247|0.995283|1.000000|0.998661|0.944186  |
|precision|0.999740|0.995434|1.000000|0.999564|0.999330|0.996618|0.999669|0.995509|0.999442|1.000000|0.990148|0.814935|0.995601|0.945137|1.000000|0.966942|1.000000|1.000000|0.736842|0.993671|1.000000|0.876289|0.990909|0.963964|1.000000|0.955157|0.982906|0.750000|0.739130|0.997936|0.997636|1.000000|0.998661|0.894273|
|recall|0.993542|0.986744|0.989792|0.999782|0.999330|0.988077|0.996037|0.986281|0.984057|0.999007|0.980488|0.992095|0.989796|0.997368|1.000000|1.000000|0.991228|1.000000|0.973913|0.993671|0.998273|1.000000|0.964602|1.000000|0.990632|0.995327|1.000000|1.000000|0.962264|0.978745|0.992941|1.000000|0.998661|1.000000|


|metric| 0|1|2|3|4|5|6|7|8|9|A|B|C|D|E|F|G|H|I|J|K|L|M|N|P|Q|R|S|T|U|W|X|Y|Z  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|F1 | 0.996631|0.991070|0.994870|0.999673|0.999330|0.992329|0.997849|0.990874|0.991690|0.999503|0.985294|0.894831|0.992690|0.970551|1.000000|0.983193|0.995595|1.000000|0.838951|0.993671|0.999136|0.934066|0.977578|0.981651|0.995294|0.974828|0.991379|0.857143|0.836066|0.988247|0.995283|1.000000|0.998661|0.944186  |
|precision|0.999740|0.995434|1.000000|0.999564|0.999330|0.996618|0.999669|0.995509|0.999442|1.000000|0.990148|0.814935|0.995601|0.945137|1.000000|0.966942|1.000000|1.000000|0.736842|0.993671|1.000000|0.876289|0.990909|0.963964|1.000000|0.955157|0.982906|0.750000|0.739130|0.997936|0.997636|1.000000|0.998661|0.894273|
|recall|0.993542|0.986744|0.989792|0.999782|0.999330|0.988077|0.996037|0.986281|0.984057|0.999007|0.980488|0.992095|0.989796|0.997368|1.000000|1.000000|0.991228|1.000000|0.973913|0.993671|0.998273|1.000000|0.964602|1.000000|0.990632|0.995327|1.000000|1.000000|0.962264|0.978745|0.992941|1.000000|0.998661|1.000000|

## 相关论文：
1. [ImageNet Classification with Deep Convolutional Neural Networks](https://raw.githubusercontent.com/JDwangmo/digitRecognition/master/reference/imagenet-classification-with-deep-convolutional-nn.pdf)
    - Alex Krizhevsky, NIPS 2012
    - **keywords: convolution. dropout.softmax.**
    - 本文贡献：
    - 模型：
        1. Our final network contains five convolutional and three fully-connected layers, and this depth seems to be important: we found that removing any convolutional layer (each of which contains no more than 1% of the model’s parameters) resulted in inferior performance.[P2] 
        2. Dataset： ImageNet is a dataset of over 15 million labeled high-resolution images belonging to roughly 22,000 categories. The images were collected from the web and labeled by human labelers using Ama-zon’s Mechanical Turk crowd-sourcing tool.
            - ILSVRC-2010 is the only version of ILSVRC for which the test set labels are available, so this is the version on which we performed most of our experiments.
            - ImageNet consists of variable-resolution images, while our system requires a constant input dimen-sionality. Therefore, we down-sampled the images to a fixed resolution of 256*256. 
        3. RelUs activition.原因是速度可以更快，而 saturating nonlinearities are much slower than the non-saturating nonlinearity f(x) = max(0;x). 

2. [Network In Network](https://raw.githubusercontent.com/JDwangmo/digitRecognition/master/reference/1312.4400v3-Network-in-Network.pdf)
    - M. Lin, Q. Chen, S. Yan，International Conference on Learning Representations, 2014 (arXiv:1409.1556)
    - **keywords:**
    - 本文贡献：
    
3. [Deep-Convolutional-Network-for-Handwritten-Chinese-Character-Recognition.pdf](https://raw.githubusercontent.com/JDwangmo/digitRecognition/master/reference/Deep-Convolutional-Network-for-Handwritten-Chinese-Character-Recognition.pdf)
    - Yuhao Zhang
    - **keywords:** chinese character; 中文字符的识别; 卷积神经网络，模型融合。
    - 主要围绕三个问题：
        1. How does **the depth of the network** influence the clas-sification accuracy? 
        2. How does **the number of filters** in the convlutional layers influence the classification accuracy? 
        3. Does **adding a convolutional layer help more or adding a fully-connected layer** help more in terms of the test accuracy? 
    - 图片预处理：
        - resizing, 
        - contrast maximization ：forced the image to have values ranging from 0 to 255
        - image mean subtraction. Given a raw input image describing
    - 结论：
        - In this project we explored the problem of recognizing handwritten Chinese characters. Specifically we use deep convolutional neural networks and achieved very impressive performance. We ran experiments on a 200-class and a 3755-class dataset using convolutional networks with different depth and filter numbers.**Our main findings are that for convolutional neural network with small filter sizes**: 
            - 1) the deeper the network, the larger the accuracy; 
            - 2) increas-ing the depth gives us diminishing returns in terms of ac-curacybut highly increases the difficulty of training; 
            - 3) increasing the filter number in a moderate range can increase the accuracy; 
            - 4) for networks with relatively few convolu-tional layers, the benefit of adding extra convolutional layer beats that of adding extra fully-connected layer. We also find that using model ensemble of networks with similar accuracies beats all single network.
    
4. [Multi-Column Deep Neural Networks for Offline Handwritten Chinese Character Classification](https://raw.githubusercontent.com/JDwangmo/digitRecognition/master/reference/Multi-Column-Deep-Neural-Networks-for-Offline-Handwritten-Chinese-Character-Classification.pdf)
    - Dan Cires ̧an and J ̈urgen Schmidhube,arXiv:1309.0261v,2013.9
    - **keywords:** 模型融合;中文字符的识别; 卷积神经网络。
    - 整理模型的方法和图不错。    
    
5. [Transfer Learning for Latin and Chinese Characters with Deep Neural Networks](https://raw.githubusercontent.com/JDwangmo/digitRecognition/master/reference/Transfer-Learning-for-Latin-and-Chinese-Characters-with-Deep-Neural-Networks.pdf)
    - Dan C. Cireşan, Ueli Meier, J̈urgen Schmidhuber. IJCNN 2012.
    - **keywords:** Trasnfer Learning(迁移学习)。
    - 挺不错一篇论文，实验方案和方法很非常详细和清晰，值得借鉴。
    - 训练过程中对图片进行 distortion（扭曲）/ deformed（变形）：`P2-left` During training a given dataset is continually deformed **prior to each epoch of an online learning algorithm**. Deformations are stochastic and applied to each image during training, using random but bounded values for **translation, rotation and scaling**. These distortions allow us to train DNN with many free parameters without overfitting and greatly improve generalization performance. Training stops when either the validation error becomes 0, the learning rate reaches its predefined minimum or there is no improvement on the validation set for 50 consecutive epochs. The undistorted, original training set is used as validation set (使用没有扭曲的训练集图片做验证集)。
    - 迁移学习的思想： Is learning the n-th thing any easier than learning the first? 可以举出很多例子，比如如果你学会骑单车，那么将会更容易学会开摩托等等。
    - 迁移学习方案：
        - A. Latin characters: from digits to uppercase letters ：通过比较随机初始化和使用数字训练出来的权重初始化，发现确实有帮助。
        - B. Learning uppercase letters from few samples per class：通过比较不同的训练集大小，发现小训练集下， pre-training 效果更加明显。
        - C. Chinese characters to uppercase Latin letters： 使用更复杂的中文字符，效果更好。
        - D. Chinese characters: speeding up training： 使用训练子集或者字母来 pre-training 中文字符模型，同样有效果。甚至发现，transfer learning from Latin letters to Chinese characters works as well as pretraining a net with 1% of the classes of the Chinese training task, despite the lower apparent complexity of Latin letters。 
    - 结论：
        - Transfer learning between different handwritten character recognition tasks is possible.
        - 迁移学习的好处有：
            - less training time is needed to obtain good results, 
            - and much better results are obtained when only few labeled samples per class are available for the destination task. 
    
6. [Convolutional Neural Network Committees For Handwritten Character Classification](https://raw.githubusercontent.com/JDwangmo/digitRecognition/master/reference/Convolutional-Neural-Network-Committees-For-Handwritten-Character-Classification.pdf)
    - 
    - **keywords:** 模型
