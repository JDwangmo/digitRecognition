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
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
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
https://mhosseiniresearch.wordpress.com/2015/08/26/cat-vs-dog-the-first-experience-on-using-a-deep-neural-network-framework-keras-on-cloud-ec2/
2. [Network In Network](https://raw.githubusercontent.com/JDwangmo/digitRecognition/master/reference/1312.4400v3-Network-in-Network.pdf)
    - M. Lin, Q. Chen, S. Yan，International Conference on Learning Representations, 2014 (arXiv:1409.1556)
    - **keywords:**
    - 本文贡献：