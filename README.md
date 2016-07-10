# digitRecognition
###Digit Recognition

### summary:
1. 

### 环境:
- Ubuntu 14.04 / Linux mint 17.03
- Python: 2.7.6版本.
- 

### 数据：
1. train_test_data/20160426/image_data.csv

## 项目架构


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