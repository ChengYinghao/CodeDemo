# CodeDemo
该项目为个人机器学习项目，目标为识别中国的四位验证码（数字加字母），其中尝试了两种识别验证码的方法：多层感知机识别和卷积神经网络识别，目前项目尚未完结。

数据集产生方法：
两种方法都用scikit-image库生成验证码数据，scikit-image库能够接收PIL库导出的numpy数组格式的图像数据进行错切变化。

两种方法简单介绍：
多层感知机识别：
这部分先将图像切分为单个的字母，再用简单三层结构的多层感知机（sklearn库中的MLPclassifier）将字母识别问题转化为分类问题。已经实现的部分为对四个英文字母进行识别，准确率90%以上。

cnn识别：
这部分将四位验证码图片看作整体进行识别，网络结构为三层卷积（+池化）和一层全连接层，其中也有dropout等防止过拟合的部分，本质上也是分类问题。已经实现的部分为对四个数字组成的验证码进行识别，准确率99%以上。

代码运行：
MLP方法可运行MLP_method目录下的MLP.py用已经训练好的模型对随机验证码图像进行预测并输出结果。

cnn方法可运行CNN_method目录下的Network_onlyNumbers.py用训练好的模型对随机验证码图像进行预测并输出结果。

结果：
MLP方法下对分割后的单个字母预测准确率超过95%，但是对整体全部预测正确的准确率还不是很高。

cnn方法下对4位验证码整体预测准确率超过99%，接下来准备训练神经网络预测同时包含数字和字母的验证码。

This project is a personal machine learning project, aiming at identifying China's four-digit verification code (numeral plus letter), in which two methods of identifying verification code are tried: multi-layer perceptron recognition and convolution neural network recognition,The project is not yet complete.

Data set generation method:Both methods generate captcha data with the scikit-image library, which can receive image data in the numpy array format exported by the PIL library for miscutting changes.

A brief introduction to the two methods：
Multi-layer perceptron recognition:This part divides the image into single letters, and then transforms the letter recognition problem into classification problem by using a simple three-layer multi-layer perceptron (MLPclassifier in sklearn library).The part that has been implemented is the recognition of four English letters, with an accuracy rate of over 90%.

The identification of the CNN:In this part, four verification code images are identified as a whole, and the network structure is three-layer convolution (+ pooling) and a layer of full connection layer, among which there are some sections that prevent overfitting, such as dropout, which are also classification problems in nature.The implemented part is to identify the verification code composed of four Numbers, with an accuracy rate of more than 99%.

Code run:
The MLP method can run the mlp.py under the MLP_method directory to predict the random verification code image with the trained model and output the results.

CNN method can run network_onlynumber.py in CNN_method directory to predict the random verification code image and output the result with the trained model.

Result:
Under the MLP method, the accuracy of the single letter after segmentation is over 95%, but the accuracy of the overall prediction is not very high.

Under the cnn method, the overall prediction accuracy of the 4-digit verification code is over 99%, and then it is ready to train the neural network to predict the verification code containing both numbers and letters.
