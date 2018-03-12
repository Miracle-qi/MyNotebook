![Tensorflow官方中文文档](http://cwiki.apachecn.org/display/TensorFlow)
# TensorFlow学习笔记
> 本文档是个人在学习官方文档摘录的重点整理而成，可以作为官方文档简化版阅读<br>
> Copyright © Qi Shuhao. All Rights Reserved
## 初学MNIST
本节主要内容是阅读
![mnist_softmax.py](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_softmax.py)
代码，熟悉tensorflow概念和框架。
#### MNIST数据
MNIST数据分为三个部分：训练数据（mnist.train），55000个训练数据集，10,000个测试数据集（mnist.test）和5,000个验证数据集（mnist.validation）。MNIST中的每个图像都具有相应的标签，0到9之间的数字表示图像中的数字。我们把图像设为“x”，把标签设为“y”。训练集和测试集都包含图像及其相应的标签;例如训练的图像是mnist.train.images 和训练的标签是mnist.train.labels。<br>
我们可以把这个数组变成一个28×28 = 784数字的向量。只要图像之间平铺方式保持一致，那么我们如何平铺这个数组并不重要。从这个角度来看，MNIST图像只是784维向量空间中的一个点，并且结构非常复杂 （注意：计算密集型的可视化）。<br>
mnist.train.images是一个具有[55000, 784]形状的张量（n维数组）。第一个维度代表图像列表中的索引，第二个维度是每个图像中每个像素的索引。对于给定图像中的某个像素，张量中的每个元素表示0-1之间的像素强度。<br>
注意：
* softmax回归不会利用2Ｄ结构信息，但优秀的算法会利用；
* 理解“one-hot vectors”标签编码；
#### Softmax回归
MNIST中的每个图像都是零到九之间的手写数字。所以给定的图像只能有十个可能。而我们希望给定的一个图像，给出它是哪个数字的概率。例如，我们的模型会得到一个数字可能为9的图片，它的概率为80%，但它是8的概率是5%（因为8和9上部都有一个相近的圆），而代表其它数字的概率更小。**而softmax函数所要做的就是给不同的对象分配概率，并且softmax的特性决定所有概率值加起来等于1。**
softmax回归有两个步骤：首先我们对输入图片属于某个数字类别的依据进行叠加，然后将该依据转换成概率。
#### 回归实现
1. tensorflow
```
import tensorflow as tf
```
2. 创建占位符
```
x = tf.placeholder(tf.float32, [None, 784]) 
```
3. 创建模型参数变量，并初始化为0
```
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```
4. softmax模型
```
y = tf.nn.softmax(tf.matmul(x, W) + b) 
```
5. 交叉熵损失函数
```
y_ = tf.placeholder(tf.float32, [None, 10]) 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) 
```
6. 梯度下降优化算法
```
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) 
```
![其他优化算法](https://www.tensorflow.org/api_guides/python/train#Optimizers)<br>
7. 启动会话
```
sess = tf.InteractiveSession() 
```
8. 创建一个操作来初始化我们创建的变量
```
tf.global_variables_initializer().run() 
```
9. 开始训练
```
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) 
```
10. 计算准确率
```
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
y_: mnist.test.labels}))

```
