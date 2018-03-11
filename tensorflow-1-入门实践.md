![Tensorflow官方中文文档](http://cwiki.apachecn.org/display/TensorFlow)
# TensorFlow学习笔记
> 本文档是个人在学习官方文档摘录的重点整理而成，可以作为官方文档简化版阅读<br>
> Copyright © Qi Shuhao. All Rights Reserved
## 入门实践
首先要理解TensorFlow中的中心数据单位是张量（tensor），张量由一组成形为任意数量的数组的原始值组成，而张量的等级是其维数。下面跟着官方文档
利用Tensorflow建立一个简单的神经网络：
###导入TensorFlow
TensorFlow程序的规范导入声明如下：
```
import tensorflow as tf
```
### 计算图 
计算图形是一系列排列成节点的图形TensorFlow操作。每个节点采用零个或多个张量作为输入，并产生张量作为输出。而TensorFlow Core程序往往
由两个独立部分组成：<br>
* 构建计算图
* 运行计算图<br>
#### 创建常数节点
一种类型的节点是一个常数。像所有TensorFlow常数一样，它不需要任何输入，它输出一个内部存储的值。创建两个浮点式传感器node1，node2如下所示：
```
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
```
#### 创建会话
会话封装了TensorFlow运行时的控制和状态，创建会话：
```
sess = tf.Session()
```
#### 创建操作节点
可以通过将Tensor节点与操作相结合来构建更复杂的计算（操作也是节点）。例如，添加我们的两个常量节点并生成一个新的图，如下所示：
```
node3 = tf.add(node1, node2)
```

#### 创建占位符
可以将图形参数化为接受外部输入，称为占位符。一个占位符是一个承诺后提供一个值。
```
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
```

#### 创建变量
在机器学习中，我们通常会想要一个可以接受任意输入的模型，比如上面的一个。
为了使模型可训练，我们需要能够修改图形以获得具有相同输入的新输出。 变量允许我们向图中添加可训练的参数。它们的构造类型和初始值：
```
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
```

#### 初始化变量
常数被调用时初始化tf.constant，其值永远不会改变。相比之下，调用时，变量不会被初始化tf.Variable。
要初始化TensorFlow程序中的所有变量，必须显式调用特殊操作，如下所示：
```
init = tf.global_variables_initializer()
sess.run(init)
```
#### 占位符赋值
重要的是实现initTensorFlow子图的一个句柄，初始化所有的全局变量。在我们调用之前sess.run，变量未初始化。
既然x是占位符，我们可以同时评估linear_model几个值，x如下所示：
```
print(sess.run(linear_model, {x:[1,2,3,4]}))
```
#### 创建损失函数
我们创建了一个模型，但是我们不知道它有多好。为了评估培训数据的模型，我们需要一个y占位符来提供所需的值，我们需要编写一个损失函数。
损失函数测量当前模型与提供的数据之间的距离。我们将使用线性回归的标准损失模型，其将当前模型和提供的数据之间的三角形的平方相加。linear_model - y创建一个向量，其中每个元素都是对应的示例的错误增量。我们打电话tf.square给这个错误。
然后，我们求和所有平方误差，创建一个单一的标量，使用tf.reduce_sum以下方法抽象出所有示例的错误：
```
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
```
我们可以手动重新分配的值提高这W和b为-1和1变量的值，完美初始化为提供的价值 tf.Variable，但可以使用操作等来改变tf.assi
```
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
```
#### 优化器
机器学习的完整讨论超出了本教程的范围。然而，TensorFlow提供了优化器，缓慢地更改每个变量，以便最大程度地减少损失函数。最简单的优化器是梯度下降。
它根据相对于该变量的损失导数的大小修改每个变量。通常，手动计算符号导数是乏味且容易出错的。
因此，TensorFlow可以使用该函数自动生成仅给出模型描述的导数tf.gradients。为了简单起见，优化器通常为您做这个。例如，
```
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```
### 完整程序
通过上面的概念讲解，大概应该读懂完整程序，这是学习Tensorflow的第一步，我们对Tensorflow有了最初的认知。
```
import numpy as np
import tensorflow as tf
 
# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})
 
# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
```
Tip：TensorFlow提供了一个名为TensorBoard的实用程序，可以显示计算图的图片，帮助人们理解模型和编写程序。下图为该程序的可视化计算图：
！[计算图可视化](MyNotebook/p-1.png)
