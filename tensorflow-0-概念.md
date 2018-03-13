# TensorFlow学习笔记
> 本文主要针对初学者搭建起tensorflow框架的基本认识
> Copyright © Qi Shuhao. All Rights Reserved
## 基本概念


### tensor张量
TensorFlow的数据模型为tensor（张量），可简单理解为类型化的多维数组，0维张量是一个数字，也称为标量， 1维张量称为“向量”， 2维张量称为矩阵…… 但和Numpy中的数组不同是，一个张量中主要保留了三个属性：name，shape，type。


### session会话
TensorFlow的运行模型 为 session（会话），会话拥有和管理TensorFlow程序运行时的所有资源，当计算完成后，需要关闭会话来帮助系统回收资源， 否则就可能出现资源泄露的问题， TensorFlow使用会话的方式主要有如下两种模式：
```
#env = python 2.7
sess = tf.Session()#创建一个会话
sess.run(result) #运行此会话
Sess.close() #运行完毕关闭此会话
```
```
#env = python 2.7
with tf.Session() as sess:
sess.run(result)#通过python的上下文管理器来使用会话，当上下文退出时会话管理和资源释放也自动完成
```
当使用第一种模式，程序因为异常而退出时，关闭会话的函数就可能不会执行从而导致资源泄露，因此，推荐使用第二种模式。


### graph图
TensorFlow的计算模型为grap（图） ，一个TensorFlow图描述了计算的过程.为了进行计算, 图必须在会话里被启动. 会话将图的op(节点)分发到诸如CPU或GPU之类的设备上, 同时提供执行op的方法. 这些方法执行后,将产生的tensor返回. 在Python语言中,返回的tensor是numpy ndarray对象; 在C/C++语言中, 返回的tensor是tensorflow::Tensor实例.<br>
TensorFlow 的Tensor表明了它的数据结构，而Flow则直观地表现出张量之间通过计算相互转化的过程ＴensorFlow中的每一个计算都是图上的一个节点，而节点之间的边描述了计算之间的依赖关系，a,b为常量，不依赖任何计算，而add计算则依赖读取两个常量的取值。

### GPU&CPU
在实现上, TensorFlow将图形定义转换成**分布式执行**的操作。一般不需要显式指定使用CPU 还是GPU, TensorFlow 能自动检测。如果检测到GPU,TensorFlow 会尽可能地利用找到的第一个GPU 来执行操作。 如果机器上有超过一个可用的GPU, 除第一个外的其它GPU默认是不参与计算的。 为了让 TensorFlow 使用这些GPU, 你必须将 op 明确指派给它们执行.with...Device语句用来指派特定的CPU或GPU执行操作.
```
with tf.Session() as sess:
  with tf.device("/gpu:1"):
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmu
```
为了获取你的operations和Tensor被指派到哪个设备上运行, 用log_device_placement新建一个session, 并设置为True.
```
Sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
```

### Variableb变量
当训练模型时，用变量来 存储和更新 参数。变量包含张量 (Tensor)存放于内存的缓存区。建模时它们需要被明确地 初始化，模型训练后它们必须被存储到磁盘。这些变量的值可在之后模型训练和分析是被加载。<br>
#### 创建变量
当创建一个变量时，你将一个张量作为初始值传入构造函数Variable()。 TensorFlow提供了一系列操作符来初始化张量，初始值是 常量 或是 随机值。在初始化时需要指定张量的shape，变量的shape通常是固定的，但TensorFlow提供了高级的机制来重新调整。
```
weights = tf.Variable(tf.random_normal([2,3], stddev = 2)， name="weights")
#这里生成 2×3 的矩阵，元素的均值为0，标准差为2（可用mean指定均值，默认值为0）。
```
随机数生成函数:
* tf.truncated_normal() 正太分布，偏离平均值超过两个偏差，重新选择
* tf.random_uniform（最小值，最大值） 平均分布 （控制字符中可加入种子， seed = n ）
常数生成函数：
* 全零数组 tf.zeros([2,3],int32)
* 全一数组 tf.ones()
* 全为给定的某数(例如9）：tf.fill([2,3],9)
* 产生给定值得常量 tf.cinstant([1,2,3])
除了tf.Variable函数， TensorFlow还提供了tf.get_variable函数来创建或获取变量。它们的区别在于，对于tf.Variable 函数，变量名称是个可选的参数，通过 name = ‘v’ 给出，但对于tf.get_variable函数，变量名称是必填的选项，它会首先根据变量名试图创建一个参数，如果遇到同名的参数就会报错。![知乎-tensor变量名和name属性的区别](https://www.zhihu.com/question/61426401/answer/189905912) & ![TensorFlow图变量tf.Variable的用法解析](http://blog.csdn.net/gg_18826075157/article/details/78368924)<br>
#### 变量初始化
在TensorFlow的世界里，变量的定义和初始化是分开的，所有关于图变量的赋值和计算都要通过tf.Session的run来进行。想要将所有图变量进行集体初始化时应该使用tf.global_variables_initializer。
有时候会需要用另一个变量的初始化值给当前变量初始化。 tf.initialize_all_variables()是并行地初始化所有变量。 使用其它变量的值初始化一个新的变量时，可以直接把已初始化的值作为新变量的初始值，或者把它当做tensor计算得到一个值赋予新变量。
```
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),name="weights")
w2 = tf.Variable(weights.initialized_value(), name="w2")
w_twice = tf.Variable(weights.initialized_value() * 0.2, name="w_twice")
```
#### 保存和恢复变量
用同一个Saver对象来恢复变量。当从文件中恢复变量时，不需要对它们做初始化。tf.train.Saver() 默认保存和加载计算图上定义的全部变量，如果只需加载模型的部分变量，例如可以使用tf.train.Saver([v1])构建，这样只有变量 v1 会被加载进来。如果要对变量名进行修改，可通过字典将模型保存时的变量名和需要加载的变量联系起来。

#### Xaiver初始化方法
如果深度学习模型的权重初始化得太小，那信号将在每层间传递时逐渐缩小而难以产生作用；如果权重初始化得太大，那信号将在每层间传递时逐渐放大并导致发散和失效。Xaiver initialization由Xavier Glorot和Yoshua Bengio在2010年提出，Xavier让权重满足均值为0，其基本思想是使前向传播和反向传播时每一层的方差一致。![Xavier Initialization 的理解与推导（及实现）](http://blog.csdn.net/lanchunhui/article/details/70318941)

### Fetch & Feed
为了取回操作的输出内容, 可以在使用 Session 对象的 run() 调用执行图时, 传入一些 tensor, 来取回结果。当获取的多个 tensor 值时，是在 op 的一次运行中一起获得而不是逐个去获取的。<br>
feed使用一个tensor值临时替换一个操作的输出结果. 你可以提供feed数据作为run()调用的参数。feed只在调用它的方法内有效, 方法结束,feed就会消失。最常见的用例是将某些特殊的操作指定为“feed”操作, 标记的方法是使用tf.placeholder()为这些操作创建占位符。
对比如下两段程序：
```
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)
with tf.Session() as sess:
result = sess.run([mul, intermed])
print result
```
```
input1 = tf.placeholder(tf.float32) #占位符
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
print sess.run([output], feed_dict={input1:[7.], input2:[2.]})
```
### TensorFlow训练神经网络
三个步骤： 
* 定义神经网络的结构和前向传播的输出结果；
* 定义损失函数以及选择反向传播优化的算法；
* 生成会话并且在训练数据上反复运行反向传播优化算法。
#### 激活函数
线性模型的最大特点是任意线性模型的组合仍然是线性模型，只通过线性变换，任意层的全连接神经网络和单层神经网络没有任何区别，因此非线性是深度学习的重要特性。目前TensorFlow提供了7种不同的非线性激活函数，常见的有：tf.nn.relu 、tf. sigmoid和tf.tanh。当遇到多分类问题时，我们可以使用softmax函数。当类别数k = 2时，softmax回归退化为logistic回归。这表明 softmax 回归是logistic回归的一般形式。当所分的类别互斥时，更适于选择 softmax回归分类器，当不互斥时，建立多个独立的logistic回归分类器更加合适。
#### 代价函数
神经网络模型的效果和优化的目标是通过代价函数（lost function）， 也称损失函数来定义的。训练神经网络通过反向传播代价，以减少代价为导向，调整参数。参数主要有：神经元之间的连接权重w，以及每个神经元的偏置b。调参的方式是采用梯度下降算法（ Gradient descent），沿着梯度方向调整参数大小。训练神经网络通过反向传播代价，以减少代价为导向，调整参数。参数主要有：神经元之间的连接权重w，以及每个神经元的偏置b。调参的方式是采用梯度下降算法（Gradient descent），沿着梯度方向调整参数大小。<br>
sigmoid函数（还有tanh）有很多优点，非常适合用来做激活函数, 要解决这个问题，就可以更改代价函数，使用交叉熵（cross entropy）代价函数。交叉熵代价函数同样有两个性质：
* 非负性，所以我们的目标就是最小化代价函数；
* 当真实输出a与期望输出y接近的时候，代价函数接近于0；
这样，当误差大的时候，权重更新就快，当误差小的时候，权重的更新就慢。这是很好的性质。因为交叉熵一般会与 softmax 函数一起使用， 所以TensorFlow 对这两个功能进行了 统一封装，并提供了tf.nn.softmax_cross_ebtropy_with_logits函数。TensorFlow 不仅支持经典的代价函数，还可以优化任意的自定义代价函数。
#### 梯度下降
神经网络的优化算法可以分为两个阶段，第一阶段先通过前向传播算法计算得到预测值，并将预测值和真实值做对比，得出两者之间的差距。然后在第二阶段通过反向传播算法计算损失函数对每一个参数的梯度，再根据梯度和学习率使用 梯度下降算法 更新每一个参数。<br>
只有当代价函数为凸函数时，梯度下降算法才能保证被优化的函数达到全局最优解。除此之外，梯度下降算法另一个问题就是计算时间过长，根据统计，如果一个操作需要 n次 浮点运算，那么计算这个梯度则需要三倍计算量。在海量训练数据下，要计算所有训练数据的损失函数是非常耗时间的，为了加速训练过程，可以使用随机梯度下降算法（Stochastic Gradient Descent, SGD） , 它优化的并不是在全部训练数据上的损失函数，而是在每一轮迭代中，随机优化某一条训练数据上的损失函数。这样每一轮参数更新的速度大大加快，也不会陷入局部最优。但某一条数据上损失函数更小并不代表在全部数据上损失函数更小，因此，使用随机梯度下降优化得到的神经网络需要更多的步数，有时甚至很难达到局部最优。<br>
为了综合梯度下降算法和随机梯度下降算法的优缺点，在实际应用中一般采取两个算法的折中，每一计算一小部分训练数据（称为一个batch）的损失函数，也称 mini-batch梯度下降，通常batch的取值为 2 到 100。在随机梯度下降中，还可以引入 冲量（ Momentum） ，将上一步的梯度方向保留一部分在下一步中，它的思想是，若当前梯度方向与上一次相同，那么，此次的速度增强，否则，应该相应减弱。 通常保留的系数在初始学习稳定之前，取0.5，之后取0.9或更大。<br>

#### 学习率
神经网络的学习率决定了参数每次更新的幅度，如果学习率过小，需要更多轮的迭代才能达到一个比较理想的优化效果，如果过大，可能导致参数在极优值的两侧来回移动。 TensorFlow提供了一种灵活的学习率设置方法—— 指数衰减法。
```
tf.train.exponential_decay（ learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None）
```
通过这个函数，可以先使用较大的学习率来快速得到一个比较优的解，然后随着迭代的继续逐步减小学习率。它实现了以下代码的功能：
```
decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
```
其中 learning_rate 为事先设定的初始学习率， decay_rate 为衰减系数（取值0到1）， decay_steps 为衰减速度 ， global_step 为全局步数，系统会自动更新这个参数的值，从1开始，每迭代一次加1。这样随着迭代步数的增加， decayed_learning_rate 就会逐渐减小。decay_steps 通常设置为完整训练一个
batch 所需要的迭代轮数，这样一个batch中的每个数据对模型训练有相等的作用，而每完整的训练一个batch，学习率就减小一次。

#### 正则化







