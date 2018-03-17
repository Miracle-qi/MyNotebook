# TensorFlow函数
1. **tf.transpose()**交换输入张量的不同维度，二维时相当于转置；
2. **ops.reset_default_graph()**tensorflow 在生产环境下，需要将 default graph 重新初始化，以保证内存中没有其他的 Graph，或者说我们需要在每个session之后清理相应的 Graph；
3. **tf.set_random_seed()**设置图级随机seed。依赖于随机seed的操作实际上从两个seed中获取：图级和操作级seed。 这将设置图级别的seed。其与操作级seed的相互作用如下：
* 如果没有设置图形级别和操作seed，则使用随机seed进行操作。
* 如果设置了图级seed，但操作seed没有设置：系统确定性地选择与图级seed一起的操作seed，以便获得唯一的随机序列。
* 如果没有设置图级seed，但是设置了操作seed：使用默认的图级seed和指定的操作seed来确定随机序列。
* 如果图级和操作seed都被设置：两个seed联合使用以确定随机序列。
4. 
