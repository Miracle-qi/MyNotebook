
# Python编程实践笔记
---
> copyright@Qi Shuhao
1. 编码问题：decode，encode,utf-8
  * decode的作用是将其他编码的字符串转换成unicode编码
  * encode的作用是将unicode编码转换成其他编码的字符串
  * Unicode 是「字符集」,UTF-8 是「编码规则」。广义的 Unicode 是一个标准，定义了一个字符集以及一系列的编码规则，即 Unicode 字符集和 UTF-8、UTF-16、UTF-32 等等编码。
2. 遍历文件夹及多层子文件夹下所有数据文件的问题
3. lambda称为匿名函数，和普通的函数相比，就是省去了函数名称而已，同时这样的匿名函数，又不能共享在别的地方调用。
4. **ValueError: Masked arrays must be 1-D** 在用scatter绘制散点图时需要用.tolist()先把矩阵转换成list
5.  在python中，普通的列表list和numpy中的数组array是不一样的，最大的不同是：一个列表中可以存放不同类型的数据，包括int、float和str，甚至布尔型；而一个数组中存放的数据类型必须全部相同，int或float。也正因为列表可以存放不同类型的数据，因此列表中每个元素的大小可以相同，也可以不同，也就不支持一次性读取一列，即使是对于标准的二维数字列表。**所以列表不支持列读取。**
6. [nose](http://www.cnblogs.com/liaofeifight/p/5148717.html) python代码的自动测试工具
7. [Argparse](https://www.cnblogs.com/jianboqi/archive/2013/01/10/2854726.html)命令行解析工具
8. **virtualenv**是一个创建隔绝的Python环境的工具。virtualenv创建一个包含所有必要的可执行文件的文件夹，用来使用Python工程所需的包。
9. [Ros与Vrep平台搭建](http://www.cnblogs.com/zhuxuekui/p/5662159.html)
10. assert 断言语句
11. matplotlib.animation 实时图像
12. 一般来说，要使用某个类的方法，需要先实例化一个对象再调用方法。而使用@staticmethod或@classmethod，就可以不需要实例化，直接类名.方法名()来调用。
这有利于组织代码，把某些应该属于某个类的函数给放到那个类里去，同时有利于命名空间的整洁。<br>
  既然@staticmethod和@classmethod都可以直接类名.方法名()来调用，那他们有什么区别呢从它们的使用上来看,@staticmethod不需要表示自身对象的self和自身类的cls参数，就跟使用函数一样。@classmethod也不需要self参数，但第一个参数需要是表示自身类的cls参数。
13. 字符串强制格式化s.format()
14. [os.path](http://www.cnblogs.com/dkblog/archive/2011/03/25/1995537.html)
```
os.walk(top, topdown=True, onerror=None, followlinks=False) # 返回(root, dirs, files)
```
15. [python socket编程详细介绍](https://blog.csdn.net/rebelqsp/article/details/22109925)
