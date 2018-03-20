# Reinforcement Learning学习笔记
> Copyright © Qi Shuhao. All Rights Reserved<br>
> 参考[莫烦-python](https://morvanzhou.github.io/)
----
## 基础概念
1. 强化学习是机器学习大家族中的一大类, 使用强化学习能够让机器学着如何在环境中拿到高分, 表现出优秀的成绩. 而这些成绩背后却是他所付出的辛苦劳动, 不断的试错, 不断地尝试, 累积经验, 学习经验.
2. 强化学习是一类算法, 是让计算机实现从一开始什么都不懂, 脑袋里没有一点想法, 通过不断地尝试, 从错误中学习, 最后找到规律, 学会了达到目的的方法. 原来计算机也需要一位虚拟的老师, 这个老师比较吝啬, 他不会告诉你如何移动, 如何做决定, 他为你做的事只有给你的行为打分, 那我们应该以什么形式学习这些现有的资源, 或者说怎么样只从分数中学习到我应该怎样做决定呢? 很简单, 我只需要记住那些高分, 低分对应的行为, 下次用同样的行为拿高分, 并避免低分的行为.
3. RL经典算法：
* 通过价值选行为：Q-learning、Sarsa、 Deep Q-Network
* 直接选行为：Policy Gradients
* 想象环境并从中学习：Model based RL
4. ** Model-free和Model-based ** Model-free 的方法有很多, 像 Q learning, Sarsa, Policy Gradients 都是从环境中得到反馈然后从中学习. 而 model-based RL 只是多了一道程序, 为真实世界建模, 也可以说他们都是 model-free 的强化学习, 只是 model-based 多出了一个虚拟环境, 我们不仅可以像 model-free 那样在现实中玩耍,还能在游戏中玩耍, 而玩耍的方式也都是 model-free 中那些玩耍方式, 最终 model-based 还有一个杀手锏是 model-free 超级羡慕的. 那就是想象力.
5. ** 基于概率和基于价值 ** 基于概率是强化学习中最直接的一种, 他能通过感官分析所处的环境, 直接输出下一步要采取的各种动作的概率, 然后根据概率采取行动, 所以每种动作都有可能被选中, 只是可能性不同. 而基于价值的方法输出则是所有动作的价值, 我们会根据最高价值来选着动作, 相比基于概率的方法, 基于价值的决策部分更为铁定, 毫不留情, 就选价值最高的, 而基于概率的, 即使某个动作的概率最高, 但是还是不一定会选到他.<br>比如在基于概率这边, 有 Policy Gradients, 在基于价值这边有 Q learning, Sarsa 等. 而且我们还能结合这两类方法的优势之处, 创造更牛逼的一种方法, 叫做 Actor-Critic, actor 会基于概率做出动作, 而 critic 会对做出的动作给出动作的价值, 这样就在原有的 policy gradients 上加速了学习过程.
6. ** 回合更新和单步更新 ** Monte-carlo learning 和基础版的 policy gradients 等 都是回合更新制, Qlearning, Sarsa, 升级版的 policy gradients 等都是单步更新制. 
7. ** On-Policy和Off-Policy** 这个视频的最后一种分类方式是 在线学习和离线学习, 所谓在线学习, 就是指我必须本人在场, 并且一定是本人边玩边学习, 而离线学习是你可以选择自己玩, 也可以选择看着别人玩, 通过看别人玩来学习别人的行为准则, 离线学习 同样是从过往的经验中学习, 但是这些过往的经历没必要是自己的经历, 任何人的经历都能被学习. 或者我也不必要边玩边学习, 我可以白天先存储下来玩耍时的记忆, 然后晚上通过离线学习来学习白天的记忆.<br>
最典型的在线学习就是 Sarsa 了, 还有一种优化 Sarsa 的算法, 叫做 Sarsa lambda, 最典型的离线学习就是 Q learning, 后来人也根据离线学习的属性, 开发了更强大的算法, 比如让计算机学会玩电动的 Deep-Q-Network.
## Q-Learning
8. ![Q-Learning 更新](https://github.com/Miracle-qi/MyNotebook/raw/master/p-3.png)<br>
有了现实和估计值, 我们就能更新Q(s1, a2) , 根据 估计与现实的差距, 将这个差距乘以一个学习效率 alpha 累加上老的 Q(s1, a2) 的值 变成新的值. 但时刻记住, 我们虽然用 maxQ(s2) 估算了一下 s2 状态, 但还没有在 s2 做出任何的行为, s2 的行为决策要等到更新完了以后再重新另外做. 这就是 off-policy 的 Q learning 是如何决策和学习优化决策的过程.<br>
![Q-Learning 整体算法](https://github.com/Miracle-qi/MyNotebook/raw/master/p-2.png)<br>
这一张图概括了我们之前所有的内容. 这也是 Q learning 的算法, 每次更新我们都用到了 Q 现实和 Q 估计, 而且 Q learning 的迷人之处就是 在 Q(s1, a2) 现实 中, 也包含了一个 Q(s2) 的最大估计值, 将对下一步的衰减的最大估计和当前所得到的奖励当成这一步的现实, 很奇妙吧.Q(s)中的gamma从0变到1, 相当于眼镜的度数由浅变深, 对远处的价值看得越清楚, 所以机器人渐渐变得有远见, 不仅仅只看眼前的利益, 也为自己的未来着想.<br>
Q-Learning 算法就是一直不断更新 Q table 里的值, 然后再根据新的值来判断要在某个 state 采取怎样的 action. Qlearning 是一个 off-policy 的算法, 因为里面的 max action 让 Q table 的更新可以不基于正在经历的经验(可以是现在学习着很久以前的经验,甚至是学习他人的经验). 当然 Qlearning 可以用在 on-policy 上, 也就是现学现卖, 将现在经历的直接当场学习并运用. 
### Sarsa
9. Sarsa 的决策部分和 Q learning 一模一样, 因为我们使用的是 Q 表的形式决策, 所以我们会在 Q 表中挑选值较大的动作值施加在环境中来换取奖惩. 但是不同的地方在于 Sarsa 的更新方式是不一样的.<br>
如果你用的是 Q learning, 你会观看一下在 s2 上选取哪一个动作会带来最大的奖励, 但是在真正要做决定时, 却不一定会选取到那个带来最大奖励的动作, Q-learning 在这一步只是估计了一下接下来的动作值. 而 Sarsa 是实践派, 他说到做到, 在 s2 这一步估算的动作也是接下来要做的动作. 所以 Q(s1, a2) 现实的计算值, 我们也会稍稍改动, 去掉maxQ, 取而代之的是在 s2 上我们实实在在选取的 a2 的 Q 值. 最后像 Q learning 一样, 求出现实和估计的差距 并更新 Q 表里的 Q(s1, a2).<br>
![Q-Learning 和 Sarsa 对比](https://github.com/Miracle-qi/MyNotebook/raw/master/p-4.png)<br>
从算法来看, 这就是他们两最大的不同之处了. 因为 Sarsa 是说到做到型, 所以我们也叫他 on-policy, 在线学习, 学着自己在做的事情. 而 Q learning 是说到但并不一定做到, 所以它也叫作 Off-policy, 离线学习. 而因为有了 maxQ, Q-learning 也是一个特别勇敢的算法.<br>
 Sarsa 是一种单步更新法, 在环境中每走一步, 更新一次自己的行为准则, 我们可以在这样的 Sarsa 后面打一个括号, 说他是 Sarsa(0), 因为他等走完这一步以后直接更新行为准则. 如果延续这种想法, 走完这步, 再走一步, 然后再更新, 我们可以叫他 Sarsa(1). 同理, 如果等待回合完毕我们一次性再更新呢, 比如这回合我们走了 n 步, 那我们就叫 Sarsa(n). 为了统一这样的流程, 我们就有了一个 lambda 值来代替我们想要选择的步数, 这也就是 **Sarsa(lambda)** 的由来. <br>
 **单步更新 & 回合更新** 虽然我们每一步都在更新, 但是在没有获取宝藏的时候, 我们现在站着的这一步也没有得到任何更新, 也就是直到获取宝藏时, 我们才为获取到宝藏的上一步更新为: 这一步很好, 和获取宝藏是有关联的, 而之前为了获取宝藏所走的所有步都被认为和获取宝藏没关系. 回合更新虽然我要等到这回合结束, 才开始对本回合所经历的所有步都添加更新, 但是这所有的步都是和宝藏有关系的, 都是为了得到宝藏需要学习的步, 所以每一个脚印在下回合被选则的几率又高了一些. 在这种角度来看, 回合更新似乎会有效率一些.
 其实 lambda 就是一个**衰变值**, 他可以让你知道离奖励越远的步可能并不是让你最快拿到奖励的步, 所以我们想象我们站在宝藏的位置, 回头看看我们走过的寻宝之路, 离宝藏越近的脚印越看得清, 远处的脚印太渺小, 我们都很难看清, 那我们就索性记下离宝藏越近的脚印越重要, 越需要被好好的更新. 和之前我们提到过的 奖励衰减值 gamma 一样, lambda 是脚步衰减值, 都是一个在 0 和 1 之间的数.当 lambda 取0, 就变成了 Sarsa 的单步更新, 当 lambda 取 1, 就变成了回合更新, 对所有步更新的力度都是一样. 
 
## Deep Q Network
10. 我们使用表格来存储每一个状态 state, 和在这个 state 每个行为 action 所拥有的 Q 值. 而当今问题是在太复杂, 状态可以多到比天上的星星还多(比如下围棋). 如果全用表格来存储它们, 恐怕我们的计算机有再大的内存都不够, 而且每次在这么大的表格中搜索对应的状态也是一件很耗时的事. 不过, 在机器学习中, 有一种方法对这种事情很在行, 那就是**神经网络**. 我们可以将状态和动作当成神经网络的输入, 然后经过神经网络分析后得到动作的 Q 值, 这样我们就没必要在表格中记录 Q 值, 而是直接使用神经网络生成 Q 值. <br>
 我们知道, 神经网络是要被训练才能预测出准确的值. 那在强化学习中, 神经网络是如何被训练的呢? 首先, 我们需要 a1, a2 正确的Q值, 这个 Q 值我们就用之前在 Q learning 中的 Q 现实来代替. 同样我们还需要一个 Q 估计 来实现神经网络的更新. 所以神经网络的的参数就是老的 NN 参数 加学习率 alpha 乘以 Q 现实 和 Q 估计 的差距. 我们整理一下.<br>
 ![Q-Learning 整体算法](https://github.com/Miracle-qi/MyNotebook/raw/master/p-5.png)<br>
 我们通过 NN 预测出Q(s2, a1) 和 Q(s2,a2) 的值, 这就是 Q 估计. 然后我们选取 Q 估计中最大值的动作来换取环境中的奖励 reward. 而 Q 现实中也包含从神经网络分析出来的两个 Q 估计值, 不过这个 Q 估计是针对于下一步在 s’ 的估计. 最后再通过刚刚所说的算法更新神经网络中的参数. 但是这并不是 DQN 会玩电动的根本原因. 还有两大因素支撑着 DQN 使得它变得无比强大. 这两大因素就是 **Experience replay 和 Fixed Q-targets**.<br>
  ![Q-Learning 整体算法](https://github.com/Miracle-qi/MyNotebook/raw/master/p-6.png)<br>
简单来说, DQN 有一个记忆库用于学习之前的经历. 在之前提到的, Q learning 是一种 off-policy 离线学习法, 它能学习当前经历着的, 也能学习过去经历过的, 甚至是学习别人的经历. 所以每次 DQN 更新的时候, 我们都可以随机抽取一些之前的经历进行学习. 随机抽取这种做法打乱了经历之间的相关性, 也使得神经网络更新更有效率. Fixed Q-targets 也是一种打乱相关性的机理, 如果使用 fixed Q-targets, 我们就会在 DQN 中使用到两个结构相同但参数不同的神经网络, 预测 Q 估计 的神经网络具备最新的参数, 而预测 Q 现实 的神经网络使用的参数则是很久以前的. 有了这两种提升手段, DQN 才能在一些游戏中超越人类.<br>
  <div style="align: center"> 
  <img src="https://github.com/Miracle-qi/MyNotebook/raw/master/p-7.jpg"/> 
  </div>
11. OpenAI gym 是一个提供了很多优秀的模拟环境的模块。
12. **Double DQN 算法**<br>
* 有两个神经网络: Q_eval (Q估计中的), Q_next (Q现实中的).
* 原本的 Q_next = max(Q_next(s', a_all))，而Double DQN 中的Q_next = Q_next(s', argmax(Q_eval(s', a_all))). 
* Double DQN 的 Q值 相对于 Natural DQN 就消除了一些 overestimate.
13. **Prioritized Experience Replay (DQN)**<br>
这一套算法重点就在我们 batch 抽样的时候并不是随机抽样, 而是按照 Memory 中的**样本优先级**来抽. 所以这能更有效地找到我们需要学习的样本.<br>
那么样本的优先级是怎么定的呢? 原来我们可以用到 TD-error, 也就是 Q现实 - Q估计 来规定优先学习的程度. 如果 TD-error 越大, 就代表我们的预测精度还有很多上升空间, 那么这个样本就越需要被学习, 也就是优先级 p 越高.<br>
有了 TD-error 就有了优先级 p, 那我们如何有效地根据 p 来抽样呢? 如果每次抽样都需要针对 p 对所有样本排序, 这将会是一件非常消耗计算能力的事. 好在我们还有其他方法, 这种方法不会对得到的样本进行排序. 这就是这篇 [paper](https://arxiv.org/abs/1511.05952) 中提到的 SumTree.<br>
SumTree 是一种树形结构, 每片树叶存储每个样本的优先级 p, 每个树枝节点只有两个分叉, 节点的值是两个分叉的和, 所以 SumTree 的顶端就是所有 p 的和. <br>
  <div style="align: center"> 
  <img src="https://morvanzhou.github.io/static/results/reinforcement-learning/4-6-2.png"/> 
  </div><br>
抽样时, 我们会将 p 的总合 除以 batch size, 分成 batch size 那么多区间, (n=sum(p)/batch_size). 如果将所有 node 的 priority 加起来是42的话, 我们如果抽6个样本, 这时的区间拥有的 priority 可能是这样.<br>
[0-7], [7-14], [14-21], [21-28], [28-35], [35-42]<br>
然后在每个区间里随机选取一个数. 比如在第区间 [21-28] 里选到了24, 就按照这个 24 从最顶上的42开始向下搜索. 首先看到最顶上 42 下面有两个 child nodes, 拿着手中的24对比左边的 child 29, 如果 左边的 child 比自己手中的值大, 那我们就走左边这条路, 接着再对比 29 下面的左边那个点 13, 这时, 手中的 24 比 13 大, 那我们就走右边的路, 并且将手中的值根据 13 修改一下, 变成 24-13 = 11. 接着拿着 11 和 13 左下角的 12 比, 结果 12 比 11 大, 那我们就选 12 当做这次选到的 priority, 并且也选择 12 对应的数据.<br>
Prioritized replay 可以高效帮助 agent利用这些不常拿到的奖励, 并好好学习他们. 
14. **Dueling DQN**<br>
只要稍稍修改 DQN 中神经网络的结构, 就能大幅提升学习效果, 加速收敛. 这种新方法叫做 Dueling DQN. 用一句话来概括 Dueling DQN 就是. 它将每个动作的 Q 拆分成了 state 的 Value 加上 每个动作的 Advantage.<br>
下面这个公式解释了不同之处. 原来 DQN 神经网络直接输出的是每种动作的 Q值, 而 Dueling DQN 每个动作的 Q值 是有下面的公式确定的.
[公式](https://morvanzhou.github.io/static/results/reinforcement-learning/4-7-2.png)
它分成了这个 state 的值, 加上每个动作在这个 state 上的 advantage. 因为有时候在某种 state, 无论做什么动作, 对下一个 state 都没有多大影响.
我们发现当可用动作越高, 学习难度就越大, 不过 Dueling DQN 还是会比 Natural DQN 学习得更快. 收敛效果更好.
##  Policy Gradients 
15. 强化学习是一个通过奖惩来学习正确行为的机制. 家族中有很多种不一样的成员, 有学习奖惩值, 根据自己认为的高价值选行为, 比如 Q learning, Deep Q Network, 也有不通过分析奖励值, 直接输出行为的方法, 这就是今天要说的 Policy Gradients 了. 甚至我们可以为 Policy Gradients 加上一个神经网络来输出预测的动作. 对比起以值为基础的方法, Policy Gradients 直接输出动作的最大好处就是, 它能在一个连续区间内挑选动作, 而基于值的, 比如 Q-learning, 它如果在无穷多的动作中计算价值, 从而选择行为, 这, 它可吃不消.<br>
Policy gradient 是 RL 中另外一个大家族, 他不像 Value-based 方法 (Q learning, Sarsa), 但他也要接受环境信息 (observation), 不同的是他要输出不是 action 的 value, 而是具体的那一个 action, 这样 policy gradient 就跳过了 value 这个阶段. 而且个人认为 Policy gradient 最大的一个优势是: 输出的这个 action 可以是一个连续的值, 之前我们说到的 value-based 方法输出的都是不连续的值, 然后再选择值最大的 action. 而 policy gradient 可以在一个连续分布上选取 action.
16. policy gradient 的一个经典算法是一种基于 整条回合数据 的更新, 也叫 **REINFORCE**方法. 这种方法是 policy gradient 的最基本方法。
log(Policy(s,a))\*V 中的 log(Policy(s,a)) 表示在 状态 s 对所选动作 a 的吃惊度, 如果 Policy(s,a) 概率越小, 反向的 log(Policy(s,a)) (即 -log(P)) 反而越大. 如果在 Policy(s,a) 很小的情况下, 拿到了一个 大的 R, 也就是 大的 V, 那 -log(Policy(s, a))\*V 就更大, 表示更吃惊, (我选了一个不常选的动作, 却发现原来它能得到了一个好的 reward, 那我就得对我这次的参数进行一个大幅修改). 这就是 log(Policy)\*V 的物理意义啦.
  <div style="align: center"> 
  <img src="https://github.com/Miracle-qi/MyNotebook/raw/master/p-8.png"/> 
  </div>
[Policy gradient methods for reinforcement learning with function approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
