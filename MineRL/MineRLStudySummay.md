# MineRL Study Summay



# PPO (Proximal Policy Optimization)

PPO （Proximal Policy Optimization）算法是一种改进过的policy gradient算法。通常的policy gradient算法对于stepsize选择非常敏感，如果stepsize太小，优化会很慢，而如果stepsize太大，信号可能会淹没在噪声中，比较难得到好的结果。

在PPO 算法中引入了K-L 散度作为loss function 中的一项作为约束来控制每次迭代中的policy变化程度。于是loss function的结构为：

![PPO result](.\res\ppo_loss.jpeg)

Reference: [[1]](https://openai.com/blog/openai-baselines-ppo/),  [[2]](https://blog.csdn.net/qq_30615903/article/details/86308045) ,[[3]](https://www.newbieyxy.top/2018/10/06/PPO-code-reading/) 

## PPO result

![PPO result](.\res\ppo_snapshot.png)