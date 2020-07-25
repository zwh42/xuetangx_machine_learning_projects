# CogQA 学习笔记



## 阅读笔记

1. QA 系统目前的主要挑战：

   + Reasoning ability 推理能力

   + Explainability 可解释性
   + Scalability 可扩展性

2. 将认知科学的理论模型(Dual process theory)应用在机器学习的研究中，想法很有新意。system1 用来提取知识生成cognitive graph，system2基于cognitive graph做出决策，同时引导system1更好的生成cognitive graph。

3. 具体实现上，使用了BERT 和GNN。

4. system1: 

   +  输入： 问题(question)，线索(clues)，片段(paragraph)

## 代码：

路径： https://github.com/zwh42/xuetangx_machine_learning_projects/tree/master/GraphSGAN/code

原始来源：https://github.com/THUDM/CogQA

