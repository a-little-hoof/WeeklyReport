# Report 4

> 2023.3.13-2023.3.19

## A Nueral Conversational Model

+ 第一次将seq2seq用于对话，2015年的工作

+ 结构：lstm+seq2seq

+ 好处是显而易见的，因为问题-回答刚好对应了input-output，不需要人工确定复杂的规则。坏处是生成的的文本lack coherent personality

## DIALOGPT: Large-Scale Generative Pre-training for Conversational Response Generation

+ 模型结构
  
  a 12-48 layer transformer with layer normalization.

+ Mutual Information Maximization
  
  最大化后向模型对枯燥的output进行惩罚

## Diversifying Dialogue Generation With Non-conversational Text

+ 现有问题：生成通用回复“好的”，“OK”。目前通过改变目标函数、用结构化信息、情感、个性来增强训练语料，但主题仍被限制且需要大量人工标注。

+ 本文创新点在于利用非聊天语料来丰富通用的conversation生成

+ baseline模型
  
  + 直接用收集的语料库中的句子作为回答。
  
  + language model和seq2seq的加权：$p_t(\omega) = \alpha S2S_t(\omega) + (1-\alpha)L_t(\omega)$ 
  
  + 在混合后的语料上同时训练一个seq2seq和language model，然后decoder在两个模型间共享参数。

+ 本文的iteractive back translation
  
  $E_{X_i,Y_i\sim D} - $log$P_f(X_i|Y_i) - $log$P_b(X_i|Y_i)$  (2)
  
  $E_{T_i\sim D_T} - $log$P_f(T_i|b(T_i))$                            (3)backward模型
  
  $E_{X_i\sim D_T} -$log$P_b(X_i|f(X_i))$                         (4)forward模型

<img title="" src="file:///E:/github/WeeklyReport/pic4.1.png" alt="" data-align="center" width="297">

+ conversation部分开始的模型都比较简单，基本上是从translation的方法类比而来，感觉这部分的论文可以读得快一些？

## 代码部分

+ 学习了colab的使用，计划从第五周开始做cs231的assignment
