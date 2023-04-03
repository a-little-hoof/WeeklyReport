# Report 1

> 2023.2.20-2023.2.26

## A Neural Attention Model for Abstractive Sentence Summarization

+ [paper](https://arxiv.org/abs/1509.00685)

+ Attention-based model that generates abstractive summarization <img title="" src="file:///E:/github/WeeklyReport/pic1.1.jpg" alt="" width="236" data-align="center">

+ decoder: NNLM, encoder: attention-based, beam search to generate summarization

+ NLL($\theta$) = $-\sum_{j=1}^J$log$p$($y^j|x^j;\theta$) = $-\sum_{j=1}^J\sum_{i=1}^{N-1}$log$p(y_{i+1}^j|x^j,y_c;\theta)$

+ 本文将 seq2seq用于文本摘要技术。具体来说就是利用上图模型算出$p(y_{i+1}|y_c,x;\theta)$,然后用机器学习将NLL最大化。生成summary是采取beam search，即每次遍历整个词典找概率最大的k个词形成summary。此方法和其他模型比可以准确找到关键词，但词的正确顺序难以保证。代码不是用python写的，没看懂。

## Attention Is All You Need

+ [paper](https://www.bilibili.com/video/BV1pu411o7BE/?t=202&vd_source=f4a9d519cd04acb5ea66d4bc6a270f56) & [code](http://nlp.seas.harvard.edu/annotated-transformer/) （代码实现下周看）

<img title="" src="file:///E:/github/WeeklyReport/pic1.2.png" alt="" data-align="center" width="207">

> 输入：n个长为d的向量。第一个注意力层三个输入：key, value, query，自注意力机制。第二个注意力层类似。第三个key, value来自encoder，query来自decoder上一个输出

Attention

+ Scaled Dot-Product Attention
  
  Attention(Q, K, V) = softmax($\frac{QK^T}{\sqrt{d_k}}$)V
  
  ![](E:\github\WeeklyReport\pic1.3.jpg)

+ Multi-Head attention
  
  MultiHead(Q, K, V) = Concat(head$_1$, ..., head$_n$)$W^O$
  
  where head$_i$ = Attention(Q$W_i^Q$, K$W_i^K$, V$W_i^V$)
  
  本质上是给h次机会将向量投影，投影矩阵W是用来学习的，下面三个维度均为$\mathbb{R}^{d_{model}\times d_k}$，上面为$\mathbb{R}^{hd_{v}\times d_{model}}$，$d_k$ = $d_v$ = $d_{model}/h$

<img title="" src="file:///E:/github/WeeklyReport/pic1.4.jpg" alt="" data-align="center" width="148">

+ Transformer大致结构

## Coding

```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,input):
        output=input+1
        print(output)

model = Model()         #实例化
input = torch.tensor(1) #输入为1
model(input)            #输出为2
```

+ super().**init**()调用父类的init

+ nn.Module的forwar函数在实例化的时候不需要被调用，即不需要model.forward(input)
