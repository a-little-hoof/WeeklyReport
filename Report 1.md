# Report 1

> 2023.2.20-2023.2.28

## A Neural Attention Model for Abstractive Sentence Summarization

+ [paper](https://arxiv.org/abs/1509.00685)

+ Attention-based model that generates abstractive summarization <img title="" src="file:///E:/github/WeeklyReport/pic1.1.jpg" alt="" width="236" data-align="center">

+ decoder: NNLM, encoder: attention-based, beam search to generate summarization

+ NLL($\theta$) = $-\sum_{j=1}^J$log$p$($y^j|x^j;\theta$) = $-\sum_{j=1}^J\sum_{i=1}^{N-1}$log$p(y_{i+1}^j|x^j,y_c;\theta)$

+ 本文将 seq2seq用于文本摘要技术。具体来说就是利用上图模型算出$p(y_{i+1}|y_c,x;\theta)$,然后用机器学习将NLL最大化。生成summary是采取beam search，即每次遍历整个词典找概率最大的k个词形成summary。此方法和其他模型比可以准确找到关键词，但词的正确顺序难以保证。

## Attention Is All You Need

+ [paper](https://www.bilibili.com/video/BV1pu411o7BE/?t=202&vd_source=f4a9d519cd04acb5ea66d4bc6a270f56) & [code](http://nlp.seas.harvard.edu/annotated-transformer/)

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
