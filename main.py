import collections
import re
import os
import collections
import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import os
import random
import tarfile
import random
import zipfile
import torch.utils.data as data
import os.path
import torchtext
import string
def read_txt(path):
    file=open(path,'r')
    lines = [re.sub('^[a-z]+', ' ', line.strip().lower()) for line in file]
    return lines

from torchtext.legacy.data import Field
from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Dataset, Example
import json

def fill_article(article,options,answers):
    str1=article
    str2=options
    str3=answers
    cnt=0
    for index in range(len(str1)):
        if(str1[index]=='_'):
            opt=str2[cnt]
            if(str3[cnt]=='A'):
                str1[index]=opt[0]
            elif(str3[cnt]=='B'):
                str1[index]=opt[1]
            elif(str3[cnt]=='C'):
                str1[index]=opt[2]
            elif(str3[cnt]=='D'):
                str1[index]=opt[3]
            cnt+=1
    return str1





#定义field，对输入文本的格式进行预设置
#这里为了测试，只做了文章、选项和答案的读入
article_field = Field(sequential=True, tokenize='spacy',tokenizer_language='en_core_web_sm',
                        lower=False, batch_first=True, include_lengths=True,use_vocab=True)
option_field=Field(sequential=True, tokenize='spacy',tokenizer_language='en_core_web_sm',
                        lower=False, batch_first=True, include_lengths=True)
answer_field = Field(sequential=False, use_vocab=False)

label_field = Field(sequential=False, use_vocab=False,fix_length=20,pad_token=0)

#从Dataset类继承出MyDataset类
class MyDataset(Dataset):

    def __init__(self, data_list, article_field, option_field,answer_field):

        #fields = [('article', article_field),('option',option_field), ('answer', answer_field)]
        fields=[('article', article_field),('label',label_field)]
        examples = []
        #数据目录,为了测试，用了测试集中的部分数据
        Data_Path = '.data/CLOTH/test/high/'
        #移动操作路径至数据目录下
        os.chdir(Data_Path)
        #data_list记录了数据集中各个文件的名称
        fl = open('data_list.txt', 'r')
        #声明为列表
        self.article_list = []
        self.answer_list = []
        self.option_list= []
        #因为目前填入的全是正确选项，label全部做成了1
        self.label_list=[]
        #依次读入各个json文件中的内容
        #for i in range(478):
        for i in range(2):
            #读取数据文件名
            file_name = fl.read(13)
            dot = fl.read(1)
            data_path = file_name
            #lines = [re.sub('^[a-z]+', ' ', line.strip().lower()) for line in f]
            with open(data_path, 'r') as f_json:
                file_content = json.load(f_json)
                #str1=[re.sub('[a-z]+', ' ', line.strip().lower()) for line in file_content['article']]
                punctuation_string = string.punctuation
                #先存在str中才能用list的append方法添加文本
                article_str=file_content['article']
                options_str=file_content['options']
                answers_str=file_content['answers']
                label_str=[]
                #print(type(article_str))


                cnt=0
                for index in range(len(article_str)):
                    #print(article_str[index])

                    if(article_str[index]=='_'):
                        #label_str.append(1)
                        if(answers_str[cnt]=='A'):
                            article_str=article_str.replace('_',options_str[cnt][0],1)
                        elif (answers_str[cnt] == 'B'):
                            article_str = article_str.replace('_', options_str[cnt][1], 1)
                        elif (answers_str[cnt] == 'C'):
                            article_str = article_str.replace('_', options_str[cnt][2], 1)
                        elif (answers_str[cnt] == 'D'):
                            article_str = article_str.replace('_', options_str[cnt][3], 1)
                        cnt+=1
                while cnt<20:
                    #label_str.append(0)
                    cnt+=1
                label_str.append(1)
                #print(type(article_str))
                #print(options_str)
                #print(answers_str)
                #print(article_str)
                #将读入的数据插入到列表中

                self.label_list.append(label_str)
                self.article_list.append((article_str))

                #self.option_list.append((options_str))
                #self.answer_list.append((answers_str))
                #如下写法会报错
               # self.artile_list+=file_content['article']
                #self.answer_list+=file_content['answers']
               # self.option_list+=file_content['options']

        #print('art',len(self.article_list))
        #print('label',len(self.label_list))
        for index in range(len(self.article_list)):
            #把列表的数据汇总到example中
            #examples.append(Example.fromlist([self.article_list[index],
             #                                 self.option_list[index],self.answer_list[index]], fields))
            examples.append(Example.fromlist([(self.article_list[index]),
                                             self.label_list[index]],fields))

        super().__init__(examples, fields)



    def __len__(self):
        #返回数据集的长度
        return len(self.article_list)

    @staticmethod
    def sort_key(input):
        return len(input.article)

#声明dataset对象，读取相应的文件；此后可以从dataset中直接调用相关数据
dataset=MyDataset(data_list='data_list.txt',article_field=article_field,option_field=option_field,answer_field=answer_field)

print(dataset.__len__())
print(dataset[0])

#把程序工作目录调回源目录
os.chdir('../../../..')

import torchtext.vocab
import torchtext.legacy.datasets
min_freq=0.01#最小词频
#词向量文件名称
vector_path='glove.6B.50d.txt'
#词向量缓存目录
cache = '.vector'
#输出当前的工作目录
print(os.getcwd())
#如果缓存目录不存在，就创建
if not os.path.exists(cache):
    #print("!")
    os.mkdir(cache)
#创建词典
vectors = torchtext.vocab.Vectors(name=vector_path, cache=cache)
article_field.build_vocab(dataset, min_freq=min_freq, vectors=vectors)

#用split方法划分训练集和开发集
#因为我们的数据集已经做了划分，实际操作的时候不一定需要这么做
train_data,valid_data=dataset.split()
#输出各个训练集和开发集大小
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
#输出由article_field生成的词典大小
print(f"Unique tokens in article_field vocabulary: {len(article_field.vocab)}")
#输出第一行试试
#print(vars(train_data.examples[0])['article'])
#测试集
text_sample = " "
text_sample = text_sample.join(vars(train_data.examples[0])['article'])

#输出词典中最高频的100个词
#如何去除空格&标点？
print(article_field.vocab.freqs.most_common(100))
from torch.utils.data import DataLoader

from torchtext.legacy import data
import torch
#创建迭代器，供训练使用
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator= data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size = BATCH_SIZE,
    device = device)



#for batch in train_iterator:
 #   print(len(batch))


# -*- coding: utf-8 -*-
"""bilstm_test

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QXJAfBHo4-TXeugM9FiKaxf9Yy1FkxV9

**构建lstm模型**

输出的是一个1维的向量，即填入选项后的句子的得分（概率）
训练时，分别将四个选项填入句子，将计算得出的概率与label进行CE

正确选项的label为1，错误选项的label为0

这里用了dropout做正则化，以防止过拟合
"""

import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        # output_dim为1
        super(LSTM, self).__init__()

        # 词嵌入函数，后续调用glove的接口进行嵌入
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        # 双向lstm
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=2, bidirectional=True)

        # 有2层，所以fc层的输入维度为hidden_dim的两倍
        self.fc = nn.Linear(hidden_dim*2 , output_dim)

        # 每个神经元有50%的可能性丢弃;参数可改
        self.dropout = nn.Dropout(0.5)

    # 下面是前向传播部分
    def forward(self, article):
        # 进行词嵌入,后续用glove直接代替

        #article=article[0]
        article,_=article
        article.transpose_(0,1)
        print(article.shape)
        embedded = self.dropout(self.embedding(article))
        print(embedded.shape)
        # 计算出结果
        output, (hidden, cell) = self.lstm(embedded)

        # 把正向和反向的hidden层连接起来
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        print(hidden.shape)
        # dropout
        hidden = self.dropout(hidden)
        print(hidden.shape)
        # 过一个线性层
        fc = self.fc(hidden)
        print(fc.shape)
        # 用softmax做归一化
        ##return nn.Softmax(fc)
        #print(fc)
        #fc=torch.tanh(fc)
        #print(fc)
        return fc


"""**下面声明lstm模型**

"""

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# input的维度为词典的大小，此处暂定为10000
INPUT_DIM = len(article_field.vocab)

# 这里直接抄了助教的参数
EMBEDDING_DIM = 100
HIDDEN_DIM = 256

OUTPUT_DIM = 2
# 定义模型
model = LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

"""**查看模型中可训练的参数数量**

还是抄学姐的 d-:
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

"""**计算精确度所需函数**"""


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


"""**下面定义train模块**"""

import torch
import torch.optim as optim

# 优化器 目前采用SGD
optimizer = optim.SGD(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
#criterion=nn.MSELoss()
model = model.to(device)
criterion = criterion.to(device)


def train(model, iterator, optimizer, criterion):
    # 返回值是一个两个元素的列表,分别是平均准确率和平均loss
    print("train begin")
    epoch_loss = 0  # 这一波训练的loss
    epoch_acc = 0
    model.train()
    #for batch in enumerate(iterator):
    for batch in iterator:
        # 调用lstm，输入text，返回预测值
        # 具体写法根据数据集处理后提供的接口而定
        article,_=batch.article
        print('art',article.shape)
        pred = model(batch.article)
        #print("after",pred)
        #pred=pred.squeeze(1)
        #print("before", pred)
        optimizer.zero_grad()

        # 计算acc和loss
        # batch.label是正确答案对应的标签
        #label=batch.label.squeeze(1).long()
        label=torch.ones(batch.batch_size).long()
        #print('shape',pred.shape)
        print("pred shape",pred.shape)
        print("label shape",label.shape)
        print("pred",pred)
        print("label",label)
        loss = criterion(pred, label)

        #label  1*batch_size
        #pred batch_size*

        #acc = binary_accuracy(pred, label)


        epoch_loss += loss.item()
        #epoch_acc += acc.item()
        loss.backward()
        print("loser")
        optimizer.step()
        print("yes")
    return 1 / len(iterator), epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.article)
            label=batch.label.squeeze(1)
            print('eval_pred shape',predictions.shape)
            loss = criterion(predictions, label)

            acc = binary_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model=model, iterator=train_iterator, optimizer=optimizer, criterion=criterion)
    valid_loss, valid_acc = evaluate(model=model, iterator=valid_iterator, criterion=criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')
    print("!!!!")
    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
