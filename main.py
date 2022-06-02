#把36~525行复制上去即可
#同时上传data_list
import re
import os
import random
#import torch.utils.data as data
import os.path
import string

from torchtext.legacy.data import Field
from torchtext.legacy.data import Dataset, Example
import json

print("ok")

from data_utils import MyDataset_test_1,MyDataset_train,MyDataset_valid,MyDataset_test
from data_utils import article_field,label_field
# 定义field，对输入文本的格式进行预设置



dataset_train = MyDataset_train(data_list='data_list.txt', article_field=article_field, label_field=label_field)

dataset_valid = MyDataset_valid(data_list='data_list.txt', article_field=article_field, label_field=label_field)
print(dataset_train.__len__())
print(dataset_train[0])
os.chdir('../../../..')

import torchtext.legacy.datasets

min_freq = 0.01  # 最小词频
# 词向量文件名称
vector_path = 'glove.6B.50d.txt'
# 词向量缓存目录
cache = '.vector'
# 输出当前的工作目录
print(os.getcwd())
# 如果缓存目录不存在，就创建
if not os.path.exists(cache):
    # print("!")
    os.mkdir(cache)
# 创建词典
vectors = torchtext.vocab.Vectors(name=vector_path, cache=cache)
article_field.build_vocab(dataset_train, vectors=vectors, min_freq=min_freq)
# 用split方法划分训练集和开发集
# 因为我们的数据集已经做了划分，实际操作的时候不一定需要这么做
# train_data,valid_data=dataset.split()
# 输出各个训练集和开发集大小
print(f'Number of training examples: {len(dataset_train)}')
print(f'Number of validation examples: {len(dataset_train)}')
# 输出由article_field生成的词典大小
print(f"Unique tokens in article_field vocabulary: {len(article_field.vocab)}")
# 输出第一行试试
# print(vars(train_data.examples[0])['article'])

# 测试集
text_sample = " "
text_sample = text_sample.join(vars(dataset_train.examples[0])['article'])
print(text_sample)
# 输出词典中最高频的100个词
# 如何去除空格&标点？
print(article_field.vocab.freqs.most_common(100))

from torchtext.legacy import data
import torch

# 创建迭代器，供训练使用
BATCH_SIZE = 4
print(torch.cuda.is_available())
# device='cuda1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'
'''
train_iterator, valid_iterator= data.BucketIterator.splits(
    (dataset, dataset),
    batch_size = BATCH_SIZE,
    device = device,)
'''
train_iterator = data.BucketIterator(dataset_train, batch_size=BATCH_SIZE, shuffle=False)
valid_iterator = data.BucketIterator(dataset_valid, batch_size=BATCH_SIZE, shuffle=False)
dataset_test_1=MyDataset_test_1(data_list='data_list.txt',article_field=article_field,label_field=label_field)
test_iterator_1=data.BucketIterator(dataset_test_1,batch_size=BATCH_SIZE,shuffle=False)
dataset_test=MyDataset_test(data_list='data_list.txt',article_field=article_field,label_field=label_field)
test_iterator=data.BucketIterator(dataset_test,batch_size=BATCH_SIZE,shuffle=False)
import torch.nn as nn
import torch
import json
import os
from model import LSTM
# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'

# input的维度为词典的大小，此处暂定为10000
INPUT_DIM = len(article_field.vocab)
#INPUT_DIM=28879
# 这里直接抄了助教的参数
EMBEDDING_DIM = 100
HIDDEN_DIM = 256

OUTPUT_DIM = 1
# 定义模型
model = LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

print(os.getcwd())
print(model.parameters())
datapath_test = "GK I.json"
# os.chdir('../../../..')
# os.chdir('userhome/pythonProject2')
print("okk")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

import torch
import torch.optim as optim

# 优化器 目前采用SGD
optimizer = optim.SGD(params=model.parameters(), lr=1e-3)
print("opt ok")
#criterion = nn.CrossEntropyLoss()
criterion=nn.BCEWithLogitsLoss()
#criterion=nn.MSELoss()
model = model.to(device)
criterion = criterion.to(device)

import time
from train import train,evaluate

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 20

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
    # print("!!!!")
    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    #print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

from test import test,test_1
test(test_iterator)
test_1(model,test_iterator_1)