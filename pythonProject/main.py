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

#定义field，对输入文本的格式进行预设置
#这里为了测试，只做了文章、选项和答案的读入
article_field = Field(sequential=True, tokenize='spacy',tokenizer_language='en_core_web_sm',
                        lower=False, batch_first=True, include_lengths=True)
option_field=Field(sequential=True, tokenize='spacy',tokenizer_language='en_core_web_sm',
                        lower=False, batch_first=True, include_lengths=True)
answer_field = Field(sequential=False, use_vocab=False)

#从Dataset类继承出MyDataset类
class MyDataset(Dataset):

    def __init__(self, data_list, article_field, option_field,answer_field):

        fields = [('article', article_field),('option',option_field), ('answer', answer_field)]
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
        #依次读入各个json文件中的内容
        for i in range(478):
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
                str1=file_content['article']
                str2=file_content['options']
                str3=file_content['answers']
                #print(type(str1))
                self.article_list.append((str1))
                self.option_list.append((str2))
                self.answer_list.append((str3))
                #如下写法会报错
               # self.artile_list+=file_content['article']
                #self.answer_list+=file_content['answers']
               # self.option_list+=file_content['options']


        for index in range(len(self.article_list)):
            #直接抄来的
            examples.append(Example.fromlist([self.article_list[index],
                                              self.option_list[index],self.answer_list[index]], fields))

        super().__init__(examples, fields)


    def sort_key(input):
        #返回数据集的长度
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
#f=open('glove.6B.50d.txt','r')
#tmp=f.readlines()
#print(tmp)
if not os.path.exists(cache):
    print("!")
    os.mkdir(cache)
#创建词典
vectors = torchtext.vocab.Vectors(name=vector_path, cache=cache)
article_field.build_vocab(dataset, min_freq=min_freq, vectors=vectors)
option_field.build_vocab(dataset,min_freq=min_freq,vectors=vectors)

#用split方法划分训练集和开发集
#因为我们的数据集已经做了划分，实际操作的时候不一定需要这么做
train_data,valid_data=dataset.split()
#输出各个训练集和开发集大小
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
#输出由article_field生成的词典大小
print(f"Unique tokens in article_field vocabulary: {len(article_field.vocab)}")
#输出第一行试试
print(vars(train_data.examples[0])['answer'])
#测试集
text_sample = " "
text_sample = text_sample.join(vars(train_data.examples[0])['article'])

#输出词典中最高频的100个词
#如何去除空格&标点？
print(article_field.vocab.freqs.most_common(100))
from torch.utils.data import DataLoader

from torchtext.legacy import data
#创建迭代器，供训练使用
train_iterator = data.BucketIterator(dataset,batch_size=4,shuffle=True)

print(train_iterator.__len__())


