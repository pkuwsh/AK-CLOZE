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

def read_txt(path):
    file=open(path,'r')
    lines = [re.sub('^[a-z]+', ' ', line.strip().lower()) for line in file]
    return lines


def create_txt():
    Data_Path = '.data/CLOTH/test/high'
    os.chdir(Data_Path)
    fl = open('data_list.txt', 'r')
    for i in range(478):
        file_name = fl.read(13)
        fxxk = fl.read(1)
        path = file_name

        with open(file_name, 'r') as f:
            art = open('articles.txt', 'a')
            opt = open('options.txt', 'a')
            ans = open('answers.txt', 'a')
            lines = [re.sub('^[a-z]+', ' ', line.strip().lower()) for line in f]
            print(lines.__len__())
            # 读取article并写入 article.txt
            art.write(lines[1])
            art.write('\n')
            lines[1].__len__()
            #print('text: %d'% i)


            #读ans
            cnt=-1
            flag=False
            for j in range(4,lines.__len__()):

                if('"answers":' in lines[j] ):
                    flag=True
                    continue
                if(flag==False):
                    continue
                if ('],' in lines[j]):
                    break
                cnt+=1
                ans.write('%d:'%cnt)
                ans.write(lines[j])
                ans.write(' ')
            ans.write('\n')
            # 读取option
            
            cnt = -1;
            # opt.write('text:%d\n'%i)
            for j in range(4, lines.__len__()):
                if ('"answers":' in lines[j]):
                    break
                if ('[' in lines[j]):
                    continue
                elif (']' in lines[j]):
                    continue
                cnt += 1
                if (cnt % 4 == 0):
                    opt.write('%d:' % ((cnt / 4) + 1))
                opt.write(lines[j])
            opt.write('\n')
            #    print('%d' % ((cnt/4)+1))
            # print(lines[j])
            
#create_txt()

#简单读取：
'''
import json
Data_Path = '.data/CLOTH/test/high'

os.chdir(Data_Path)
with open('high3622.json', mode='r', encoding='gbk') as f2:
    setting = json.load(f2)
    print(len(setting))
    print(setting['answers'])

    for i in setting:
        if setting[i]:
            print(setting[i])
            '''
'''
class Mydataset(data.Dataset):
    def __init__(self,datapath):
        #读入文章
        art_path=datapath+'articles.txt'
        self.art_text=read_txt(art_path)
        #读入选项
        opt_path=datapath+'options.txt'
        self.opt_text=read_txt(opt_path)
        #读入答案
        ans_path=datapath+'answers.txt'
        self.ans_text=read_txt(ans_path)
        self.len=self.art_text.__len__()



    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.art_text[index],self.opt_text


dataset=Mydataset('.data/CLOTH/test/high/')
'''
from torchtext.legacy.data import Field
from torchtext.legacy.data import TabularDataset
# 定义Field
from torchtext.legacy.data import Dataset, Example
from torchtext import vocab
import json


article_field = Field(sequential=True, tokenize='spacy',tokenizer_language='en_core_web_sm',
                        lower=False, batch_first=True, include_lengths=True)
option_field=Field(sequential=True, tokenize='spacy',tokenizer_language='en_core_web_sm',
                        lower=False, batch_first=True, include_lengths=True)
answer_field = Field(sequential=False, use_vocab=False)


class MyDataset(Dataset):

    def __init__(self, data_list, article_field, option_field,answer_field):

        fields = [('article', article_field),('option',option_field), ('answer', answer_field)]
        examples = []

        Data_Path = '.data/CLOTH/test/high/'
        os.chdir(Data_Path)
        fl = open('data_list.txt', 'r')
        self.article_list = []
        self.answer_list = []
        self.option_list= []
        for i in range(478):
            file_name = fl.read(13)
            fxxk = fl.read(1)
            data_path = file_name

            with open(data_path, 'r') as f_json:
                file_content = json.load(f_json)
                print(type(self.article_list))
                str1=file_content['article']
                str2=file_content['options']
                str3=file_content['answers']
                #print(type(str1))
                self.article_list.append((str1))
                self.option_list.append((str2))
                self.answer_list.append((str3))
               # self.artile_list+=file_content['article']
                #self.answer_list+=file_content['answers']
               # self.option_list+=file_content['options']


        for index in range(len(self.article_list)):
            examples.append(Example.fromlist([self.article_list[index],
                                              self.option_list[index],self.answer_list[index]], fields))

        super().__init__(examples, fields)


    def sort_key(input):
        return len(input.article)


dataset=MyDataset(data_list='data_list.txt',article_field=article_field,option_field=option_field,answer_field=answer_field)

print(dataset.__len__())
print(dataset[0])

os.chdir('../../../..')

import torchtext.vocab
min_freq=0.01#最小词频
vector_path='glove.6B.50d.txt'
cache = '.vector'
print(os.getcwd())
#f=open('glove.6B.50d.txt','r')
#tmp=f.readlines()
#print(tmp)
if not os.path.exists(cache):
    print("!")
    os.mkdir(cache)
vectors = torchtext.vocab.Vectors(name=vector_path, cache=cache)
article_field.build_vocab(dataset, min_freq=min_freq, vectors=vectors)
option_field.build_vocab(dataset,min_freq=min_freq,vectors=vectors)

train_data,valid_data=dataset.split()
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f"Unique tokens in article_field vocabulary: {len(article_field.vocab)}")
#print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
print(vars(train_data.examples[0])['answer'])
text_sample = " "
text_sample = text_sample.join(vars(train_data.examples[0])['article'])
print(text_sample)

print(article_field.vocab.freqs.most_common(100))



















'''


        # ^[a-z] 匹配以小写字母 开头 的文本串 [^a-z] 表示与 不包含 小写字母的字符匹配,将所有非小写字母的串替换





#path = '/.data/CLOTH/test/high/data_list.txt'

print(os.path.basename(path))    # 查询路径中包含的文件名
print(os.path.dirname(path))     # 查询路径中包含的目录

info = os.path.split(path)      # 将路径分割成文件名和目录两个部分，放在一个表中返回
path2 = os.path.join('/', 'home', 'vamei', 'doc', 'file1.txt')  # 使用目录名和文件名构成一个路径字符串

p_list = [path, path2]
print(os.path.commonprefix(p_list))    # 查询多个路径的共同部分

'''
'''
os.chdir( '.data/CLOTH/test/high')

def read_imdb(folder='train'):  # 本函数已保存在d2lzh包中方便以后使用
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('/.data/CLOTH/test/high/', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

train_data, test_data = read_imdb('train'), read_imdb('test')
'''


'''
def read_time_machine():
    #with open('/home/kesci/input/timemachine7163/timemachine.txt', 'r') as f:
    fl=open('data_list.txt','r')
    for i in range(478):

        file_name=fl.read(13)
        fxxk=fl.read(1)
        #file_name.strip('\n')
        print(file_name)

        with open(file_name,'r') as f:
            # for l in f:
            #     continue
            # print(f.tell())
             #for s in f:
            #     print(s)
            # r的文件指针位置一直在往下移动所以遍历一次再查询会报错
            lines = [re.sub('^[a-z]+', ' ', line.strip().lower()) for line in f]
            #^[a-z] 匹配以小写字母 开头 的文本串 [^a-z] 表示与 不包含 小写字母的字符匹配,将所有非小写字母的串替换
        print(lines[1])
    return lines

'''

'''
lines = read_time_machine()
#lines[1]= re.sub(u' |、|；|，', ' ', lines[1])
print('# sentences %d' % len(lines))
#print(lines[1])
'''

'''
print(lines[1])
for i in range(127):
    print(lines[i])
def tokenize(sentences, token='word'):
    """Split sentences into word or char tokens"""
    if token == 'word':
        return [sentence.split(' ') for sentence in sentences]
    elif token == 'char':
        return [list(sentence) for sentence in sentences]
    else:
        print('ERROR: unkown token type '+token)

tokens = tokenize(lines)
print(tokens[0:2])

class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        counter = count_corpus(tokens)  # 一个字典
        self.token_freqs = list(counter.items())
        # print(counter.items())
        self.idx_to_token = []
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['', '', '', '']
            # pad 在短的句子后面补充,达到句子一样长
            # bos开头
            # eos结尾
            # unk 语料库当中的词,有些词语没有在语料库中
        else:
            self.unk = 0
            self.idx_to_token += ['']
        self.idx_to_token += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in self.idx_to_token]
        #print(self.idx_to_token)
        self.token_to_idx = dict()
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx
        #print(self.token_to_idx)
    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
#     __getitem__(self,key):

# 这个方法返回与指定键想关联的值。对序列来说，键应该是0~n-1的整数，其中n为序列的长度。对映射来说，键可以是任何类型

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(sentences):
    tokens = [tk for st in sentences for tk in st]
    return collections.Counter(tokens)  # 返回一个字典，记录每个词的出现次数
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[0:50])
print(vocab[['mother']])



for i in range(8, 10):
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])

import spacy
text=lines[1]
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
print(lines[1])
print([token.text for token in doc])
'''
