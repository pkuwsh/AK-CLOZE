import re
import os
import random
#import torch.utils.data as data
import os.path
import string
import torch
from torchtext.legacy.data import Field
from torchtext.legacy.data import Dataset, Example
import json

# 从Dataset类继承出MyDataset类
class MyDataset_train(Dataset):

    def __init__(self, data_list, article_field, label_field):

        os.chdir('/../课件/ai引论/final')
        #print("train path:::", os.getcwd())
        fields = [('article', article_field), ('label', label_field)]
        examples = []
        # 数据目录,为了测试，用了测试集中的部分数据
        Data_Path = '.data/CLOTH/train/high'
        # 移动操作路径至数据目录下
        os.chdir(Data_Path)
        # data_list记录了数据集中各个文件的名称
        fl = open('data_list.txt', 'r')
        # 声明为列表
        self.article_list = []
        self.answer_list = []
        self.option_list = []
        # 因为目前填入的全是正确选项，label全部做成了1
        self.label_list = []
        # 依次读入各个json文件中的内容
        for i in range(3172):
            # for i in range(2):
            # 读取数据文件名
            # file_name = fl.read(13)
            # dot = fl.read(1)
            file_name = fl.readline()
            data_path = file_name
            data_path = data_path.replace('\n', '')
            # lines = [re.sub('^[a-z]+', ' ', line.strip().lower()) for line in f]
            with open(data_path, 'r') as f_json:
                file_content = json.load(f_json)
                punctuation_string = string.punctuation
                # 先存在str中才能用list的append方法添加文本
                article_str_right = file_content['article']
                article_str_wrong1 = file_content['article']  # 正确答案向后平移一个选项
                article_str_wrong2 = file_content['article']  # 正确答案向后平移两个选项
                article_str_wrong3 = file_content['article']  # 正确答案向后平移三个选项
                article_str_rand = file_content['article']  # 随机填空
                article_str = file_content['article']  # 原始题目
                options_str = file_content['options']
                answers_str = file_content['answers']
                label_str = []
                # 不知道为啥 填3次空才能完全填好
                cnt = 0

                # 填出选项正确的答案,同时填出不正确的答案
                for index in range(len(article_str)):
                    # print(article_str[index])

                    if (article_str[index] == '_'):
                        # label_str.append(1)
                        rand_option = random.randint(0, 3)
                        article_str_rand = article_str_rand.replace('_', options_str[cnt][rand_option], 1)
                        if (answers_str[cnt] == 'A'):
                            article_str_right = article_str_right.replace('_', options_str[cnt][0], 1)
                            article_str_wrong1 = article_str_wrong1.replace('_', options_str[cnt][1], 1)
                            article_str_wrong2 = article_str_wrong1.replace('_', options_str[cnt][2], 1)
                            article_str_wrong3 = article_str_wrong3.replace('_', options_str[cnt][3], 1)
                        elif (answers_str[cnt] == 'B'):
                            article_str_right = article_str_right.replace('_', options_str[cnt][1], 1)
                            article_str_wrong1 = article_str_wrong1.replace('_', options_str[cnt][2], 1)
                            article_str_wrong2 = article_str_wrong1.replace('_', options_str[cnt][3], 1)
                            article_str_wrong3 = article_str_wrong3.replace('_', options_str[cnt][0], 1)
                        elif (answers_str[cnt] == 'C'):
                            article_str_right = article_str_right.replace('_', options_str[cnt][2], 1)
                            article_str_wrong1 = article_str_wrong1.replace('_', options_str[cnt][3], 1)
                            article_str_wrong2 = article_str_wrong1.replace('_', options_str[cnt][0], 1)
                            article_str_wrong3 = article_str_wrong3.replace('_', options_str[cnt][1], 1)
                        elif (answers_str[cnt] == 'D'):
                            article_str_right = article_str_right.replace('_', options_str[cnt][3], 1)
                            article_str_wrong1 = article_str_wrong1.replace('_', options_str[cnt][0], 1)
                            article_str_wrong2 = article_str_wrong1.replace('_', options_str[cnt][1], 1)
                            article_str_wrong3 = article_str_wrong3.replace('_', options_str[cnt][2], 1)
                        cnt = cnt + 1
                # print(cnt)
                label_str.append(1)
                # print("right:",article_str_right)
                # print("wrong1:",article_str_wrong1)
                # article_str_right=article_str_right.split(".")
                article_str_right = re.split('[,.]+', article_str_right)
                article_str_rand = re.split('[,.]+', article_str_rand)
                article_str_wrong1 = re.split('[,.]+', article_str_wrong1)
                article_str_wrong2 = re.split('[,.]+', article_str_wrong2)
                article_str_wrong3 = re.split('[,.]+', article_str_wrong3)
                article_str = re.split('[,.]+', article_str)
                # print(article_str_right)
                # print("right:",article_str_right)
                # print("ori:",article_str)
                # print("wrong:",article_str_wrong1)

                for index in range(len(article_str)):
                    if ('_' in article_str[index]):
                        # print("ori:",article_str[index])
                        # print("right:",type(article_str_right[index]))
                        # print("wrong1:",type(article_str_wrong1[index]))
                        # print("rand:",article_str_rand[index])
                        part_right = article_str_right[index]
                        part_wrong1 = article_str_wrong1[index]
                        part_wrong2 = article_str_wrong2[index]
                        part_wrong3 = article_str_wrong3[index]
                        # print(article_str_right)
                        # print("right:", part_right)
                        # print("wrong1:", part_wrong1)

                        if(index>0):
                            part_right=article_str_rand[index-1]+part_right
                            part_wrong1 = article_str_rand[index - 1] + part_wrong1
                            part_wrong2 = article_str_rand[index - 1] + part_wrong2
                            part_wrong3 = article_str_rand[index - 1] + part_wrong3
                        if(index<len(article_str)-1):
                            part_right = part_right+article_str_rand[index + 1]
                            part_wrong1 = part_wrong1+article_str_rand[index + 1]
                            part_wrong2 = part_wrong2+article_str_rand[index + 1]
                            part_wrong3 = part_wrong3+article_str_rand[index + 1]

                        self.article_list.append(part_right)
                        self.article_list.append(part_wrong1)
                        self.article_list.append(part_wrong2)
                        self.article_list.append(part_wrong3)
                        self.label_list.append(1)
                        self.label_list.append(0)
                        self.label_list.append(0)
                        self.label_list.append(0)
                        #
                        '''
                        tmp_art=[part_right,part_wrong1,part_wrong2,part_wrong3]
                        tmp_label=[1,0,0,0]
                        self.article_list.append(tmp_art)
                        self.label_list.append(tmp_label)
                        #print("art:", self.article_list)
                        #print("label:", self.label_list)
                        examples.append(Example.fromlist([tmp_art,
                                                          tmp_label], fields))
                        '''

                # 将读入的数据插入到列表中
                # self.label_list.append(label_str)
                # self.article_list.append((article_str_right))

        # print('art',len(self.article_list))
        # print('label',len(self.label_list))
        for index in range(len(self.article_list)):
            # 把列表的数据汇总到example中
            # examples.append(Example.fromlist([self.article_list[index],
            #                                 self.option_list[index],self.answer_list[index]], fields))
            examples.append(Example.fromlist([(self.article_list[index]),
                                              self.label_list[index]], fields))

        super().__init__(examples, fields)

    def __len__(self):
        # 返回数据集的长度
        return len(self.article_list)

    @staticmethod
    def sort_key(input):
        return len(input.article)


class MyDataset_valid(Dataset):

    def __init__(self, data_list, article_field, label_field):
        os.chdir('../../../..')
        #print("valid path", os.getcwd())
        fields = [('article', article_field), ('label', label_field)]
        examples = []
        # 数据目录,为了测试，用了测试集中的部分数据
        Data_Path = '.data/CLOTH/valid/high'
        # 移动操作路径至数据目录下
        os.chdir(Data_Path)
        # data_list记录了数据集中各个文件的名称
        fl = open('data_list.txt', 'r')
        # 声明为列表
        self.article_list = []
        self.answer_list = []
        self.option_list = []
        # 因为目前填入的全是正确选项，label全部做成了1
        self.label_list = []
        # 依次读入各个json文件中的内容
        for i in range(450):
            # for i in range(2):
            # 读取数据文件名
            file_name = fl.read(13)
            dot = fl.read(1)
            data_path = file_name
            # lines = [re.sub('^[a-z]+', ' ', line.strip().lower()) for line in f]
            with open(data_path, 'r') as f_json:
                file_content = json.load(f_json)
                punctuation_string = string.punctuation
                # 先存在str中才能用list的append方法添加文本
                article_str_right = file_content['article']
                article_str_wrong1 = file_content['article']  # 正确答案向后平移一个选项
                article_str_wrong2 = file_content['article']  # 正确答案向后平移两个选项
                article_str_wrong3 = file_content['article']  # 正确答案向后平移三个选项
                article_str_rand = file_content['article']  # 随机填空
                article_str = file_content['article']  # 原始题目
                options_str = file_content['options']
                answers_str = file_content['answers']
                label_str = []
                # 不知道为啥 填3次空才能完全填好
                cnt = 0

                # 填出选项正确的答案,同时填出不正确的答案
                for index in range(len(article_str)):
                    # print(article_str[index])

                    if (article_str[index] == '_'):
                        # label_str.append(1)
                        rand_option = random.randint(0, 3)
                        article_str_rand = article_str_rand.replace('_', options_str[cnt][rand_option], 1)
                        if (answers_str[cnt] == 'A'):
                            article_str_right = article_str_right.replace('_', options_str[cnt][0], 1)
                            article_str_wrong1 = article_str_wrong1.replace('_', options_str[cnt][1], 1)
                            article_str_wrong2 = article_str_wrong1.replace('_', options_str[cnt][2], 1)
                            article_str_wrong3 = article_str_wrong3.replace('_', options_str[cnt][3], 1)
                        elif (answers_str[cnt] == 'B'):
                            article_str_right = article_str_right.replace('_', options_str[cnt][1], 1)
                            article_str_wrong1 = article_str_wrong1.replace('_', options_str[cnt][2], 1)
                            article_str_wrong2 = article_str_wrong1.replace('_', options_str[cnt][3], 1)
                            article_str_wrong3 = article_str_wrong3.replace('_', options_str[cnt][0], 1)
                        elif (answers_str[cnt] == 'C'):
                            article_str_right = article_str_right.replace('_', options_str[cnt][2], 1)
                            article_str_wrong1 = article_str_wrong1.replace('_', options_str[cnt][3], 1)
                            article_str_wrong2 = article_str_wrong1.replace('_', options_str[cnt][0], 1)
                            article_str_wrong3 = article_str_wrong3.replace('_', options_str[cnt][1], 1)
                        elif (answers_str[cnt] == 'D'):
                            article_str_right = article_str_right.replace('_', options_str[cnt][3], 1)
                            article_str_wrong1 = article_str_wrong1.replace('_', options_str[cnt][0], 1)
                            article_str_wrong2 = article_str_wrong1.replace('_', options_str[cnt][1], 1)
                            article_str_wrong3 = article_str_wrong3.replace('_', options_str[cnt][2], 1)
                        cnt = cnt + 1
                # print(cnt)
                label_str.append(1)
                # print("right:",article_str_right)
                # print("wrong1:",article_str_wrong1)
                # article_str_right=article_str_right.split(".")
                article_str_right = re.split('[,.]+', article_str_right)
                article_str_rand = re.split('[,.]+', article_str_rand)
                article_str_wrong1 = re.split('[,.]+', article_str_wrong1)
                article_str_wrong2 = re.split('[,.]+', article_str_wrong2)
                article_str_wrong3 = re.split('[,.]+', article_str_wrong3)
                article_str = re.split('[,.]+', article_str)
                # print(article_str_right)
                # print("right:",article_str_right)
                # print("ori:",article_str)
                # print("wrong:",article_str_wrong1)

                for index in range(len(article_str)):
                    if ('_' in article_str[index]):
                        # print("ori:",article_str[index])
                        # print("right:",type(article_str_right[index]))
                        # print("wrong1:",type(article_str_wrong1[index]))
                        # print("rand:",article_str_rand[index])
                        part_right = article_str_right[index]
                        part_wrong1 = article_str_wrong1[index]
                        part_wrong2 = article_str_wrong2[index]
                        part_wrong3 = article_str_wrong3[index]
                        # print(article_str_right)
                        # print("right:", part_right)
                        # print("wrong1:", part_wrong1)

                        if(index>0):
                            part_right=article_str_rand[index-1]+part_right
                            part_wrong1 = article_str_rand[index - 1] + part_wrong1
                            part_wrong2 = article_str_rand[index - 1] + part_wrong2
                            part_wrong3 = article_str_rand[index - 1] + part_wrong3
                        if(index<len(article_str)-1):
                            part_right = part_right+article_str_rand[index + 1]
                            part_wrong1 = part_wrong1+article_str_rand[index + 1]
                            part_wrong2 = part_wrong2+article_str_rand[index + 1]
                            part_wrong3 = part_wrong3+article_str_rand[index + 1]

                        self.article_list.append(part_right)
                        self.article_list.append(part_wrong1)
                        self.article_list.append(part_wrong2)
                        self.article_list.append(part_wrong3)
                        self.label_list.append(1)
                        self.label_list.append(0)
                        self.label_list.append(0)
                        self.label_list.append(0)
                        #
                        '''
                        tmp_art=[part_right,part_wrong1,part_wrong2,part_wrong3]
                        tmp_label=[1,0,0,0]
                        self.article_list.append(tmp_art)
                        self.label_list.append(tmp_label)
                        #print("art:", self.article_list)
                        #print("label:", self.label_list)
                        examples.append(Example.fromlist([tmp_art,
                                                          tmp_label], fields))
                        '''

                # 将读入的数据插入到列表中
                # self.label_list.append(label_str)
                # self.article_list.append((article_str_right))

        # print('art',len(self.article_list))
        # print('label',len(self.label_list))
        for index in range(len(self.article_list)):
            # 把列表的数据汇总到example中
            # examples.append(Example.fromlist([self.article_list[index],
            #                                 self.option_list[index],self.answer_list[index]], fields))
            examples.append(Example.fromlist([(self.article_list[index]),
                                              self.label_list[index]], fields))

        super().__init__(examples, fields)

    def __len__(self):
        # 返回数据集的长度
        return len(self.article_list)

    @staticmethod
    def sort_key(input):
        return len(input.article)


# 声明dataset对象，读取相应的文件；此后可以从dataset中直接调用相关数据


# 把程序工作目录调回源目录




datapath_test = "GK I.json"


class MyDataset_test(Dataset):
    def __init__(self, data_list, article_field, label_field):
        fields = [('article', article_field), ('label', label_field)]
        examples = []
        self.article_list = []
        self.answer_list = []
        self.option_list = []
        # 因为目前填入的全是正确选项，label全部做成了1
        self.label_list = []
        os.chdir('/../课件/ai引论/final')
        #print(os.getcwd())
        with open(datapath_test, 'r') as f_json:
            file_content = json.load(f_json)
            punctuation_string = string.punctuation
            # 先存在str中才能用list的append方法添加文本
            article_str_right = file_content['article']
            article_str_wrong1 = file_content['article']  # 正确答案向后平移一个选项
            article_str_wrong2 = file_content['article']  # 正确答案向后平移两个选项
            article_str_wrong3 = file_content['article']  # 正确答案向后平移三个选项
            article_str_rand = file_content['article']  # 随机填空
            article_str = file_content['article']  # 原始题目
            options_str = file_content['options']
            answers_str = file_content['answers']
            label_str = []
            # 不知道为啥 填3次空才能完全填好
            cnt = 0

            # 填出选项正确的答案,同时填出不正确的答案
            for index in range(len(article_str)):
                # print(article_str[index])

                if (article_str[index] == '_'):
                    # label_str.append(1)
                    rand_option = random.randint(0, 3)
                    article_str_rand = article_str_rand.replace('_', options_str[cnt][rand_option], 1)
                    '''
                    if (answers_str[cnt] == 'A'):
                        article_str_right = article_str_right.replace('_', options_str[cnt][0], 1)
                        article_str_wrong1 = article_str_wrong1.replace('_', options_str[cnt][1], 1)
                        article_str_wrong2 = article_str_wrong1.replace('_', options_str[cnt][2], 1)
                        article_str_wrong3 = article_str_wrong3.replace('_', options_str[cnt][3], 1)
                    elif (answers_str[cnt] == 'B'):
                        article_str_right = article_str_right.replace('_', options_str[cnt][1], 1)
                        article_str_wrong1 = article_str_wrong1.replace('_', options_str[cnt][2], 1)
                        article_str_wrong2 = article_str_wrong1.replace('_', options_str[cnt][3], 1)
                        article_str_wrong3 = article_str_wrong3.replace('_', options_str[cnt][0], 1)
                    elif (answers_str[cnt] == 'C'):
                        article_str_right = article_str_right.replace('_', options_str[cnt][2], 1)
                        article_str_wrong1 = article_str_wrong1.replace('_', options_str[cnt][3], 1)
                        article_str_wrong2 = article_str_wrong1.replace('_', options_str[cnt][0], 1)
                        article_str_wrong3 = article_str_wrong3.replace('_', options_str[cnt][1], 1)
                    elif (answers_str[cnt] == 'D'):
                        article_str_right = article_str_right.replace('_', options_str[cnt][3], 1)
                        article_str_wrong1 = article_str_wrong1.replace('_', options_str[cnt][0], 1)
                        article_str_wrong2 = article_str_wrong1.replace('_', options_str[cnt][1], 1)
                        article_str_wrong3 = article_str_wrong3.replace('_', options_str[cnt][2], 1)
                        '''
                    article_str_right = article_str_right.replace('_', options_str[cnt][0], 1)
                    article_str_wrong1 = article_str_wrong1.replace('_', options_str[cnt][1], 1)
                    article_str_wrong2 = article_str_wrong2.replace('_', options_str[cnt][2], 1)
                    article_str_wrong3 = article_str_wrong3.replace('_', options_str[cnt][3], 1)
                    cnt = cnt + 1
            # print(cnt)
            label_str.append(1)
            # print("right:",article_str_right)
            # print("wrong1:",article_str_wrong1)
            # article_str_right=article_str_right.split(".")
            article_str_right = re.split('[,.]+', article_str_right)
            article_str_rand = re.split('[,.]+', article_str_rand)
            article_str_wrong1 = re.split('[,.]+', article_str_wrong1)
            article_str_wrong2 = re.split('[,.]+', article_str_wrong2)
            article_str_wrong3 = re.split('[,.]+', article_str_wrong3)
            article_str = re.split('[,.]+', article_str)
            # print(article_str_right)
            print("right:", article_str_right)
            # print("ori:",article_str)
            # print("wrong:",article_str_wrong1)

            for index in range(len(article_str)):
                if ('_' in article_str[index]):
                    # print("ori:",article_str[index])
                    # print("right:",type(article_str_right[index]))
                    # print("wrong1:",type(article_str_wrong1[index]))
                    # print("rand:",article_str_rand[index])
                    part_right = article_str_right[index]
                    part_wrong1 = article_str_wrong1[index]
                    part_wrong2 = article_str_wrong2[index]
                    part_wrong3 = article_str_wrong3[index]
                    # print(article_str_right)
                    # print("right:", part_right)
                    # print("wrong1:", part_wrong1)

                    if (index > 0):
                        part_right = article_str_rand[index - 1] + part_right
                        part_wrong1 = article_str_rand[index - 1] + part_wrong1
                        part_wrong2 = article_str_rand[index - 1] + part_wrong2
                        part_wrong3 = article_str_rand[index - 1] + part_wrong3
                    if (index < len(article_str) - 1):
                        part_right = part_right + article_str_rand[index + 1]
                        part_wrong1 = part_wrong1 + article_str_rand[index + 1]
                        part_wrong2 = part_wrong2 + article_str_rand[index + 1]
                        part_wrong3 = part_wrong3 + article_str_rand[index + 1]

                    self.article_list.append(part_right)
                    self.article_list.append(part_wrong1)
                    self.article_list.append(part_wrong2)
                    self.article_list.append(part_wrong3)
                    self.label_list.append(1)
                    self.label_list.append(0)
                    self.label_list.append(0)
                    self.label_list.append(0)
        for index in range(len(self.article_list)):
            # 把列表的数据汇总到example中
            # examples.append(Example.fromlist([self.article_list[index],
            #                                 self.option_list[index],self.answer_list[index]], fields))
            examples.append(Example.fromlist([(self.article_list[index]),
                                              self.label_list[index]], fields))
        self.cnt = cnt
        super().__init__(examples, fields)

    def __len__(self):
        # 返回数据集的长度
        # return len(self.article_list)
        return self.cnt * 4

    @staticmethod
    def sort_key(input):
        return len(input.article)

from torchtext.legacy import data
article_field = Field(sequential=True, tokenize='spacy', tokenizer_language='en_core_web_sm',
                      lower=False, batch_first=True, include_lengths=True, use_vocab=True)

label_field = Field(sequential=False, use_vocab=False, fix_length=20, pad_token=0)
BATCH_SIZE=4

dataset_test = MyDataset_test(data_list='data_list.txt', article_field=article_field, label_field=label_field)
test_iterator = data.BucketIterator(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
# print(dataset_test.len())
# print(test_iterator.batch_size)
# print(dataset_test.__len__())
# model=torch.load('model.pkl')

import torch


class MyDataset_test_1(Dataset):
    #print(os.getcwd())
    #os.chdir('userhome/pythonProject2/')
    #print(os.getcwd())

    def __init__(self, data_list, article_field, label_field):
        # os.chdir('../../../..')
        #print("valid path", os.getcwd())
        fields = [('article', article_field), ('label', label_field)]
        examples = []
        # 数据目录,为了测试，用了测试集中的部分数据
        Data_Path = '.data/CLOTH/test/high'
        # 移动操作路径至数据目录下
        os.chdir(Data_Path)
        # data_list记录了数据集中各个文件的名称
        fl = open('data_list.txt', 'r')
        # 声明为列表
        self.article_list = []
        self.answer_list = []
        self.option_list = []
        # 因为目前填入的全是正确选项，label全部做成了1
        self.label_list = []
        # 依次读入各个json文件中的内容
        for i in range(478):
            # for i in range(2):
            # 读取数据文件名
            file_name = fl.read(13)
            dot = fl.read(1)
            data_path = file_name
            # lines = [re.sub('^[a-z]+', ' ', line.strip().lower()) for line in f]
            with open(data_path, 'r') as f_json:
                file_content = json.load(f_json)
                punctuation_string = string.punctuation
                # 先存在str中才能用list的append方法添加文本
                article_str_right = file_content['article']
                article_str_wrong1 = file_content['article']  # 正确答案向后平移一个选项
                article_str_wrong2 = file_content['article']  # 正确答案向后平移两个选项
                article_str_wrong3 = file_content['article']  # 正确答案向后平移三个选项
                article_str_rand = file_content['article']  # 随机填空
                article_str = file_content['article']  # 原始题目
                options_str = file_content['options']
                answers_str = file_content['answers']
                label_str = []
                # 不知道为啥 填3次空才能完全填好
                cnt = 0

                # 填出选项正确的答案,同时填出不正确的答案
                for index in range(len(article_str)):
                    # print(article_str[index])

                    if (article_str[index] == '_'):
                        # label_str.append(1)
                        rand_option = random.randint(0, 3)
                        article_str_rand = article_str_rand.replace('_', options_str[cnt][rand_option], 1)

                        if (answers_str[cnt] == 'A'):
                            self.label_list.append(1)
                            self.label_list.append(0)
                            self.label_list.append(0)
                            self.label_list.append(0)
                        elif (answers_str[cnt] == 'B'):
                            self.label_list.append(0)
                            self.label_list.append(1)
                            self.label_list.append(0)
                            self.label_list.append(0)
                        elif (answers_str[cnt] == 'C'):
                            self.label_list.append(0)
                            self.label_list.append(0)
                            self.label_list.append(1)
                            self.label_list.append(0)
                        elif (answers_str[cnt] == 'D'):
                            self.label_list.append(0)
                            self.label_list.append(0)
                            self.label_list.append(0)
                            self.label_list.append(1)

                        article_str_right = article_str_right.replace('_', options_str[cnt][0], 1)
                        article_str_wrong1 = article_str_wrong1.replace('_', options_str[cnt][1], 1)
                        article_str_wrong2 = article_str_wrong2.replace('_', options_str[cnt][2], 1)
                        article_str_wrong3 = article_str_wrong3.replace('_', options_str[cnt][3], 1)
                        cnt = cnt + 1
                # print(cnt)
                label_str.append(1)
                # print("right:",article_str_right)
                # print("wrong1:",article_str_wrong1)
                # article_str_right=article_str_right.split(".")
                article_str_right = re.split('[,.]+', article_str_right)
                article_str_rand = re.split('[,.]+', article_str_rand)
                article_str_wrong1 = re.split('[,.]+', article_str_wrong1)
                article_str_wrong2 = re.split('[,.]+', article_str_wrong2)
                article_str_wrong3 = re.split('[,.]+', article_str_wrong3)
                article_str = re.split('[,.]+', article_str)
                # print(article_str_right)
                # print("right:",article_str_right)
                # print("ori:",article_str)
                # print("wrong:",article_str_wrong1)

                for index in range(len(article_str)):
                    if ('_' in article_str[index]):
                        # print("ori:",article_str[index])
                        # print("right:",type(article_str_right[index]))
                        # print("wrong1:",type(article_str_wrong1[index]))
                        # print("rand:",article_str_rand[index])
                        part_right = article_str_right[index]
                        part_wrong1 = article_str_wrong1[index]
                        part_wrong2 = article_str_wrong2[index]
                        part_wrong3 = article_str_wrong3[index]
                        # print(article_str_right)
                        # print("right:", part_right)
                        # print("wrong1:", part_wrong1)

                        if(index>0):
                            part_right=article_str_rand[index-1]+part_right
                            part_wrong1 = article_str_rand[index - 1] + part_wrong1
                            part_wrong2 = article_str_rand[index - 1] + part_wrong2
                            part_wrong3 = article_str_rand[index - 1] + part_wrong3
                        if(index<len(article_str)-1):
                            part_right = part_right+article_str_rand[index + 1]
                            part_wrong1 = part_wrong1+article_str_rand[index + 1]
                            part_wrong2 = part_wrong2+article_str_rand[index + 1]
                            part_wrong3 = part_wrong3+article_str_rand[index + 1]

                        self.article_list.append(part_right)
                        self.article_list.append(part_wrong1)
                        self.article_list.append(part_wrong2)
                        self.article_list.append(part_wrong3)
                        '''
                        self.label_list.append(1)
                        self.label_list.append(0)
                        self.label_list.append(0)
                        self.label_list.append(0)
                        '''
                        #
                        '''
                        tmp_art=[part_right,part_wrong1,part_wrong2,part_wrong3]
                        tmp_label=[1,0,0,0]
                        self.article_list.append(tmp_art)
                        self.label_list.append(tmp_label)
                        #print("art:", self.article_list)
                        #print("label:", self.label_list)
                        examples.append(Example.fromlist([tmp_art,
                                                          tmp_label], fields))
                        '''

                # 将读入的数据插入到列表中
                # self.label_list.append(label_str)
                # self.article_list.append((article_str_right))

        # print('art',len(self.article_list))
        # print('label',len(self.label_list))
        for index in range(len(self.article_list)):
            # 把列表的数据汇总到example中
            # examples.append(Example.fromlist([self.article_list[index],
            #                                 self.option_list[index],self.answer_list[index]], fields))
            examples.append(Example.fromlist([(self.article_list[index]),
                                              self.label_list[index]], fields))

        super().__init__(examples, fields)

    def __len__(self):
        # 返回数据集的长度
        return len(self.article_list)

    @staticmethod
    def sort_key(input):
        return len(input.article)


dataset_test_1 = MyDataset_test_1(data_list='data_list.txt', article_field=article_field, label_field=label_field)
test_iterator_1 = data.BucketIterator(dataset_test_1, batch_size=BATCH_SIZE, shuffle=False)
