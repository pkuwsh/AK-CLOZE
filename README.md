# AKCLOZE

## 1.基本结构
项目代码的结构依次如下：数据处理（训练集、验证集、测试集；分词、构建词典&词嵌入）-->模型构建-->训练-->测试

## 2.文件描述
main.py包含项目的主体框架
data_utils.py对数据集的数据进行预处理，以供训练使用.
model.py和train.py分别包含LSTM模型和训练相关方法，并在main中调用.
test.py包含测试所需函数.

## 3.数据集
采用CLOTH数据集，https://aclanthology.org/D18-1257/
处理过的数据集附在.data.rar中
