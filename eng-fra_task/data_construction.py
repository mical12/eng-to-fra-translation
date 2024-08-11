#定义开始结束标志sos,eos
import re

import torch
import unicodedata

sos_token = 0
eos_token= 1
#定义语言类->>构建单词-index index-单词的字典 后续每个单词可以变成数字  num_word 用于词嵌入
class lang():
    # 初始化类属性包括，语言（name）,词汇对应索引字典（word2index）,索引对应词汇字典，词汇个数
    def __init__(self, Lang):
        self.name = Lang
        self.word2index = {'sos_token':0, 'eos_token':1}
        self.index2word = {0: 'sos_token', 1: 'eos_token'}
        self.num_words = 2

    # 定义添加词汇函数，传一个词汇，加入到语言类的两个字典中，并更新词汇长度
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.num_words += 1

    # 因为我们传入的是句子，故在词汇基础上构建添加句子函数
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

"""
下面函数基本都是对文本数据进行清洗，使之得到的好的数据
上面的类用来构建处理好的数据的向量"""
#规范化字符串（变小写，去除空白符重音标记符，在.!?前加空格，含有其他字符的换成空格）
def noramalize_string(s):
    s=unicode_to_ascii(s.lower().strip())
    s=re.sub(r'([.!?])',r' \1',s)
    s=re.sub(r' {2}([.!?])',r' \1',s)
    s=re.sub('[^a-zA-Z.!?]+',r' ',s)
    return s

#去掉重音标记
def unicode_to_ascii(s):
    # 将字符串分解
    s_list = []
    s_new = unicodedata.normalize('NFD', s)
    # 拿到正规字符串（先用列表保存规范字符，在转化成字符串）
    for c in s_new:
        if unicodedata.category(c) != 'Mn':
            s_list.append(c)
    return ''.join(s_list)

#读取文本数据,将英法字符串分别处理规范化，返回创建的两个类对象以及处理后的字符串列表
def read_text(lang1,lang2):
    with open('./eng-fra.txt','r',encoding='utf-8') as f:
        lines = f.read().strip()
        lines = lines.split('\n')
        seq_list = []
        for line in lines:
            list1 = []
            for one in line.split('\t'):
                one = noramalize_string(one)
                list1.append(one)
            seq_list.append(list1)
    mylang1 = lang(lang1)
    mylang2 = lang(lang2)
    return mylang1, mylang2,seq_list

#过滤列表，提取我们想要的英法翻译句子（想要以某某开头的英法句子）
def filterpairs(pairs):
    max_seq_length = 10
    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )
    pairs_new = []
    for pair in pairs :
        if pair[0].startswith(eng_prefixes) and len(pair[0].split(' ')) < max_seq_length and len(pair[1].split(' ')) < max_seq_length :
            pairs_new.append(pair)
    return pairs_new
#大封装前面所有的函数，将文本数据转为两个对象，规范化后的列表
def prepare_data(lang1,lang2):
    lang1, lang2, pairs = read_text(lang1,lang2)
    pairs = filterpairs(pairs)
    for pair in pairs:
        lang1.add_sentence(pair[0])
        lang2.add_sentence(pair[1])
    return lang1, lang2, pairs

"""
下面的函数用来做词汇转化向量的用途"""
#对列表中的一对句子处理，依据各字典中的word2index完成
def tensor_from_sequence(input_lang,output_lang,pair):
    list = []
    for word in pair[0].split(' '):
        list.append(input_lang.word2index[word])
    list.append(eos_token)
    list2 = []
    for word in pair[1].split(' '):
        list2.append(output_lang.word2index[word])
    list2.append(eos_token)
    return torch.tensor(list,dtype=torch.long).view(-1,1),torch.tensor(list2,dtype=torch.long).view(-1,1)


if __name__ == '__main__':
    #测试1
    # eng1 = lang('en')
    # print(eng1.index2word)
    # print(eng1.word2index)
    # print(eng1.name)
    # print(eng1.num_words)
    # eng1.add_sentence('hello i am jay')
    # print(eng1.index2word)
    # print(eng1.word2index)
    # print(eng1.name)
    # print(eng1.num_words)
    # #测试2
    # s = "J'ai 19 ans."
    # s_new = noramalize_string(s)
    # print(s_new)
    #测试3
    # lang1, lang2,pairs=read_text('eng','fra')
    # print(lang1)
    # print(lang2)
    # print(pairs)
    # #测试4
    # pairs_new = filterpairs(pairs)
    # print(pairs_new[:5])
    #测试5
    lang1, lang2, pairs = prepare_data('eng', 'fra')
    print(lang1.num_words)
    print(lang2.num_words)
    print(pairs[:5])
    input,output = tensor_from_sequence(lang1, lang2, pairs[0])
    print(input)
    print(output)



