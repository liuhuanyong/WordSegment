#!/usr/bin/env python3
# coding: utf-8
# File: biward_ngram.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-3-27

import math

class BiWardNgram():
    def __init__(self):
        self.word_dict = {} #词语频次词典
        self.trans_dict = {} #每个词后接词的出现个数
        self.word_counts = 0 #语料库中词总数
        self.word_types = 0 #语料库中词种数

    '''初始化模型'''
    def init(self, wordict_path, transdict_path):
        self.word_dict = self.load_model(wordict_path)
        self.trans_dict = self.load_model(transdict_path)
        self.word_types = len(self.word_dict)
        self.word_counts = sum(self.word_dict.values())

    '''加载模型'''
    def load_model(self, model_path):
        f = open(model_path, 'r')
        a = f.read()
        word_dict = eval(a)
        f.close()
        return word_dict

    '''分词'''
    def cut(self, sentence):
        seg_list1 = self.max_forward(sentence)
        seg_list2 = self.max_backward(sentence)
        seg_list = []
        # differ_list1和differ_list2分别记录两个句子词序列不同的部分，用于消除歧义
        differ_list1 = []
        differ_list2 = []
        # pos1和pos2记录两个句子的当前字的位置，cur1和cur2记录两个句子的第几个词
        pos1 = pos2 = 0
        cur1 = cur2 = 0
        
        while 1:
            # 若字符串为空，则不做处理
            if cur1 == len(seg_list1) and cur2 == len(seg_list2):
                break
            # 如果当前位置一样
            if pos1 == pos2:
                # 当前位置一样，并且词也一样
                if len(seg_list1[cur1]) == len(seg_list2[cur2]):
                    pos1 += len(seg_list1[cur1])
                    pos2 += len(seg_list2[cur2])
                    # 说明此时得到两个不同的词序列，根据bigram选择概率大的
                    # 注意算不同的时候要考虑加上前面一个词和后面一个词，拼接的时候再去掉即可
                    if len(differ_list1) > 0:
                        differ_list1.insert(0, seg_list[-1])
                        differ_list2.insert(0, seg_list[-1])
                        
                        if cur1 < len(seg_list1) - 1:
                            differ_list1.append(seg_list1[cur1])
                            differ_list2.append(seg_list2[cur2])

                        p1 = self.compute_likelihood(differ_list1)
                        p2 = self.compute_likelihood(differ_list2)

                        if p1 > p2:
                            differ_list = differ_list1
                        else:
                            differ_list = differ_list2
                        differ_list.remove(differ_list[0])
                        if cur1 < len(seg_list1) - 1:
                            differ_list.remove(seg_list1[cur1])
                        for words in differ_list:
                            seg_list.append(words)
                        differ_list1 = []
                        differ_list2 = []
                        
                    seg_list.append(seg_list1[cur1])
                    cur1 += 1
                    cur2 += 1

                # pos1相同，len(seg_list1[cur1])不同，向后滑动，不同的添加到list中
                elif len(seg_list1[cur1]) > len(seg_list2[cur2]):
                    differ_list2.append(seg_list2[cur2])
                    pos2 += len(seg_list2[cur2])
                    cur2 += 1
                else:
                    differ_list1.append(seg_list1[cur1])
                    pos1 += len(seg_list1[cur1])
                    cur1 += 1
            else:
                # pos1不同，而结束的位置相同，两个同时向后滑动
                if pos1 + len(seg_list1[cur1]) == pos2 + len(seg_list2[cur2]):
                    differ_list1.append(seg_list1[cur1])
                    differ_list2.append(seg_list2[cur2])
                    pos1 += len(seg_list1[cur1])
                    pos2 += len(seg_list2[cur2])
                    cur1 += 1
                    cur2 += 1
                elif pos1 + len(seg_list1[cur1]) > pos2 + len(seg_list2[cur2]):
                    differ_list2.append(seg_list2[cur2])
                    pos2 += len(seg_list2[cur2])
                    cur2 += 1
                else:
                    differ_list1.append(seg_list1[cur1])
                    pos1 += len(seg_list1[cur1])
                    cur1 += 1
        print(seg_list)
        return seg_list

    #计算基于ngram的句子生成概率
    def compute_likelihood(self, seg_list):
        p = 0
        # 由于概率很小，对连乘做了取对数处理转化为加法
        for pos, words in enumerate(seg_list):
            if pos < len(seg_list)-1:
                # 乘以后面词的条件概率
                word1, word2 = words, seg_list[pos+1]
                if word1 not in self.trans_dict.keys():
                    # 加1平滑， 让该词至少出现1次
                    p += math.log(1.0 / self.word_counts)
                else:
                    # 加1平滑
                    fenzi, fenmu = 1.0, self.word_counts
                    #转移概率 p(y|x) = p(yi/y) = count(w)/(count(w1)+ count(w2) + count(w3)+ ..
                    for key in self.trans_dict[word1]:
                        if key == word2:
                            fenzi += self.trans_dict[word1][word2]
                        fenmu += self.trans_dict[word1][key]

                    # log(p(w0)*p(w1|w0)*p(w2|w1)*p(w3|w2)) == log(w0)+ log(p(w1|w0))+ log(p(w2|w1)) + log(p(w3|w2))
                    p += math.log(fenzi / fenmu)

            # 乘以第一个词的概率
            if ( pos == 0 and words != '<BEG>' ) or ( pos == 1 and seg_list[0] == '<BEG>'):
                if words in self.word_dict.keys():
                    p += math.log((float(self.word_dict[words]) + 1.0) / (self.word_types + self.word_counts))
                else:
                    # 加1平滑
                    p += math.log(1.0/ (self.word_types + self.word_counts))
        return p

    #最大前向分词
    def max_forward(self, sentence):
        # 1.从左向右取待切分汉语句的m个字符作为匹配字段，m为大机器词典中最长词条个数。
        # 2.查找大机器词典并进行匹配。若匹配成功，则将这个匹配字段作为一个词切分出来。
        window_size = 5
        cutlist = []
        index = 0
        while index < len(sentence):
            matched = False
            for i in range(window_size, 0, -1):
                cand_word = sentence[index: index + i]
                if cand_word in self.word_dict.keys():
                    cutlist.append(cand_word)
                    matched = True
                    break

            # 如果没有匹配上，则按字符切分
            if not matched:
                i = 1
                cutlist.append(sentence[index])
            index += i
        return cutlist

    #最大后向分词
    def max_backward(self, sentence):
        # 1.从右向左取待切分汉语句的m个字符作为匹配字段，m为大机器词典中最长词条个数。
        # 2.查找大机器词典并进行匹配。若匹配成功，则将这个匹配字段作为一个词切分出来。
        window_size = 5
        cutlist = []
        index = len(sentence)
        while index > 0:
            matched = False
            for i in range(window_size, 0, -1):
                tmp = (i + 1)
                cand_word = sentence[index - tmp: index]
                # 如果匹配上，则将字典中的字符加入到切分字符中
                if cand_word in self.word_dict.keys():
                    cutlist.append(cand_word)
                    matched = True
                    break
            # 如果没有匹配上，则按字符切分
            if not matched:
                tmp = 1
                cutlist.append(sentence[index - 1])

            index -= tmp

        return cutlist[::-1]


if __name__ == '__main__':
    wordict_path = './model/word_dict.model'
    transdict_path = './model/trans_dict.model'
    cuter = BiWardNgram()
    cuter.init(wordict_path, transdict_path)
    sentence = "习近平在慰问电中表示，惊悉贵国克麦罗沃市发生火灾，造成重大人员伤亡和财产损失。我谨代表中国政府和中国人民，并以我个人的名义，对所有遇难者表示沉痛的哀悼，向受伤者和遇难者家属致以深切的同情和诚挚的慰问。"
    #sentence = '2018年12月23日，而我们用到的分词算法是基于字符串的分词方法中的正向最大匹配算法和逆向最大匹配算法。然后对两个方向匹配得出的序列结果中不同的部分运用Bi-gram计算得出较大概率的部分。最后拼接得到最佳词序列。'
    seg_sentence = cuter.cut(sentence)
    print("original sentence: " , sentence)
    print("segment result: ", seg_sentence)
