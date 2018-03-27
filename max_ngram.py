#!/usr/bin/env python3
# coding: utf-8
# File: max_ngram.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-3-27

import math

class MaxProbCut:
    def __init__(self):
        self.word_dict = {}  # 记录概率,1-gram
        self.word_dict_count = {}  # 记录词频,1-gram
        self.trans_dict = {}  # 记录概率,2-gram
        self.trans_dict_count = {}  # 记录词频,2-gram
        self.max_wordlen = 0  #词的最长长度
        self.all_freq = 0  # 所有词的词频总和,1-gram
        word_count_path = "./model/word_dict.model"
        word_trans_path = './model/trans_dict.model'
        self.init(word_count_path, word_trans_path)

    # 加载词典
    def init(self, word_count_path, word_trans_path):
        self.word_dict_count = self.load_model(word_count_path)
        self.all_freq = sum(self.word_dict_count.values())  # 所有词的词频
        self.max_wordlen = max(len(key) for key in self.word_dict_count.keys())
        for key in self.word_dict_count:
            self.word_dict[key] = math.log(self.word_dict_count[key] / self.all_freq)
        #计算转移概率
        Trans_dict = self.load_model(word_trans_path)
        for pre_word, post_info in Trans_dict.items():
            for post_word, count in post_info.items():
                word_pair = pre_word + ' ' + post_word
                self.trans_dict_count[word_pair] = float(count)
                if pre_word in self.word_dict_count.keys():
                    self.trans_dict[key] = math.log(count / self.word_dict_count[pre_word])  # 取自然对数，归一化
                else:
                    self.trans_dict[key] = self.word_dict[post_word]

    #加载预训练模型
    def load_model(self, model_path):
        f = open(model_path, 'r')
        a = f.read()
        word_dict = eval(a)
        f.close()
        return word_dict

    # 估算未出现的词的概率,根据beautiful data里面的方法估算，平滑算法
    def get_unknow_word_prob(self, word):
        return math.log(1.0 / (self.all_freq ** len(word)))

    # 获取候选词的概率
    def get_word_prob(self, word):
        if word in self.word_dict.keys():  # 如果字典包含这个词
            prob = self.word_dict[word]
        else:
            prob = self.get_unknow_word_prob(word)
        return prob

    #获取转移概率
    def get_word_trans_prob(self, pre_word, post_word):
        trans_word = pre_word + " " + post_word
        
        if trans_word in self.trans_dict_count.keys():
            trans_prob = math.log(self.trans_dict_count[trans_word] / self.word_dict_count[pre_word])
        else:
            trans_prob = self.get_word_prob(post_word)
        return trans_prob

    # 寻找node的最佳前驱节点，方法为寻找所有可能的前驱片段
    def get_best_pre_node(self, sentence, node, node_state_list):
        # 如果node比最大词长小，取的片段长度以node的长度为限
        max_seg_length = min([node, self.max_wordlen])
        pre_node_list = []  # 前驱节点列表

        # 获得所有的前驱片段，并记录累加概率
        for segment_length in range(1, max_seg_length + 1):
            segment_start_node = node - segment_length
            segment = sentence[segment_start_node:node]  # 获取片段
            pre_node = segment_start_node  # 取该片段，则记录对应的前驱节点
            if pre_node == 0:
                # 如果前驱片段开始节点是序列的开始节点，
                # 则概率为<S>转移到当前词的概率
                segment_prob = self.get_word_trans_prob("<BEG>", segment)
            else:  # 如果不是序列开始节点，按照二元概率计算
                # 获得前驱片段的前一个词
                pre_pre_node = node_state_list[pre_node]["pre_node"]
                pre_pre_word = sentence[pre_pre_node:pre_node]
                segment_prob = self.get_word_trans_prob(pre_pre_word, segment)

            pre_node_prob_sum = node_state_list[pre_node]["prob_sum"]  # 前驱节点的概率的累加值
            # 当前node一个候选的累加概率值
            candidate_prob_sum = pre_node_prob_sum + segment_prob
            pre_node_list.append((pre_node, candidate_prob_sum))

        # 找到最大的候选概率值
        (best_pre_node, best_prob_sum) = max(pre_node_list, key=lambda d: d[1])

        return best_pre_node, best_prob_sum

    #切词主函数
    def cut_main(self, sentence):
        sentence = sentence.strip()
        # 初始化
        node_state_list = []  # 记录节点的最佳前驱，index就是位置信息
        # 初始节点，也就是0节点信息
        ini_state = {}
        ini_state["pre_node"] = -1  # 前一个节点
        ini_state["prob_sum"] = 0  # 当前的概率总和
        node_state_list.append(ini_state)
        # 字符串概率为2元概率， P(a b c) = P(a|<S>)P(b|a)P(c|b)
        # 逐个节点寻找最佳前驱节点
        for node in range(1, len(sentence) + 1):
            # 寻找最佳前驱，并记录当前最大的概率累加值
            (best_pre_node, best_prob_sum) = self.get_best_pre_node(sentence, node, node_state_list)

            # 添加到队列
            cur_node = {}
            cur_node["pre_node"] = best_pre_node
            cur_node["prob_sum"] = best_prob_sum
            node_state_list.append(cur_node)
            # print "cur node list",node_state_list

        # step 2, 获得最优路径,从后到前
        best_path = []
        node = len(sentence)  # 最后一个点
        best_path.append(node)
        while True:
            pre_node = node_state_list[node]["pre_node"]
            if pre_node == -1:
                break
            node = pre_node
            best_path.append(node)
        best_path.reverse()

        # step 3, 构建切分
        word_list = []
        for i in range(len(best_path) - 1):
            left = best_path[i]
            right = best_path[i + 1]
            word = sentence[left:right]
            word_list.append(word)

        return word_list

    #测试接口
    def cut(self, sentence):
        return self.cut_main(sentence)

'''
def test():
    cuter = MaxProbCut()
    sentence = "今天我不知道你为什么会这个样子"
    seg_sentence = cuter.cut(sentence)
    print("original sentence: " , sentence)
    print("segment result: ", seg_sentence)

test()
'''
