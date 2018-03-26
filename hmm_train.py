#!/usr/bin/env python3
# coding: utf-8
# open: hmm_train.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-3-26

class HmmTrain:
    def __init__(self):
        self.line_index = -1
        self.char_set = set()

    def init(self):  #初始化字典
        trans_dict = {}  # 存储状态转移概率
        emit_dict = {}  # 发射概率(状态->词语的条件概率)
        Count_dict = {}  #存储所有状态序列 ，用于归一化分母
        start_dict = {}  # 存储状态的初始概率
        state_list = ['B', 'M', 'E', 'S'] #状态序列
    
        for state in state_list:
            trans_dict[state] = {}
            for state1 in state_list:
                trans_dict[state][state1] = 0.0
    
        for state in state_list:
            start_dict[state] = 0.0
            emit_dict[state] = {}
            Count_dict[state] = 0
    
       # print(trans_dict) #{'B': {'B': 0.0, 'S': 0.0, 'M': 0.0, 'E': 0.0}, 'S': {'B': 0.0, 'S': 0.0, 'M': 0.0, 'E': 0.0},。。。}
       # print(emit_dict) # {'B': {}, 'S': {}, 'M': {}, 'E': {}}
       # print(start_dict) # {'B': 0.0, 'S': 0.0, 'M': 0.0, 'E': 0.0}
       # print(Count_dict) # {'B': 0, 'S': 0, 'M': 0, 'E': 0}
        return trans_dict, emit_dict, start_dict, Count_dict
    
    '''保存模型'''
    def save_model(self, word_dict, model_path):
        f = open(model_path, 'w')
        f.write(str(word_dict))
        f.close()
    
    '''词语状态转换'''
    def get_word_status(self, word):  #根据词语，输出词语对应的SBME状态
        '''
        S:单字词
        B:词的开头
        M:词的中间
        E:词的末尾
        能 ['S']
        前往 ['B', 'E']
        科威特 ['B', 'M', 'E']
        '''
        word_status = []
        if len(word) == 1:
            word_status.append('S')
        elif len(word) == 2:
            word_status = ['B','E']
        else:
            M_num = len(word) - 2
            M_list = ['M'] * M_num
            word_status.append('B')
            word_status.extend(M_list)
            word_status.append('E')
            
        return word_status
    
    '''基于人工标注语料库，训练发射概率，初始状态， 转移概率'''
    def train(self, train_filepath, trans_path, emit_path, start_path):
        trans_dict, emit_dict, start_dict, Count_dict = self.init()
        for line in open(train_filepath):
            self.line_index += 1

            line = line.strip()
            if not line:
                continue
    
            char_list = []
            for i in range(len(line)):
                if line[i] == " ":
                    continue
                char_list.append(line[i])

            self.char_set = set(char_list)   #训练预料库中所有字的集合
    
            word_list = line.split(" ")
            line_status = [] #统计状态序列
    
            for word in word_list:
                line_status.extend(self.get_word_status(word))   #一句话对应一行连续的状态
    
            if len(char_list) == len(line_status):
               # print(word_list) # ['但', '从', '生物学', '眼光', '看', '就', '并非', '如此', '了', '。']
               # print(line_status) # ['S', 'S', 'B', 'M', 'E', 'B', 'E', 'S', 'S', 'B', 'E', 'B', 'E', 'S', 'S']
               # print('******')
                for i in range(len(line_status)):
                    if i == 0:#如果只有一个词，则直接算作是初始概率
                        start_dict[line_status[0]] += 1   #start_dict记录句子第一个字的状态，用于计算初始状态概率
                        Count_dict[line_status[0]] += 1   #记录每一个状态的出现次数
                    else:#统计上一个状态到下一个状态，两个状态之间的转移概率
                        trans_dict[line_status[i-1]][line_status[i]] += 1    #用于计算转移概率
                        Count_dict[line_status[i]] += 1
                        #统计发射概率
                        if char_list[i] not in emit_dict[line_status[i]]:
                            emit_dict[line_status[i]][char_list[i]] = 0.0
                        else:
                            emit_dict[line_status[i]][char_list[i]] += 1   #用于计算发射概率
            else:
                continue
    
        # print(emit_dict)#{'S': {'否': 10.0, '昔': 25.0, '直': 238.0, '六': 1004.0, '殖': 17.0, '仗': 36.0, '挪': 15.0, '朗': 151.0
        # print(trans_dict)#{'S': {'S': 747969.0, 'E': 0.0, 'M': 0.0, 'B': 563988.0}, 'E': {'S': 737404.0, 'E': 0.0, 'M': 0.0, 'B': 651128.0},
        # print(start_dict) #{'S': 124543.0, 'E': 0.0, 'M': 0.0, 'B': 173416.0}
    
        #进行归一化
        for key in start_dict:  # 状态的初始概率
            start_dict[key] = start_dict[key] * 1.0 / self.line_index
        for key in trans_dict:  # 状态转移概率
            for key1 in trans_dict[key]:
                trans_dict[key][key1] = trans_dict[key][key1] / Count_dict[key]
        for key in emit_dict:  # 发射概率(状态->词语的条件概率)
            for word in emit_dict[key]:
                emit_dict[key][word] = emit_dict[key][word] / Count_dict[key]
    
       # print(emit_dict)#{'S': {'否': 6.211504202703743e-06, '昔': 1.5528760506759358e-05, '直': 0.0001478338000243491,
       # print(trans_dict)#{'S': {'S': 0.46460125869921165, 'E': 0.0, 'M': 0.0, 'B': 0.3503213832274479},
       # print(start_dict)#{'S': 0.41798844132394497, 'E': 0.0, 'M': 0.0, 'B': 0.5820149148537713}
        self.save_model(trans_dict, trans_path)
        self.save_model(emit_dict, emit_path)
        self.save_model(start_dict, start_path)
    
        return trans_dict, emit_dict, start_dict

    



if __name__ == "__main__":
    train_filepath = './data/train.txt'
    trans_path = './model/prob_trans.model'
    emit_path = './model/prob_emit.model'
    start_path = './model/prob_start.model'
    trainer = HmmTrain()
    trainer.train(train_filepath, trans_path, emit_path, start_path)


