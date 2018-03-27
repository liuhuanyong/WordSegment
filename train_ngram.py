#!/usr/bin/env python3
# coding: utf-8
# File: train_ngram.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-3-27

class TrainNgram():
    def __init__(self):
        self.word_dict = {}  # 词语频次词典
        self.transdict = {}  # 每个词后接词的出现个数

    '''训练ngram参数'''
    def train(self, train_data_path, wordict_path, transdict_path):
        print('start training...')
        self.transdict[u'<BEG>'] = {}
        self.word_dict['<BEG>'] = 0
        
        for sentence in open(train_data_path):
            self.word_dict['<BEG>'] += 1
            sentence = sentence.strip()
            sentence = sentence.split(' ')
            sentence_list = []
            # ['７月１４日', '', '下午４时', '', '，', '', '渭南市', '', '富平县庄里粮站', '', '。'], 得到每个词出现的个数
            for pos, words in enumerate(sentence):
                if words != '':
                    sentence_list.append(words)
            # ['７月１４日', '下午４时', '渭南市', '富平县庄里粮站']
            for pos, words in enumerate(sentence_list):
                if words not in self.word_dict.keys():
                    self.word_dict[words] = 1
                else:
                    self.word_dict[words] += 1
                # 词频统计
                # 得到每个词后接词出现的个数，bigram <word1, word2>
                words1, words2 = '', ''
                # 如果是句首，则为<BEG，word>
                if pos == 0:
                    words1, words2 = u'<BEG>', words
                # 如果是句尾，则为<word, END>
                elif pos == len(sentence_list) - 1:
                    words1, words2 = words, u'<END>'
                # 如果非句首，句尾，则为 <word1, word2>
                else:
                    words1, words2 = words, sentence_list[pos + 1]
                # 统计当前词后接词语出现的次数：{‘我’：{‘是’：1， ‘爱’：2}}
                if words not in self.transdict.keys():
                    self.transdict[words1] = {}
                if words2 not in self.transdict[words1]:
                    self.transdict[words1][words2] = 1
                else:
                    self.transdict[words1][words2] += 1

        self.save_model(self.word_dict, wordict_path)
        self.save_model(self.transdict, transdict_path)

    '''保存模型'''
    def save_model(self, word_dict, model_path):
        f = open(model_path, 'w')
        f.write(str(word_dict))
        f.close()

if __name__ == '__main__':
    train_data_path = './data/train.txt'
    wordict_path = './model/word_dict.model'
    transdict_path = './model/trans_dict.model'
    trainer = TrainNgram()
    trainer.train(train_data_path, wordict_path, transdict_path)
