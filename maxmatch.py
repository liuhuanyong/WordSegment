#!/usr/bin/env python3
# coding: utf-8
# File: maxmatch.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-3-26

class CutWords:
    def __init__(self, dict_path, sent):
        self.dict_path = dict_path
        self.sent = sent
        
    #最大向前匹配
    def max_forward_cut(self, word_dict):
        #1.从左向右取待切分汉语句的m个字符作为匹配字段，m为大机器词典中最长词条个数。
        #2.查找大机器词典并进行匹配。若匹配成功，则将这个匹配字段作为一个词切分出来。
        window_size = 5
        cutlist = []
        index = 0
        while index < len(self.sent):
            matched = False
            for i in range(window_size, 0, -1):
                cand_word = self.sent[index : index + i]
                if cand_word in word_dict:
                    cutlist.append(cand_word)
                    matched = True
                    break
    
            #如果没有匹配上，则按字符切分
            if not matched:
                i = 1
                cutlist.append(self.sent[index])
            index += i
        return cutlist
    
    #最大向后匹配
    def max_backward_cut(self, word_dict):
        #1.从右向左取待切分汉语句的m个字符作为匹配字段，m为大机器词典中最长词条个数。
        #2.查找大机器词典并进行匹配。若匹配成功，则将这个匹配字段作为一个词切分出来。
        window_size = 5
        cutlist = []
        index = len(self.sent)
        while index > 0 :
            matched = False
            for i in range(window_size, 0, -1):
                tmp = (i + 1)
                cand_word = self.sent[index - tmp : index]
                #如果匹配上，则将字典中的字符加入到切分字符中
                if cand_word in word_dict:
                    cutlist.append(cand_word)
                    matched = True
                    break
            #如果没有匹配上，则按字符切分
            if not matched:
                tmp = 1
                cutlist.append(self.sent[index-1])
    
            index -= tmp
    
        return cutlist[::-1]

    # 双向最大向前匹配
    def max_biward_cut(self, forward_cutlist, backward_cutlist):
        # 双向最大匹配法是将正向最大匹配法得到的分词结果和逆向最大匹配法的到的结果进行比较，从而决定正确的分词方法。
        # 启发式规则：
        # 1.如果正反向分词结果词数不同，则取分词数量较少的那个。
        # 2.如果分词结果词数相同 a.分词结果相同，就说明没有歧义，可返回任意一个。 b.分词结果不同，返回其中单字较少的那个。
        count_forward = len(forward_cutlist)
        count_backward = len(backward_cutlist)
    
        def compute_single(word_list):
            num = 0
            for word in word_list:
                if len(word) == 1:
                    num += 1
            return num
    
        if count_forward == count_backward:
            if compute_single(forward_cutlist) > compute_single(backward_cutlist):
                return backward_cutlist
            else:
                return forward_cutlist
    
        elif count_backward > count_forward:
            return forward_cutlist
    
        else:
            return backward_cutlist
    
    #加载词典
    def load_words(self):
        words = list()
        for line in open(self.dict_path):
            words += line.strip().split(' ')
        return words
    
    #分词主函数
    def cut_main(self):
        print('loading words........')
        words = self.load_words()
        print('{0} words in total'.format(len(words)))
        forward_cutlist = self.max_forward_cut(words)
        #['我们', '在野', '生动', '物', '园', '玩']
        print('forward_cutlist: ', forward_cutlist)
        backward_cutlist = self.max_backward_cut(words)
        #['我们', '在', '野生', '动物园', '玩']
        print('backward_cutlist: ', backward_cutlist)
        biward_seglit = self.max_biward_cut(forward_cutlist, backward_cutlist)
        print('biward_seglit: ', biward_seglit)
        #['我们', '在', '野生', '动物园', '玩']

        return biward_seglit

if __name__=='__main__':
    dictpath = './dict/dict.txt'
    sent = '''我们在野生动物园玩'''
    sent = '''目前在自然语言处理技术中，中文处理技术比西文处理技术要落后很大一段距离，许多西文的处理方法中文不能直接采用，就是因为中文必需有分词这道工序。中文分词是其他中文信息处理的基础，搜索引擎只是中文分词的一个应用。'''
    #sent = '北京大学学生前来应聘'
    cuter = CutWords(dictpath, sent)
    cuter.cut_main()

