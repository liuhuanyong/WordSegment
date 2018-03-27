# WordSegment
Chinese WordSegment based on algorithms including Maxmatch (forward, backward, bidirectional), HMM etc...

# 项目介绍  
1、MaxMatch：    
   dict.txt: 分词用词典位置  
   max_forward_cut：正向最大匹配分词    
   max_backward_cut：逆向最大匹配分词    
   max_biward_cut：双向最大匹配分词  
   result:  
   输入：我们在野生动物园玩  
   输出：  
    forward_cutlist:  ['我们', '在野', '生动', '物', '园', '玩']  
    backward_cutlist:  ['我们', '在', '野生', '动物园', '玩']  
    biward_seglit:  ['我们', '在', '野生', '动物园', '玩']  
    
2、HMM:  
   hmm_train.py:基于人民日报语料29W句子，训练初始状态概率，发射概率，转移概率  
             data:训练语料，放在 ./data/train.txt  
             model: 保存训练的概率模型，训练完成后可直接调用     
                  trans_path = './model/prob_trans.model'  
                  emit_path = './model/prob_emit.model'  
                  start_path = './model/prob_start.model'  
                
   hmm_cut.py:基于训练得到的model，结合viterbi算法进行分词    
           输入：我们在野生动物园玩  
           输出：['我们', '在', '野', '生动', '物园', '玩']  
           
3、N-gram  
   train_ngram.py:基于人民日报语料29W句子，训练词语出现概率，2-gram条件概率  
              data: 训练语料，放在 ./data/train.txt    
              model: 保存概率模型，训练完成后可直接调用  
              word_path = './model/word_dict.model' (词语出现概率）   
              trans_path = './model/trans_dict.model'（2-gram条件概率）  
   max_ngram.py: 最大化概率2-gram分词算法  
   biward_ngram.py: 基于ngram的前向后向最大匹配算法  
   
 4、算法比较  
   
   
