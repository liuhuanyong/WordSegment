from __future__ import division
from hmm_cut import *
from maxmatch import *
from max_ngram import *
from biward_ngram import *
import time

hmm_cuter = HmmCut()
maxmatch_cuter = CutWords()
maxngram_cuter = MaxProbCut()
biwardngram_cuter = BiWardNgram()


def score(testfile, mode):
	start_time = time.time()
	count = 1
	count_right = 0
	count_split = 0
	count_gold = 0
	process_count = 0
	with open(testfile) as f:
		for line in f:
			process_count += 1
			if process_count % 1000 == 0:
				print(process_count)
			line = line.strip()
			goldlist = line.split(' ')
			sentence = line.replace(' ','')
			if mode == 'hmm':
				inlist = hmm_cuter.cut(sentence)
			elif mode == 'forward':
				inlist = maxmatch_cuter.max_forward_cut(sentence)
			elif mode == 'backward':
				inlist = maxmatch_cuter.max_backward_cut(sentence)
			elif mode == 'biward':
				inlist = maxmatch_cuter.max_biward_cut(sentence)
			elif mode == 'maxngram':
				inlist = maxngram_cuter.cut(sentence)
			elif mode == 'biwardngram':
				try:
					inlist = biwardngram_cuter.cut(sentence)
				except:
					pass
			count += 1
			count_split += len(inlist)
			count_gold += len(goldlist)
			tmp_in = inlist
			tmp_gold = goldlist

			for key in tmp_in:
				if key in tmp_gold:
					count_right += 1
					tmp_gold.remove(key)

		P = count_right / count_split
		R = count_right / count_gold
		F = 2 * P * R / (P + R)

	end_time = time.time()
	cost = (end_time - start_time)
	print(P, R, F, cost)

	return P, R, F, cost


if __name__ == "__main__":
	testfile = './data/test.txt'
	#P, R, F, cost = score(testfile, 'hmm')
	#P, R, F, cost = score(testfile, 'forward')
	#P, R, F, cost = score(testfile, 'backward')
	#P, R, F, cost = score(testfile, 'biward')
	#P, R, F, cost = score(testfile, 'maxngram')
	#P, R, F, cost = score(testfile, 'biwardngram')


		

