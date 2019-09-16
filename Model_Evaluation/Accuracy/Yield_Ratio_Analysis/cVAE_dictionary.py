import DataLoading
import sys
import os
import pickle

DATA_DIR = '../../../Data/Datasets/Final_Data/'
seq_avail = DataLoading.columns_to_lists(DATA_DIR + 'seq_availability')

path = '../../../Results/Accuracy/Yield_Ratio_Result/cVAE/'
PATH = path + 'YR_statistic/'

if not os.path.exists('../../../Results'):
    os.system('mkdir ../../../Results')
if not os.path.exists('../../../Results/Accuracy'):
    os.system('mkdir ../../../Results/Accuracy')
if not os.path.exists('../../../Results/Accuracy/Yield_Ratio_Result'):
    os.system('mkdir ../../../Results/Accuracy/Yield_Ratio_Result')
if not os.path.exists(PATH):
    os.system('mkdir ' + PATH)
if not os.path.exists(PATH):
    os.system('mkdir ' + PATH)


fold_list = os.listdir(path)
fold_dic = {}
for f in fold_list:
    if '.' in f:
        info = DataLoading.columns_to_lists(path + f)
    	index = seq_avail[0].index(f)
    	sa = int(seq_avail[1][index])
    	if sa > 50:
            rank = 'easy'
    	elif sa <= 5:
            rank = 'hard'
        else:
            rank = 'medium'
        fold_dic[f] = [rank,info[3][0]]    

f_file = open(PATH + 'yr_dic','wb')
pickle.dump(fold_dic,f_file)
f_file.close()

