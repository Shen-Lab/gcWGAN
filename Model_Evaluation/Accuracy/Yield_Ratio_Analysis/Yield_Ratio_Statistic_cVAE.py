import DataLoading
import pickle
import numpy as np
import os

DATA_DIR = '../../../Data/Datasets/Final_Data/'
fold_train = DataLoading.file_list(DATA_DIR + 'unique_fold_train') #SZ add
fold_vali = DataLoading.file_list(DATA_DIR + 'fold_val') #SZ add
fold_test = DataLoading.file_list(DATA_DIR + 'fold_test')  #SZ add

path = '../../../Results/Accuracy/Yield_Ratio_Result/cVAE/YR_statistic/'

yr_file = open(path + 'yr_dic','rb')
yr_dic = pickle.load(yr_file)
yr_file.close()

def yr_sort(dic,kind,path):
    sort_list = sorted(dic.items(),key = lambda item:item[1][1],reverse = True)
    fil = open(path + kind + '/yr_sorted_' + kind,'w')
    for i in sort_list:
        fil.write(i[0] + '\t' + i[1][0] + '\t' + str(i[1][1]) + '\n')
    fil.close()

def yr_stat(dic,kind,path):
    yr_all = float(sum(i[1][1] for i in dic.items()))/float(len(dic.keys()))
    stat_dic = {}.fromkeys([chr(i) for i in range(97,104)] + ['easy','medium','hard'])
    print len(dic.keys())
    for f in dic.keys():
        if stat_dic[f[0]] == None:
            stat_dic[f[0]] = [dic[f][1]]
        else:
            stat_dic[f[0]].append(dic[f][1])
        if stat_dic[dic[f][0]] == None:
            stat_dic[dic[f][0]] = [dic[f][1]]
        else:
            stat_dic[dic[f][0]].append(dic[f][1])
    fil = open(path + kind + '/yr_stat_' + kind,'w')
    fil.write('aver YR: ' + str(yr_all) + '\t' + str(len(dic.keys()))  + '\n')
    for i in stat_dic.keys():
        #print len(stat_dic[i])
        fil.write(i +': ' + str(np.mean(stat_dic[i])) + '\t' + str(len(stat_dic[i])) + '\n')
    fil.close()

def yr_process(dic,fold_list,kind,path):
    if not (kind in os.listdir(path)):
        os.system('mkdir ' + path + kind)
    used_dic = {}.fromkeys(fold_list)
    for f in fold_list:
        used_dic[f] = [dic[f][0],float(dic[f][1])]
    yr_sort(used_dic,kind,path)
    yr_stat(used_dic,kind,path)

yr_process(yr_dic,fold_train,'train',path)
yr_process(yr_dic,fold_test,'test',path)
yr_process(yr_dic,fold_vali,'vali',path)

