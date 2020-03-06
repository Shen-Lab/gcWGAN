#############################################################################################
# Find out the average yield ratio of the folds in different oracle accuracy range.
# 11/12/2019
# Input: the path of sorted yield ratio file
# Output: the statistic file that contains the average yield ratio of the folds in different 
#         oracle accuracy rangein the same folder as the sorted yield ratio file
#############################################################################################

import sys
import DataLoading
import numpy as np

sort_file_name = sys.argv[1]
path = '/'.join(sort_file_name.split('/')[:-1]) + '/'
fold_list,rank_list,yr_list = DataLoading.columns_to_lists(sort_file_name)
oracle_data = DataLoading.columns_to_lists('../../../Oracle/DeepSF_modified/deepsf_accuracy.txt')
fold_ora = oracle_data[0][1:]
top_10_acc = [float(i) for i in oracle_data[4][1:]]

ora_fold_num = len(fold_ora)
fold_num = len(fold_list)

print 'Number of folds for the Oracle:',ora_fold_num
print 'Number of folds in the processed set:',fold_num

Oracle_dict = {}
for i in range(ora_fold_num):
    Oracle_dict[fold_ora[i]] = top_10_acc[i]

yieldratio_dict = {}
for i in range(fold_num):
    yieldratio_dict[fold_list[i]] = float(yr_list[i])

class_dict = {'0~0.25:':[[0,0.25],[]],'0.25~0.5:':[[0.25,0.5],[]],'0.5~0.75:':[[0.5,0.75],[]],'0.75~1:':[[0.75,1.01],[]]}

class_list = ['0~0.25:','0.25~0.5:','0.5~0.75:','0.75~1:']

for fold in fold_list:
    ora_acc = Oracle_dict[fold]
    yr = yieldratio_dict[fold]
    for clas in class_list:
        lower_bound = class_dict[clas][0][0]
        upper_bound = class_dict[clas][0][1]
        if (ora_acc >= lower_bound) and (ora_acc < upper_bound):
            class_dict[clas][1].append(yr)

class_len = []  #Check whether all the folds in the processes set are included

with open(path + 'OracleAccuracy_YieldRatio_Stat','w') as ora_yr_file:
    ora_yr_file.write('Range\tAverage_yield_ratio\n')
    for clas in class_list:
        ora_yr_file.write(clas + '\t')
        aver_yr = np.mean(class_dict[clas][1])
        class_len.append(len(class_dict[clas][1]))
        print clas,'%d folds,'%len(class_dict[clas][1]),'average yield ratio is',aver_yr
        ora_yr_file.write(str(aver_yr) + '\n')

if sum(class_len) != fold_num:
    print 'Error! Missing some folds!'
