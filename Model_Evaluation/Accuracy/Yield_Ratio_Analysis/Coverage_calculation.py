#########################################################################################
# Find out the portion of the folds with a non-zero yield ratio.
# 11/12/2019
# Input: the path of sorted yield ratio file
# Output: the coverage statistic file in the same folder as the sorted yield ratio file
#########################################################################################

import sys
import DataLoading

sort_file_name = sys.argv[1]
fold_list,rank_list,yr_list = DataLoading.columns_to_lists(sort_file_name)

path = '/'.join(sort_file_name.split('/')[:-1]) + '/'

num = len(fold_list)
class_dict = {'All':[0,0],'a':[0,0],'b':[0,0],'c':[0,0],'d':[0,0],'e':[0,0],'f':[0,0],'g':[0,0],'easy':[0,0],'medium':[0,0],'hard':[0,0]}
for i in range(num):
    fold = fold_list[i]
    rank = rank_list[i]
    yr = float(yr_list[i])
    class_dict['All'][0] += 1
    class_dict[fold[0]][0] += 1
    class_dict[rank][0] += 1
    if yr > 0:
        class_dict['All'][1] += 1
        class_dict[fold[0]][1] += 1
        class_dict[rank][1] += 1

class_list = ['All','a','b','c','d','e','f','g','easy','medium','hard']

with open(path + 'coverage_result','w') as cov_file:
    cov_file.write('Class\tfold_num\tnonzero_num\tcoverage\n')
    for clas in class_list:
        cov_file.write(clas + '\t')
        cov = float(class_dict[clas][1])/float(class_dict[clas][0])
        cov_file.write(str(class_dict[clas][0]) + '\t' + str(class_dict[clas][1]) + '\t')
        cov_file.write(str(cov) + '\n')


