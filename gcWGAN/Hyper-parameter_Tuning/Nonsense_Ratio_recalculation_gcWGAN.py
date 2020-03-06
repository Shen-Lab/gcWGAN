##########################################################################
# Recalculate the nonsense sequences ratio for different fold sets based on # pre-generated samples.
##########################################################################

import sys
import DataLoading
import os
import numpy as np

test_index = sys.argv[1]
fold_set = sys.argv[2]

sample_path = 'gcWGAN_Validation_Samples/NonsenseRatio_Sample_%s/'%test_index

DATA_DIR = '../../Data/Datasets/Final_Data/'

sample_title = 'sample_' + fold_set + '_'

if fold_set == 'train':
   fold_list = DataLoading.file_list(DATA_DIR + 'unique_fold_train')
elif fold_set == 'vali':
   fold_list = DataLoading.file_list(DATA_DIR + 'fold_val')
elif fold_set == 'test':
   fold_list = DataLoading.file_list(DATA_DIR + 'fold_test')
else:
   print 'No set named "%s"'%fold_set
   quit()

file_list = [fi for fi in os.listdir(sample_path) if sample_title in fi]
file_num = len(file_list)

result_path = 'gcWGAN_Validation_Results/'

if not os.path.exists(result_path):
   os.system('mkdir ' + result_path)

file_name = result_path + 'NR_reca_%s_%s.fa'%(test_index,fold_set)
file_result = open(file_name,'w')
file_result.close()

for i in range(1,file_num + 1):
    with open(sample_path + sample_title + str(i) + '.fa','r') as sample_file:
        lines = sample_file.readlines()
    sample_dic = {}
    for line in lines:
        if line != '\n':
            line = line.strip('\n').split(': ')
            fold = line[0]
            if fold in sample_dic.keys():
                sample_dic[fold][0] += 1
            else:
                sample_dic[fold] = [1,0]
            
            seq = line[-1]
            if len(seq) > 0:
                while(seq[0] == ' '):
                    seq = seq[1:]

            if seq[0] == '!':
                sample_dic[fold][1] += 1
            else:
                seq = seq.strip('!')
                if '!' in seq or seq == '':
                    sample_dic[fold][1] += 1
    
    for f in fold_list:
        if sample_dic[f][0] != 100:
            print 'Error! Sequence amount error!'
            break
    if set(fold_list) != set(sample_dic.keys()):
        print 'Error! Fold set error!'

    NR_list = [float(sample_dic[f][1])/sample_dic[f][0] for f in fold_list]
    with open(file_name,'a') as file_result:
        file_result.write(str(np.mean(NR_list)) + '\n')


                         







