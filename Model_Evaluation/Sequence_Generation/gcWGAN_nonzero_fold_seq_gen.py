####################################################
# Generate gcWGAN sequences for the folds with 
# non-zero yield ratio.
####################################################

import sys
import os

KIND = sys.argv[1].lower()
NUM = sys.argv[2]

if not KIND in ['train','vali','test']:
    print 'Error! Wrong kind!'
    quit()

path = '../../Results/Accuracy/Yield_Ratio_Result/gcWGAN/model_0.0001_5_64_0.02_semi_diff_100/%s/yield_ratio_sort_0.0001_5_64_0.02_semi_diff_100_fold_%s.fa'%(KIND,KIND)

with open(path,'r') as yr_file:
    lines = yr_file.readlines()

fold_list = []
for line in lines:
    line = line.strip('\n').split('\t')
    fold = line[0]
    yr = float(line[2])
    if yr > 0:
        fold_list.append(fold)

print '%d folds with non-zero yield ratio.'%(len(fold_list))

for fold in fold_list:
    comment = './wrapper_seq_gen.sh gcWGAN %s Success %s'%(fold,NUM)
    os.system(comment)
    
