##########################################################
# Find the neighbor folds of the input fold that the L2 
# distances are less than the input threshold.
##########################################################

import sys
import numpy as np
import data_helpers
import TM_helper

FOLD = sys.argv[1]
THRESHOLD = float(sys.argv[2])

if len(sys.argv) >= 4:
    FILE = sys.argv[3]
    Write_flag = True
else:
    Write_flag = False

DATA_DIR = '../Datasets/Final_Data/'

fold_dict = {}

repre_path = '../Datasets/Origin_SCOPE/represent_file'
with open(repre_path,'r') as repre_file:
    repre_lines = repre_file.readlines()

for i in range(len(repre_lines)):
    fold = repre_lines[i].split(' ')[1].split('.')
    fold = fold[0] + '.' + fold[1]
    fold_dict[fold] = DATA_DIR + '/pdbs/' + str(i+1) + '.pdb'

if 'nov' in FOLD.lower():
    fold_pdb = DATA_DIR + 'Case_Groundtruth/novel_groundtruth.pdb'
else:
    fold_pdb = fold_dict[FOLD]

neighbor_list = []

for f in fold_dict.keys():
    simi = TM_helper.TM_score(fold_pdb,fold_dict[f])
    if (f != FOLD) and (simi >= THRESHOLD):
        print f,simi
        neighbor_list.append((f,simi))

if Write_flag:
    with open(FILE,'w') as ne_file:
        for nf in neighbor_list:
            ne_file.write(nf[0] + '\t' + str(nf[1]) + '\n')
        
