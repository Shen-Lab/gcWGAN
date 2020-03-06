##########################################################
# Record the alignments between the target fold and the 
# folds in the input file.
##########################################################

import sys
import numpy as np
import DataLoading
import TM_helper

FOLD = sys.argv[1]
FILE = sys.argv[2]
PATH = sys.argv[3]

if not PATH.endswith('/'):
    PATH += '/'

if len(sys.argv) > 4:
    ORDER = int(sys.argv[4])
else:
    ORDER = 0

DATA_DIR = '../Datasets/Final_Data/'

if 'nov' in FOLD.lower():
    fold_name = 'nov'
    fold_pdb = DATA_DIR + 'Case_Groundtruth/novel_groundtruth.pdb'
else:
    fold_name = '_'.join(FOLD.split('.'))
    fold_pdb = DATA_DIR + 'pdbs_withName/' + fold_name + '.pdb'

fold_list = DataLoading.columns_to_lists(FILE)[0]

for fo in fold_list:
    print fo
    fo_name = '_'.join(fo.split('.'))
    if ORDER == 0:
        pdb_1 = DATA_DIR + 'pdbs_withName/' + fo_name + '.pdb'
        pdb_2 = fold_pdb
        index = PATH + fo_name + '_and_' + fold_name
    else:
        pdb_1 = fold_pdb
        pdb_2 = DATA_DIR + 'pdbs_withName/' + fo_name + '.pdb'
        index = PATH + fold_name + '_and_' + fo_name
    print pdb_1
    print pdb_2
    print index
    TM_helper.Alignment(pdb_1,pdb_2,index)








