##########################################################
# Find the neighbor folds of the input fold that the L2 
# distances are less than the input threshold.
##########################################################

import sys
import numpy as np
import data_helpers

FOLD = sys.argv[1]
THRESHOLD = float(sys.argv[2])

if len(sys.argv) >= 4:
    FILE = sys.argv[3]
    Write_flag = True
else:
    Write_flag = False

DATA_DIR = '../Datasets/Final_Data/'
seqs, folds, folds_dict, charmap, inv_charmap = data_helpers.load_dataset_protein( #MK change
    max_length=160,
    max_n_examples=50000,
    data_dir=DATA_DIR
)

if 'nov' in FOLD.lower():
    fold_array = np.loadtxt(DATA_DIR + 'novel_coordinate')
else:
    fold_array = np.array(folds_dict[FOLD])

neighbor_list = []

for f in folds_dict.keys():
    dist = np.linalg.norm(fold_array - np.array(folds_dict[f]))
    if (f != FOLD) and (dist <= THRESHOLD):
        print f,dist
        neighbor_list.append((f,dist))

if Write_flag:
    with open(FILE,'w') as ne_file:
        for nf in neighbor_list:
            ne_file.write(nf[0] + '\t' + str(nf[1]) + '\n')
        
