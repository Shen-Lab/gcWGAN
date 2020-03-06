##########################################################
# Find the average number of neighbor folds that the L2 
# distances are less than the input threshold.
##########################################################

import sys
import numpy as np
import data_helpers

THRESHOLD = float(sys.argv[1])

DATA_DIR = '../Datasets/Final_Data/'
seqs, folds, folds_dict, charmap, inv_charmap = data_helpers.load_dataset_protein( #MK change
    max_length=160,
    max_n_examples=50000,
    data_dir=DATA_DIR
)

print 'Treshold:',THRESHOLD
print '%d folds condisdered.'%len(folds_dict.keys())
print ''

neighbor_num = []

for FOLD in folds_dict.keys():
    num = 0
    fold_array = np.array(folds_dict[FOLD])
    for f in folds_dict.keys():
        dist = np.linalg.norm(fold_array - np.array(folds_dict[f]))
        if (f != FOLD) and (dist <= THRESHOLD):
            num += 1
    neighbor_num.append(num)

print '%d folds considered.'%len(neighbor_num)

print 'Average amount of neignbor folds:',np.mean(neighbor_num)

        
