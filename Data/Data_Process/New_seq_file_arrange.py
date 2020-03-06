########################################################
# Arrange the validation and test sequences with their 
# labels in related files.
########################################################

import DataLoading

DATA_DIR = '../Datasets/Final_Data/'

inter_dic = DataLoading.Interval_dic(DATA_DIR + 'Interval_1.fa') 
FOLD_VALI = DataLoading.file_list(DATA_DIR + 'fold_val')
FOLD_TEST = DataLoading.file_list(DATA_DIR + 'fold_test')

def seq_fold_file_arrange(fold_list,inter_dict,seq_path,fold_path):
    seq_file = open(seq_path,'w')
    fold_file = open(fold_path,'w')
    for f in fold_list:
        seqs = inter_dict[f][1:]
        if len(seqs) != inter_dict[f][0]:
            print 'Error! Sequences amount does not match the labels amount!'
            quit()
        for s in seqs:
            if len(s) > 160:
                print f,'Too long a sequences!'
            elif len(s) < 60:
                print f,'Too short a sequences!'
            seq_file.write(s + '\n')
        for i in range(inter_dict[f][0]):
            fold_file.write(f + '\n')
    seq_file.close()
    fold_file.close()
    return 0

seq_fold_file_arrange(FOLD_VALI,inter_dic,DATA_DIR + 'seq_vali',DATA_DIR + 'fold_label_vali')
seq_fold_file_arrange(FOLD_TEST,inter_dic,DATA_DIR + 'seq_test',DATA_DIR + 'fold_label_test')
