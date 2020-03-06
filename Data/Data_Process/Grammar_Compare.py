##############################################################
# Compare the grammar assigned to gcWGAN folds with different
# methods and the previous result.
# 10-25-2019
# Input: path of the Compare File
# Output: Compare File
##############################################################

import sys

output = sys.argv[1]

################### Load Previous Result #####################

fold_list = []

pre_dic = {}

pre_file = open('../Datasets/cVAE_Data/Previous_mapping/all_folds_map','r')
pre_lines = pre_file.readlines()
pre_file.close()

for line in pre_lines:
    line = line.strip('\n').split('\t')
    fold = line[0]
    fold_list.append(fold)
    map_pdb = line[3].strip('.pdb')
    grammar = line[-1]
    pre_dic[fold] = [map_pdb,grammar]

print 'Previous Length:',len(pre_dic.keys())

############# Load Symmetric TMscore Result ###################

sym_dic = {}

sym_file = open('../Datasets/cVAE_Data/gcWGAN_cVAE_mapping_all_all','r')
sym_lines = sym_file.readlines()
sym_file.close()

for line in sym_lines[1:]:
    line = line.strip('\n').split('\t')
    fold = line[0]
    map_pdb = line[3].strip('.pdb')
    grammar = ''.join(line[-1].split('.'))
    sym_dic[fold] = [map_pdb,grammar]

print 'Sym Length:',len(sym_dic.keys())

################## Load Ref TMscore Result ####################

ref_dic = {}

ref_file = open('../Datasets/cVAE_Data/gcWGAN_cVAE_mapping_all_all_ref','r')
ref_lines = ref_file.readlines()
ref_file.close()

for line in ref_lines[1:]:
    line = line.strip('\n').split('\t')
    fold = line[0]
    map_pdb = line[3].strip('.pdb')
    grammar = ''.join(line[-1].split('.'))
    ref_dic[fold] = [map_pdb,grammar]

print 'Ref Length:',len(ref_dic.keys())

################## Load Rank and Yied Ratio ###################

yr_dic = {'nov':'-'}
rank_dic = {'nov':'-'}

DIR = '../../Results/Accuracy/Yield_Ratio_Result/gcWGAN/model_0.0001_5_64_1.0_semi_diff_100/'
kind_list = ['train','vali','test']
for kind in kind_list:
    yr_file = open(DIR + kind + '/' + 'yield_ratio_sort_0.0001_5_64_1.0_semi_diff_100_fold_' + kind + '.fa','r')
    yr_lines = yr_file.readlines()
    yr_file.close()

    for line in yr_lines:
        line = line.strip('\n').split('\t')
        fold = line[0]
        rank = line[1]
        yr = line[2]
        yr_dic[fold] = yr
        rank_dic[fold] = rank

print 'YR Length:',len(yr_dic.keys())
print 'Rank Length:',len(rank_dic.keys())

###################### Write Compare File #####################

print 'Fold List Length:',len(fold_list)
file_w = open(output,'w')
file_w.write('fold\trank\tpre_sym\tpre_ref\tsym_def\tyr\t')
file_w.write('pre_map\tpre_grammar\tsym_map\tsym_grammar\tref_map\tref_grammar\n')

for fold in fold_list:
    file_w.write(fold + '\t')
    ws_1 = str((pre_dic[fold][1] == sym_dic[fold][1]))
    ws_2 = str((pre_dic[fold][1] == ref_dic[fold][1]))
    ws_3 = str((ref_dic[fold][1] == sym_dic[fold][1]))
    file_w.write(rank_dic[fold] + '\t')
    file_w.write(ws_1 + '\t' + ws_2 + '\t' + ws_3 + '\t')
    file_w.write(yr_dic[fold] + '\t')
    file_w.write(pre_dic[fold][0] + '\t' + pre_dic[fold][1] + '\t')
    file_w.write(sym_dic[fold][0] + '\t' + sym_dic[fold][1] + '\t')
    file_w.write(ref_dic[fold][0] + '\t' + ref_dic[fold][1] + '\n')

file_w.close()

