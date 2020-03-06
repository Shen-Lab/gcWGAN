##############################################################
# Help to select new representative folds (6 cases) based on 
# previous and new results.
# 10-25-2019
# Input: path of the Statistic File
# Output: Statistic File
##############################################################

import sys
import DataLoading

output = sys.argv[1]

####### Load Previous Results and Selected Grammar ###########

fold_list = ['nov','a.35','b.2','c.56','c.94','d.107','g.44']

pre_dic = {}

pre_file = open('../Datasets/cVAE_Data/Previous_mapping/all_folds_map','r')
pre_lines = pre_file.readlines()
pre_file.close()

for line in pre_lines:
    line = line.strip('\n').split('\t')
    fold = line[0]
    map_pdb = line[3].strip('.pdb')
    grammar = line[-1]
    pre_dic[fold] = [map_pdb,grammar]

Pre_map = []
Pre_grammar = []

for f in fold_list:
    Pre_map.append(pre_dic[f][0])
    Pre_grammar.append(pre_dic[f][1])

pre_check_file = open('../Datasets/cVAE_Data/Previous_mapping/Case_fold_grammar','r')
Pre_Check = [i.strip('\n') for i in pre_check_file.readlines()]
pre_check_file.close()

if Pre_grammar == Pre_Check:
    print 'Previous Grammar Checked.'
    print ''
else:
    print 'Previous Grammar Error!'
    print Pre_grammar
    print Pre_Check
    quit()

grammar_sym_dic = {}
grammar_ref_dic = {}
for g in Pre_grammar:
    grammar_sym_dic[g] = []
    grammar_ref_dic[g] = []

############# Load Symmetric TMscore Result ###################

DATA_DIR = '../Datasets/Final_Data/'
fold_test = DataLoading.file_list(DATA_DIR + 'fold_test')

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
    if (grammar in Pre_grammar) and (fold in fold_test):
        grammar_sym_dic[grammar].append(fold)

print 'Sym Loaded.'

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
    if (grammar in Pre_grammar) and (fold in fold_test):
        grammar_ref_dic[grammar].append(fold)

print 'Ref Loaded.'

################## Load Rank and Yield Ratio ###################

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

print 'Rank and Yield Ratio Yield Loaded.'

#################### Write Statistic File #####################

file_w = open(output,'w')

for i in range(len(fold_list)):
    fold = fold_list[i]
    map_pre = Pre_map[i]
    grammar_pre = Pre_grammar[i]

    file_w.write(fold + '\t' + map_pre + '\t' + grammar_pre + '\n')

    file_w.write('Symmetric TMscore:\n')
    for f in grammar_sym_dic[grammar_pre]:
        file_w.write(f + '\t' + rank_dic[f] + '\t' + yr_dic[f] + '\t')
        file_w.write(sym_dic[f][0] + '\t' + sym_dic[f][1] + '\n')
    
    file_w.write('Direct TMscore:\n')
    for f in grammar_ref_dic[grammar_pre]:
        file_w.write(f + '\t' + rank_dic[f] + '\t' + yr_dic[f] + '\t') 
        file_w.write(ref_dic[f][0] + '\t' + ref_dic[f][1] + '\n')
    
    file_w.write('Both Satisfied:\n')
    for f in grammar_sym_dic[grammar_pre]:
        if f in grammar_ref_dic[grammar_pre]:
            file_w.write(f + '\t' + rank_dic[f] + '\t' + yr_dic[f] + '\t') 
            file_w.write(sym_dic[f][0] + '\t' + ref_dic[f][0] + '\n') 
    
    file_w.write('\n')
file_w.close()

