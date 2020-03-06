######################################################################
# Check whether the folds with the same pdb name were mapped.
# 10-25-2019
######################################################################

import os 
import pickle 

cVAE_grammar_file = open('../Datasets/cVAE_Data/cVAE_grammar_dict','rb')
cVAE_grammar_dic = pickle.load(cVAE_grammar_file)
cVAE_grammar_file.close()

##################### Load Overlap Data ##############################

overlap_dic = {}
overlap_file_list = [i for i in os.listdir('../Datasets/cVAE_Data/Previous_mapping/') if 'overlap' in i]

for of in overlap_file_list:
    print of
    overlap_file = open('../Datasets/cVAE_Data/Previous_mapping/' + of,'r')
    ovl_lines = overlap_file.readlines()
    overlap_file.close()
    for line in ovl_lines:
        line = line.strip('\n').split('\t')
        fold = line[0]
        if line[3] != line[1]:
            print 'Not overlap:',fold
        map_pdb = line[3].strip('.pdb')
        grammar = line[-1]
        overlap_dic[fold] = [map_pdb,grammar]

print 'Size:',len(overlap_dic.keys())
print 'Overlap Data Loaded.'

##################### Symmetric TMscore Check ########################

print 'Symmetric TMscore Check:'

sym_list = []

gram_file = open('../Datasets/cVAE_Data/gcWGAN_cVAE_mapping_all_all','r')
info_lines = gram_file.readlines()[1:]
gram_file.close()

print 'Size:',len(info_lines)

num = 0
diff_num = 0
for line in info_lines:
    line = line.strip('\n').split('\t')
    fold = line[0]
    map_pdb = line[3]
    grammar_ori = line[-1]
    grammar = ''.join(line[-1].split('.'))
    if fold in overlap_dic.keys():
        num += 1 
        if (map_pdb != overlap_dic[fold][0]) or (grammar != overlap_dic[fold][1]):
            if grammar_ori != cVAE_grammar_dic[map_pdb]:
                sym_list.append(fold)
                print fold
                print '  ',map_pdb,overlap_dic[fold][0]
                print '  ',grammar,overlap_dic[fold][1]
                diff_num += 1

print 'Checked Num:',num
print 'Different Num:',diff_num

print 'Done'

##################### Ref TMscore Check #############################

print 'Ref TMscore Check:'

ref_list = []

ref_gram_file = open('../Datasets/cVAE_Data/gcWGAN_cVAE_mapping_all_all_ref','r')
ref_info_lines = ref_gram_file.readlines()[1:]
ref_gram_file.close()

print 'Size:',len(ref_info_lines)

num = 0
diff_num = 0

for line in ref_info_lines:
    line = line.strip('\n').split('\t')
    fold = line[0]
    map_pdb = line[3]
    grammar_ori = line[-1]
    grammar = ''.join(line[-1].split('.'))
    if fold in overlap_dic.keys():
        num += 1
        if (map_pdb != overlap_dic[fold][0]) or (grammar != overlap_dic[fold][1]):     
            if grammar_ori != cVAE_grammar_dic[map_pdb]:
                ref_list.append(fold)
                print fold
                print '  ',map_pdb,overlap_dic[fold][0]
                print '  ',grammar,overlap_dic[fold][1]
                diff_num += 1

print 'Checked Num:',num
print 'Different Num:',diff_num

print 'Done',sym_list == ref_list

