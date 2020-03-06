import os
import pickle
import sys

path = '../Datasets/cVAE_Data/cVAE_pdbs/'

dire_dic = {}
corr_dic = {}
dire_pdb_list = []
corr_pdb_list = []
dire_grammar_list = []
corr_grammar_list = []

f_dire = open(path + 'assign_directly.txt','r')
f_tms = open(path + 'assign_by_tmscores.txt','r')
lines_dire = f_dire.readlines()
lines_tms = f_tms.readlines()
f_dire.close()
f_tms.close()

#num = 0

for line in lines_dire[1:]:
    line = [i for i in line.strip('\n').split(' ') if i != '']
    pdb = line[0]
    grammar = line[-1]
    dire_pdb_list.append(pdb)
    if grammar in dire_grammar_list:
        same_p = []
        for p in dire_dic.keys():
            if dire_dic[p] == grammar:
                same_p.append(p)
        print pdb,same_p,grammar
    dire_grammar_list.append(grammar)
    dire_dic[pdb] = grammar
    #if num < 5:
    #    print pdb,grammar
    #    num += 1
     
#print ''

for line in lines_tms[1:]:
    line = [i for i in line.strip('\n').split(' ') if i != '']
    pdb = line[0]
    pdb_corr = line[3]
    grammar = line[-1]
    corr_pdb_list.append(pdb_corr)
    corr_grammar_list.append(grammar)
    if (pdb_corr in corr_dic.keys()) and (corr_dic[pdb_corr] != grammar):
        print 'Error! Contradict grammar!'
    corr_dic[pdb_corr] = grammar
#    if num < 10:
#        print pdb_corr,grammar
#        num += 1

print ''

print 'dire_pdb:',len(dire_pdb_list),len(set(dire_pdb_list))
print 'corr_pdb:',len(corr_pdb_list),len(set(corr_pdb_list))
print 'dire_grammar:',len(dire_grammar_list),len(set(dire_grammar_list))
print 'corr_grammar:',len(corr_grammar_list),len(set(corr_grammar_list))

######################### Build cVAE grammar dictionary ###################
print ''
save_name = sys.argv[1]
all_dic = dict(dire_dic,**corr_dic)
print 'All_dic Size:',len(all_dic.keys())
save_file =  open(save_name,'wb') 
pickle.dump(all_dic,save_file)
save_file.close()


