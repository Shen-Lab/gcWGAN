import os

path = '../Datasets/cVAE_Data/cVAE_pdbs/'

pdb_list = [i.strip('.pdb') for i in os.listdir(path) if i.endswith('.pdb')]

pdb_dire_list = []
pdb_tms_list = []
pdb_corresponding_list = []

f_dire = open(path + 'assign_directly.txt','r')
f_tms = open(path + 'assign_by_tmscores.txt','r')
lines_dire = f_dire.readlines()
lines_tms = f_tms.readlines()
f_dire.close()
f_tms.close()

num = 0

for line in lines_dire[1:]:
    pdb_dire_list.append(line.split(' ')[0])
    if num < 5:
        print line.split(' ')[0]
        num += 1

print ''

for line in lines_tms[1:]:
    line = [i for i in line.strip('\n').split(' ') if i != '']
    pdb = line[0]
    pdb_corr = line[3]
    pdb_tms_list.append(pdb)
    pdb_corresponding_list.append(pdb_corr)
    if num < 10:
        print pdb,pdb_corr
        num += 1

print ''

pdb_set = set(pdb_list)
pdb_dire_set = set(pdb_dire_list)
pdb_tms_set = set(pdb_tms_list)
pdb_corr_set = set(pdb_corresponding_list)

print 'pdb_set:',len(pdb_set)
print 'pdb_dire_set:',len(pdb_dire_set)
print 'pdb_tms_set:',len(pdb_tms_set)
print 'pdb_corr_set:',len(pdb_corr_set)
print 'tms and dire:',len(pdb_tms_set & pdb_dire_set)
print 'corr and dire:',len(pdb_corr_set & pdb_dire_set)
print 'pdb and coor:',len(pdb_corr_set & pdb_set)
print 'pdb and dire:',len(pdb_set & pdb_dire_set)
print 'pdb and (dire + corr):',len(pdb_set & (pdb_dire_set | pdb_corr_set))
print ''


if len(pdb_set) != len(pdb_list):
    print 'pdb repetition!'
if len(pdb_dire_set) != len(pdb_dire_list):
    print 'pdb_dire repetition!'
if len(pdb_tms_set) != len(pdb_tms_list):
    print 'pdb_tms repetition!'
if len(pdb_corr_set) != len(pdb_corresponding_list):
    print 'pdb_corr repetition!'

if not pdb_corr_set <= pdb_dire_set:
    print 'pdb_corr out of pdb_dire_list'
    print len(pdb_corr_set - pdb_dire_set)
else:
    print 'pdb_corr belong to pdb_dire_list'

print ''

if not pdb_dire_set <= pdb_set:
    print 'pdb_dire out of pdb_list'
    print len(pdb_dire_set - pdb_set)
else:
    print 'pdb_dire belong to pdb_list'

print ''

if not pdb_set <= ((pdb_tms_set | pdb_dire_set)):
    print 'pdb out of tms_or_dire_list'
    print len(pdb_set - (pdb_tms_set | pdb_dire_set))
    if not pdb_set <= ((pdb_tms_set | pdb_dire_set)|pdb_corr_set):
        print 'pdb out of pdb_grammar_list'
        print len(pdb_set - ((pdb_tms_set | pdb_dire_set)|pdb_corr_set))
    else:
        print 'pdb belong to pdb_grammar_list'
else:
    print 'pdb belong to tms_or_dire_list'

if pdb_set == (pdb_corr_set | pdb_dire_set):
    print 'pbd equal to dire | corr'
