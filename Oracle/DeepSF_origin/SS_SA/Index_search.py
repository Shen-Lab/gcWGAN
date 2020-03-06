#######################################################################################
# Find out the index of the characters in AA, SS and SA for one-hot encoding. 
# 11/08/2019
# Input: the name of the sequence <seq_name> 
# Requirment: <seq_name>.fasta, <seq_name>.ss, <seq_name>.acc and <seq_name>.fea_aa_ss_sa
#             from deepsf, and the files should be in the correct path. The files can only
#             be related to one sequences.
#######################################################################################

import sys
import numpy as np

seq_name = sys.argv[1]
if '.' in seq_name:
    seq_name = seq_name.split('.')[0]

with open('test/seq_test/' + seq_name + '.fasta','r') as aa_file:
    aa_lines = aa_file.readlines()
    seq = ''
    for line in aa_lines[1:]:
        seq += line.strip('\n')

with open('test/index_search/' + seq_name + '.ss','r') as ss_file:
    ss_lines = ss_file.readlines()
    ss = ''
    for line in ss_lines[1:]:
        ss += line.strip('\n')

with open('test/index_search/' + seq_name + '.acc','r') as sa_file:
    sa_lines = sa_file.readlines()
    sa = ''
    for line in sa_lines[1:]:
        sa += line.strip('\n')

with open('test/aa_ss_sa_deepsf/' + seq_name + '.fea_aa_ss_sa','r') as fea_file:
    feature = fea_file.readlines()[1].strip('\n').split('\t')[-1]
    feature = [int(i.split(':')[-1]) for i in feature.split(' ') if i != '']
    feature = np.array(feature).reshape(-1,25)

print len(seq)
print len(ss)
print len(sa)
print feature.shape

aa_onehot = feature[:,0:20]
ss_onehot = feature[:,20:23]
sa_onehot = feature[:,23:]

length = len(seq)

aa_dict = {}
ss_dict = {}
sa_dict = {}
for i in range(length):
    #print list(aa_onehot[i]),len(list(aa_onehot[i]))
    if seq[i] in aa_dict.keys():
        if aa_dict[seq[i]] != list(aa_onehot[i]).index(1):
            print 'Error!',seq[i],i,aa_dict[seq[i]],list(aa_onehot[i]).index(1)
    else:
        aa_dict[seq[i]] = list(aa_onehot[i]).index(1)

    if ss[i] in ss_dict.keys():
        if ss_dict[ss[i]] != list(ss_onehot[i]).index(1):
            print 'Error!',ss[i],i,ss_dict[ss[i]],list(ss_onehot[i]).index(1) 
    else:
        ss_dict[ss[i]] = list(ss_onehot[i]).index(1)

    if sa[i] in sa_dict.keys():
        if sa_dict[sa[i]] != list(sa_onehot[i]).index(1):
            print 'Error!',sa[i],i,sa_dict[sa[i]],list(sa_onehot[i]).index(1)
    else:
        sa_dict[sa[i]] = list(sa_onehot[i]).index(1)

print 'AA Index:'
for ch in aa_dict.keys():
    print ch,aa_dict[ch]
print ''

print 'SS Index:'
for ch in ss_dict.keys():
    print ch,ss_dict[ch]
print ''

print 'SA Index:'
for ch in sa_dict.keys():
    print ch,sa_dict[ch]

