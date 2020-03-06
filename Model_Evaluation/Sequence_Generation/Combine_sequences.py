#################################################################################
# Combine several squence files into one with an upperbound of sequence numbers.
#################################################################################

import sys

upper_bound = int(sys.argv[1])

file_lists = sys.argv[2:-1]

file_new = sys.argv[-1]

file_w = open(file_new,'w')

seq_num = 0
for fi in file_lists:
    with open(fi,'r') as file_r:
        lines = file_r.readlines()
    l = len(lines)
    for i in range(l):
        if lines[i][0] == '>':
            seq_num += 1
            if seq_num <= upper_bound:
                 file_w.write('>' + str(seq_num) + '\n')
                 file_w.write(lines[i+1])
            else:
                 break

file_w.close()
