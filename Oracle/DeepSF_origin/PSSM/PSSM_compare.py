import numpy as np
import sys 

file_1 = sys.argv[1]
file_2 = sys.argv[2]

def read_PSSM(input_file):
    with open(input_file,'r') as f:
        lines = f.readlines()
    return np.array([int(i.split(':')[1]) for i in lines[1].strip('\n').split('\t')[1].split(' ')])

array_1 = read_PSSM(file_1)
array_2 = read_PSSM(file_2)

len_1 = len(array_1)
len_2 = len(array_2)

if len_1 != len_2:
    print 'Error! The size of the PSSMs do not match!'
    quit()


same_num = list(array_1 == array_2).count(True)
same_ratio = float(same_num)/len_1
l2_norm = np.linalg.norm(array_1 - array_2)
div_dim_l2 = l2_norm/len_1
div_var_l2 = div_dim_l2/np.var(np.hstack((array_1,array_2)))
div_max_l2 = div_dim_l2/(max(np.hstack((array_1,array_2))) - min(np.hstack((array_1,array_2))))

print 'Number of same entries:',same_num
print 'Ratio pf same entries:',same_ratio
print 'L2 distance:',l2_norm
print 'L2 (divided by dimension):',div_dim_l2
print 'L2 (divided by dimension and variance):',div_var_l2
print 'L2 (divided by dimension and range):',div_max_l2
