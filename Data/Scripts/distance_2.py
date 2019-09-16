import numpy as np
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')

coordinate_file = open('folds_coordinate','r')
n_file = open('novel_coordinate','r')
matrix_file = open('folds_diatance_2','w')

lines = coordinate_file.readlines()
dic = {}

for i in lines:
    line = i.strip('\n').split(' ')
    fold = line[0]
    vec = []
    for j in line[1:]:
        if j != '':
            vec.append(j)
    vec = [float(m) for m in vec]
    dic[fold] = np.array(vec)

line_new = n_file.readlines()[0].strip('\n').split(' ')
f_new = []
for i in line_new:
    if i != '':
        f_new.append(float(i))

dic['new'] = np.array(f_new)

f_list = dic.keys()

matrix_file.write('\t')
for f in f_list:
    matrix_file.write('\t' + f)
matrix_file.write('\n')

for f_1 in f_list:
    matrix_file.write(f_1)
    for f_2 in f_list:
        d = np.linalg.norm(dic[f_1] - dic[f_2])
        matrix_file.write('\t' + str(d))
    matrix_file.write('\n')

coordinate_file.close()
n_file.close()
matrix_file.close()



