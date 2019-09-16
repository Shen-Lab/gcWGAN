import numpy as np
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')

coordinate_file = open('folds_coordinate','r')
matrix_file = open('folds_diatance','w')

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
    
f_list = dic.keys()
l = len(f_list)
N_num = []

CRI = [0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39]
for cri in CRI:
    for i in range(l):
        num = 0
        for j in range(l):
            d = dic[f_list[i]] - dic[f_list[j]]
            d = math.sqrt(np.dot(d,d))
            if d <= cri:
                num += 1
        N_num.append(num)

    print cri, np.mean(N_num)
