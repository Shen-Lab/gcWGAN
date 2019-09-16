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
Dist = []

matrix_file.write('\t')
for f in f_list:
    matrix_file.write('\t' + f)
matrix_file.write('\n')

for f_1 in f_list:
    matrix_file.write(f_1)
    for f_2 in f_list:
        d = dic[f_1] - dic[f_2]
        d = math.sqrt(np.dot(d,d))
        matrix_file.write('\t' + str(d))
    matrix_file.write('\n')

num = 0
for i in range(l):
    for j in range(i+1,l):
        d = dic[f_list[i]] - dic[f_list[j]]
        d = math.sqrt(np.dot(d,d))
        num += 1
        Dist.append(d)

hist, bin_edges = np.histogram(Dist,bins = 1000)
SUM = sum(hist)
CDF = np.cumsum(hist) 
print SUM
print CDF[-1]
print len(CDF)
print len(bin_edges)

CDF = CDF/float(SUM)

for i in range(len(CDF)):
    if CDF[i] >= 0.073:
        print bin_edges[i+1]
        break

plt.figure(1)
plt.hist(Dist,bins = 100)
plt.savefig('../Images/Data/Distance_Distribution.png')
plt.figure(2)
plt.plot(bin_edges[1:],hist)
plt.savefig('../Images/Data/Distance_Hist.png')
plt.figure(3)
plt.plot(bin_edges[1:],CDF)
plt.savefig('../Images/Data/Distance_CDF.png')


coordinate_file.close()
matrix_file.close()

