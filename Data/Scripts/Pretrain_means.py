import DataLoading
import numpy as np

inter_dic = DataLoading.Interval_dic('data_1/Interval_1.fa')
input_file = open('Pretrain/folds_coordinate','r')
output_file = open('Pretrain/folds_means','w')

lines = input_file.readlines()
num = 0
v_sum = np.zeros(20)

for i in lines:
    j = i.strip('\n').split(' ')
    fold = j[0]
    if fold in inter_dic.keys():
        weight = inter_dic[fold][0]
        num += weight
        v = []
        for m in j[1:]:
            if m != '' and m != ' ':
                v.append(float(m))
        v = np.array(v)
        v_sum += v*weight

means = v_sum/num

print num

for i in means:
    output_file.write(str(i) + '\t')

output_file.write('\n')

input_file.close()
output_file.close()
