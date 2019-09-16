import DataLoading
import sys
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')
import os

X_RANGE = 100000

fold_list_1 = ['b.2','c.56','c.94']
fold_list_2 = ['a.35','d.107','g.44']

if not os.path.exists('../../Results/'):
    os.system('mkdir ../../Results/')
if not os.path.exists('../../Results/Generating_Rate/'):
    os.system('mkdir ../../Results/Generating_Rate/')

def AXIS(y_data,x_range):
    x_axis = range(x_range + 1)
    start = 0
    y_axis = []
    for i in x_axis:
        if i in y_data:
            start += 1
        y_axis.append(start)
    return x_axis,y_axis

for j in range(len(fold_list_1)):
    f = fold_list_1[j]
    path_0 = 'Pipeline_Sample/Generating_Ratio_Samples/cWGAN_Stat_gen_' + f + '_100000'
    path_2 = 'Pipeline_Sample/Generating_Ratio_Samples/gcWGAN_Stat_gen_' + f + '_100000'
    Data_0 = DataLoading.columns_to_lists(path_0)
    #suc_0 = [0] + [int(i) for i in Data_0[0]]
    all_0 = [int(i) for i in Data_0[1]]
    x_0, y_0 = AXIS(all_0,X_RANGE)
    Data_2 = DataLoading.columns_to_lists(path_2)
    #suc_2 = [0] + [int(i) for i in Data_2[0]]
    all_2 = [0] + [int(i) for i in Data_2[1]]
    x_2, y_2 = AXIS(all_2,X_RANGE)
    font = {'weight' : 'bold', 'size' : 20}

    reg_0 = LinearRegression().fit(np.array(x_0).reshape(-1,1), np.array(y_0).reshape(-1,1))
    reg_2 = LinearRegression().fit(np.array(x_2).reshape(-1,1), np.array(y_2).reshape(-1,1))
    slope_0 = reg_0.coef_[0]
    slope_2 = reg_2.coef_[0]
    flag = False
    if slope_0 != 0:
    	s_ratio = slope_2/slope_0
	flag = True

    y_max = max(max(y_0),max(y_2))

    plt.figure()
    plt.xlim(0,X_RANGE)
    plt.ylim(-1,y_max + 1)
    plt.plot(x_0,y_0,label = 'cwGAN') # (slope = %.4f)'%slope_0)
    plt.plot(x_2,y_2,label = 'guided cwGAN') # (slope = %.4f)'%slope_2)
    plt.xticks(fontsize = 15,fontweight = 'bold')
    plt.yticks(fontsize = 15,fontweight = 'bold')
    if flag:
        plt.text(2000,3.4/5.0*y_max,'slope ratio = %.4f'%s_ratio,size = 15,weight = 'bold')
        plt.draw()
    plt.xlabel('Generated Sequence Number', **font)
    plt.ylabel('Successful Sequence Number', **font)
    plt.legend(fontsize = 15, prop = {'weight':'bold'})
    plt.tight_layout()
    plt.savefig('../../Results/Generating_Rate/slope_' + f + '_success_all.eps')

    
for k in range(len(fold_list_2)):
    f = fold_list_2[k]
    path_0 = 'Pipeline_Sample/Generating_Ratio_Samples/cWGAN_Stat_time_' + f + '_200'
    path_2 = 'Pipeline_Sample/Generating_Ratio_Samples/gcWGAN_Stat_time_' + f + '_200'
    Data_0 = DataLoading.columns_to_lists(path_0)
    all_0 = []
    for i in range(len(Data_0[0])):
        all_0.append(int(Data_0[1][i]))
        if int(Data_0[1][i]) >= 100000:
            break
    x_0, y_0 = AXIS(all_0,X_RANGE)
    Data_2 = DataLoading.columns_to_lists(path_2)
    #suc_2 = [0]
    all_2 = []
    for i in range(len(Data_2[0])):
        #suc_2.append(int(Data_2[0][i]))
        all_2.append(int(Data_2[1][i]))
        if int(Data_2[1][i]) >= 100000:
            break
    x_2, y_2 = AXIS(all_2,X_RANGE)
    font = {'weight' : 'bold', 'size' : 20}
    
    reg_0 = LinearRegression().fit(np.array(x_0).reshape(-1,1), np.array(y_0).reshape(-1,1))
    reg_2 = LinearRegression().fit(np.array(x_2).reshape(-1,1), np.array(y_2).reshape(-1,1))
    slope_0 = reg_0.coef_[0]
    slope_2 = reg_2.coef_[0]
    flag = False
    if slope_0 != 0:
        s_ratio = slope_2/slope_0
        flag = True

    y_max = max(max(y_0),max(y_2))   
 
    plt.figure()
    plt.xlim(0,X_RANGE)
    plt.ylim(-1,y_max + 1)
    plt.plot(x_0,y_0,label = 'cwGAN') # (slope = %.4f)'%slope_0)
    plt.plot(x_2,y_2,label = 'guided cwGAN') # (slope = %.4f)'%slope_2)
    plt.xticks(fontsize = 15,fontweight = 'bold')
    plt.yticks(fontsize = 15,fontweight = 'bold')
    if flag:
        plt.text(2000,3.4/5.0*y_max,'slope ratio = %.4f'%s_ratio,size = 15,weight = 'bold')
    	plt.draw()
    plt.xlabel('Generated Sequence Number', **font)
    plt.ylabel('Successful Sequence Number', **font)
    plt.legend(fontsize = 15, prop = {'weight':'bold'})
    plt.tight_layout()
    plt.savefig('../../Results/Generating_Rate/slope_' + f + '_success_all.eps')


