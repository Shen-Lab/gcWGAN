import matplotlib.pyplot as plt
plt.switch_backend('agg')
import DataLoading
import sys
import os

FOLD = sys.argv[2]
KIND = sys.argv[1]

if len(sys.argv) >= 4:
    target_flag = True
    target = '_' + sys.argv[3]
else:
    target_flag = False
    target = ''

if 'nov' in FOLD:
    fold_name = 'nov'
    i_xy_range = [0,1,0,25]
    n_xy_range = [0,0.3,0,30]
    i_bins = [0.01*i for i in range(0,101)]
    n_bins = [0.01*i for i in range(0,31)]
elif '.' in FOLD:
    fold_name = FOLD.split('.')[0] + '_' + FOLD.split('.')[1]
    i_xy_range = [0,1,0,25]
    n_xy_range = [0,0.7,0,50]
    i_bins = [0.01*i for i in range(0,101)]
    n_bins = [0.01*i for i in range(0,71)]
else:
    print 'Error! Wrong fold name!'

if not os.path.exists('../../Results'):
    os.system('mkdir ../../Results')

path = '../../Results/Diversity_and_Novelty/'

if not os.path.exists(path):
    os.system('mkdir ' + path)
if not os.path.exists(path + 'Div_Nov_Image/' ):
    os.system('mkdir ' + path + 'Div_Nov_Image/' )

def hist_contour(x,bins):
    l = len(bins) - 1
    x_axis = []
    y_axis = []
    for i in range(l):
        x_axis.append((bins[i+1] + bins[i])/2.0)
        y_axis.append(0)
    for j in x:
        for k in range(l):
            if j >= bins[k] and j < bins[k+1]:
                y_axis[k] += 1
                break
            elif j == bins[k+1] and k == l-1:
                y_axis[k] += 1
                break
    return x_axis,y_axis

def case_columns_to_lists(file_name):
    fil = open(file_name,'r')
    lines = fil.readlines()
    fil.close()
    n = len(lines[0].strip('\n').split('\t'))
    result = []
    for i in range(n):
        result.append([])
    for line in lines:
        line = line.strip('\n').split('\t')
        if len(line) == n:
            for i in range(n):
                result[i].append(line[i])
    return result

def merge_hist_plot(Data_1,Data_2,labels_1,labels_2,colors_1,colors_2,lines,xlabel,ylabel,path,xy_range,figsize = None,bins = None):
    '''
    Data_1: list of the data lists for hist plot
    Data_2: list of the data lists for hist contour plot
    labels_1: legends of the data in Data_1
    labels_2: legends of the data in Data_2
    colors_1: colors of the data in Data_1
    colors_2: colors of the data in Data_2
    lines: types of the lines used for Data_2
    '''   
    l_1 = len(Data_1)
    l_2 = len(Data_2)
 
    if l_1 > len(labels_1) or l_2 > len(labels_2) :
        print 'Error! Label not defined for some data'
        return None
    if l_1 > len(colors_1) or l_2 > len(colors_2) :
        print 'Error! Color not defined for some data'
        return None
    if l_2 > len(lines) :
        print 'Error! Line type not defined for some data'
        return None

    if figsize != None:
        plt.figure(figsize = figsize)
    else:
        plt.figure()
    plt.axis(xy_range)
    step = int((xy_range[3] - xy_range[2])/5)
    y_ticks = range(xy_range[2],xy_range[3] + 1,step)

    for i in range(l_1):
        plt.hist(Data_1[i],bins = bins,color = colors_1[i],density = True,alpha = 0.5,label = labels_1[i])

    for j in range(l_2):
        x,y = hist_contour(Data_2[j],bins)
        y_sum = float(sum(y))
        y = [100.0*(y_i/y_sum) for y_i in y]
        plt.plot(x,y,color = colors_2[j],linestyle=lines[j],label = labels_2[j])

    plt.legend(fontsize = 15,prop = {'weight':'bold'})
    plt.xticks(fontsize = 15,fontweight = 'bold')
    plt.yticks(y_ticks,fontsize = 15,fontweight = 'bold')
    plt.xlabel(xlabel,fontsize = 20, weight = 'bold')
    plt.ylabel(ylabel,fontsize = 20, weight = 'bold')
    plt.tight_layout()
    plt.savefig(path)
    plt.savefig(path + '.eps')

Iden_s = [float(i) for i in DataLoading.columns_to_lists(path + 'Identity_' + KIND + '_Fasta_100_success_' + fold_name + target)[0][0:-2]]
Iden_r = [float(i) for i in DataLoading.columns_to_lists(path + 'Identity_' + KIND + '_Fasta_100_random_' + fold_name)[0][0:-2]]
Iden_cVAE_r = [float(i) for i in DataLoading.columns_to_lists(path + 'Identity_cVAE_100_' + fold_name)[0][0:-2]]
Iden_cVAE_r_noX = [float(i) for i in DataLoading.columns_to_lists(path + 'Identity_cVAE_100_noX_' + fold_name)[0][0:-2]]

Diver_Data_2 = [Iden_r,Iden_cVAE_r]
Diver_Data_2_noX = [Iden_r,Iden_cVAE_r_noX]
Diver_Data_1 = [Iden_s]

if os.path.exists(path + 'Identity_cVAE_Fasta_100_success_' + fold_name + target):
    Iden_cVAE_s = [float(i) for i in DataLoading.columns_to_lists(path + 'Identity_cVAE_Fasta_100_success_' + fold_name + target)[0][0:-2]]
    Diver_Data_1.append(Iden_cVAE_s)

if fold_name == 'nov':
    nov_s = [float(i) for i in DataLoading.columns_to_lists(path + 'Novelty_' + KIND + '_Fasta_100_success_' + fold_name + target)[0][0:-2]]
    nov_r = [float(i) for i in DataLoading.columns_to_lists(path + 'Novelty_' + KIND + '_Fasta_100_random_' + fold_name)[0][0:-2]]
    nov_cVAE_r = [float(i) for i in DataLoading.columns_to_lists(path + 'Novelty_cVAE_100_' + fold_name)[0][0:-2]]
    nov_cVAE_r_noX = [float(i) for i in DataLoading.columns_to_lists(path + 'Novelty_cVAE_100_noX_' + fold_name)[0][0:-2]]
    if os.path.exists(path + 'Novelty_cVAE_Fasta_100_success_' + fold_name + target):
        nov_cVAE_s = [float(i) for i in DataLoading.columns_to_lists(path + 'Novelty_cVAE_Fasta_100_success_' + fold_name + target)[0][0:-2]]

else:
    nov_s = [float(i) for i in case_columns_to_lists(path + 'Novelty_' + KIND + '_Fasta_100_success_' + fold_name + target)[1]]
    nov_r = [float(i) for i in case_columns_to_lists(path + 'Novelty_' + KIND + '_Fasta_100_random_' + fold_name)[1]]
    nov_cVAE_r = [float(i) for i in case_columns_to_lists(path + 'Novelty_cVAE_100_' + fold_name)[1]]
    nov_cVAE_r_noX = [float(i) for i in case_columns_to_lists(path + 'Novelty_cVAE_100_noX_' + fold_name)[1]]
    #print len(nov_s)
    if os.path.exists(path + 'Novelty_cVAE_Fasta_100_success_' + fold_name + target):
        nov_cVAE_s = [float(i) for i in case_columns_to_lists(path + 'Novelty_cVAE_Fasta_100_success_' + fold_name + target)[1]]

Nov_Data_2 = [nov_r,nov_cVAE_r]
Nov_Data_2_noX = [nov_r,nov_cVAE_r_noX]
Nov_Data_1 = [nov_s]

if os.path.exists(path + 'Novelty_cVAE_Fasta_100_success_' + fold_name + target):
    Nov_Data_1.append(nov_cVAE_s)

Label_1 = [KIND + ' (oracle-filtered)','cVAE (oracle-filtered)'] 
Label_2 = [KIND + ' (not filtered)','cVAE (not filtered)']
Color_1 = ['blue','darkorange']
Color_2 = ['black','darkgrey']
Line = ['--',':']

merge_hist_plot(Diver_Data_1,Diver_Data_2,Label_1,Label_2,Color_1,Color_2,Line,'Sequence Identity','Distribution Density',path + 'Div_Nov_Image/Identity_' + KIND + '_Distribution_' + fold_name + target,xy_range = i_xy_range,bins = i_bins)    # plot the diversity distribution

merge_hist_plot(Diver_Data_1,Diver_Data_2_noX,Label_1,Label_2,Color_1,Color_2,Line,'Sequence Identity','Distribution Density',path + 'Div_Nov_Image/Identity_' + KIND + '_Distribution_noX_' + fold_name + target,xy_range = i_xy_range,bins = i_bins)    # plot the diversity distribution of sequences without X

merge_hist_plot(Nov_Data_1,Nov_Data_2,Label_1,Label_2,Color_1,Color_2,Line,'Maximum Sequence Identity','Distribution Density',path + 'Div_Nov_Image/Novelty_' + KIND + '_Distribution_' + fold_name + target,xy_range = n_xy_range,bins = n_bins)    # plot the novelty distribution

merge_hist_plot(Nov_Data_1,Nov_Data_2_noX,Label_1,Label_2,Color_1,Color_2,Line,'Maximum Sequence Identity','Distribution Density',path + 'Div_Nov_Image/Novelty_' + KIND + '_Distribution_noX_' + fold_name + target,xy_range = n_xy_range,bins = n_bins)    # plot the novelty distribution of sequences without X
