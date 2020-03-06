#################################################################################
# Plot the loss value, nonsense ratio, padding radio and sequence novelty to 
# tune the initial learning rate.
# 11/21/2019
# Requirement: Have the related values calculated on the input set.
#################################################################################

import matplotlib.pyplot as plt
import DataLoading as DL
plt.switch_backend('agg')

import os
import sys

set_kind = sys.argv[1]

if not set_kind in ['train','test','vali']:
    print 'Error! No set named %s'%set_kind
    quit()

if not os.path.exists('cWGAN_Validation_Results/cWGAN_Validation_Images'):
    os.system('mkdir cWGAN_Validation_Results/cWGAN_Validation_Images')

hp_list = ['0.00001','0.00005','0.0001','0.0002','0.0005']
show_list = [str('%.0e' %float(i)) for i in hp_list]
num = len(hp_list)

Cri = []
Gen = []
NR = []
Nov = []
PR = []

for i in range(num):
    index = hp_list[i] + '_10_128_'

    Loss_data = DL.columns_to_lists('cWGAN_Validation_Results/loss_' + index + set_kind)
    cr = [-float(j) for j in Loss_data[1][1:101]]
    Cri.append(cr)
    ge = [float(j) for j in Loss_data[0][1:101]]
    Gen.append(ge)

    nr = DL.file_list('cWGAN_Validation_Results/NR_reca_' + index + set_kind + '.fa')
    nr = [float(j) for j in nr[0:100]]
    NR.append(nr)

for i in range(num - 1):
    index = hp_list[i] + '_10_128_'

    no = DL.file_list('cWGAN_Validation_Results/Identity_' + index + set_kind)
    no = [float(j) for j in no[0:51]]
    Nov.append(no)
    pr = DL.file_list('cWGAN_Validation_Results/Padding_Ratio_' + index + set_kind)
    pr = [float(j) for j in pr[0:51]]
    PR.append(pr)

x_1 = range(1,101)
x_50 = range(50,101)

plt.figure(1)
for i in range(num):
    plt.plot(x_1,Cri[i], label = show_list[i])
plt.xlabel('Epoch')
plt.ylabel('Critic Loss')
plt.legend()
plt.savefig('cWGAN_Validation_Results/cWGAN_Validation_Images/CriticLoss_LearningRate_' + set_kind + '.eps')

plt.figure(2)
for i in range(num):
    plt.plot(x_1,NR[i], label = show_list[i])
plt.ylim(0,1.1)
plt.xlabel('Epoch')
plt.ylabel('Average Nonsense Sequence Ratio')
plt.legend()
plt.savefig('cWGAN_Validation_Results/cWGAN_Validation_Images/NonsenseSequenceRatio_LearningRate_' + set_kind + '.eps')

plt.figure(3)
for i in range(num-1):
    plt.plot(x_50,Nov[i], label = show_list[i])
plt.xlim(xmin = 0)
plt.ylim(0,0.4)
plt.xlabel('Epoch')
plt.ylabel('Average Sequence Identity')
plt.legend()
plt.savefig('cWGAN_Validation_Results/cWGAN_Validation_Images/SequenceNovelty_LearningRate_' + set_kind + '.eps')

plt.figure(4)
for i in range(num-1):
    plt.plot(x_50,PR[i], label = show_list[i])
plt.xlim(xmin = 0)
plt.ylim(0,1.1)
plt.xlabel('Epoch')
plt.ylabel('Padding Ratio')
plt.legend()
plt.savefig('cWGAN_Validation_Results/cWGAN_Validation_Images/PaddingRatio_LearningRate_' + set_kind + '.eps')







