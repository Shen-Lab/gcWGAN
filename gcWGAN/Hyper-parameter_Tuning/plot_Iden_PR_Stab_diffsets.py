#################################################################################
# Plot the sequence identity, padding ratio and sequence stability for different 
# feedback penalty weight.
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

if not os.path.exists('gcWGAN_Validation_Results/gcWGAN_Validation_Images'):
    os.system('mkdir gcWGAN_Validation_Results/gcWGAN_Validation_Images')

hp_list = ['0.001','0.01','0.02','0.05']
show_list = [str('%.0e' %float(i)) for i in hp_list]
num = len(hp_list)

Nov = []
PR = []
Stab = []

for i in range(num):
    index = '0.0001_5_64_' + hp_list[i] + '_'

    no = DL.file_list('gcWGAN_Validation_Results/Identity_' + index + set_kind)
    no = [float(j) for j in no[0:51]]
    Nov.append(no)
    pr = DL.file_list('gcWGAN_Validation_Results/Padding_Ratio_' + index + set_kind)
    pr = [float(j) for j in pr[0:51]]
    PR.append(pr)
    sta = DL.file_list('gcWGAN_Validation_Results/Stability_' + index + set_kind)
    sta = [float(j) for j in sta[0:51]]
    Stab.append(sta)

x_50 = range(50,101)

plt.figure(1)
for i in range(num):
    plt.plot(x_50,Nov[i], label = show_list[i])
plt.xlim(xmin = 0)
plt.ylim(0,0.4)
plt.xlabel('Epoch')
plt.ylabel('Average Sequence Identity')
plt.legend()
plt.savefig('gcWGAN_Validation_Results/gcWGAN_Validation_Images/SequenceNovelty_' + set_kind + '.eps')

plt.figure(2)
for i in range(num):
    plt.plot(x_50,PR[i], label = show_list[i])
plt.xlim(xmin = 0)
plt.ylim(0,1.1)
plt.xlabel('Epoch')
plt.ylabel('Padding Ratio')
plt.legend()
plt.savefig('gcWGAN_Validation_Results/gcWGAN_Validation_Images/PaddingRatio_' + set_kind + '.eps')

plt.figure(3)
for i in range(num):
    plt.plot(x_50,Stab[i], label = show_list[i])
plt.xlim(xmin = 0)
plt.ylim(0,60)
plt.xlabel('Epoch')
plt.ylabel('Average Sequence Stability')
plt.legend()
plt.savefig('gcWGAN_Validation_Results/gcWGAN_Validation_Images/SequenceStability_' + set_kind + '.eps')





