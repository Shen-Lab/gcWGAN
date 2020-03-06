#################################################################################
# Plot the nonsense ratio for different feedback penalty weight.
# 11/21/2019
# Requirement: Have the nonsense ratios calculated on the input set.
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

hp_list = ['0.001','0.01','0.02','0.05','0.1','1.0','10.0']
show_list = [str('%.0e' %float(i)) for i in hp_list]
num = len(hp_list)

NR = []

for i in range(num):
    index = '0.0001_5_64_' + hp_list[i] + '_'

    nr = DL.file_list('gcWGAN_Validation_Results/NR_reca_' + index + set_kind + '.fa')
    nr = [float(j) for j in nr[0:100]]
    NR.append(nr)

x_1 = range(1,101)

plt.figure()
for i in range(num):
    plt.plot(x_1,NR[i], label = show_list[i])
plt.ylim(0,1.1)
plt.xlabel('Epoch')
plt.ylabel('Average Nonsense Sequence Ratio')
plt.legend()
plt.savefig('gcWGAN_Validation_Results/gcWGAN_Validation_Images/Reca_NR_' + set_kind + '.eps')

