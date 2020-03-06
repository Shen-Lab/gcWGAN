import matplotlib.pyplot as plt
import DataLoading as DL
plt.switch_backend('agg')

import os
if not os.path.exists('cWGAN_Validation_Results'):
    os.system('mkdir cWGAN_Validation_Results')
if not os.path.exists('cWGAN_Validation_Results/cWGAN_Validation_Images'):
    os.system('mkdir cWGAN_Validation_Results/cWGAN_Validation_Images')

hp_list = ['64','128','256']
show_list = ['  64','128','256']
num = len(hp_list)

Cri = []
Gen = []
NR_train = []
NR_new = []
Nov = []
PR = []

for i in range(num):
    index = '0.0001_5_' + hp_list[i] 

    cr = DL.file_list('../cWGAN_Training_Samples/TrainingSamples_' + index +'/Critic_Cost')
    cr = [-float(j) for j in cr[0:100]]
    Cri.append(cr)
    ge = DL.file_list('../cWGAN_Training_Samples/TrainingSamples_' + index +'/Generator_Cost')
    ge = [float(j) for j in ge[0:100]]
    Gen.append(ge)

    nr_t = DL.file_list('cWGAN_Validation_Results/' + index + '_Nonsense_Ratio_train.fa')
    nr_t = [float(j) for j in nr_t[0:100]]
    NR_train.append(nr_t)
    nr_n = DL.file_list('cWGAN_Validation_Results/' + index + '_Nonsense_Ratio_new.fa')
    nr_n = [float(j) for j in nr_n[0:100]]
    NR_new.append(nr_t)

    no = DL.file_list('cWGAN_Validation_Results/Identity_' + index)
    no = [float(j) for j in no[0:51]]
    Nov.append(no)
    pr = DL.file_list('cWGAN_Validation_Results/Padding_Ratio_' + index)
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
plt.savefig('cWGAN_Validation_Results/cWGAN_Validation_Images/CriticLoss_NoiseLength.eps')

plt.figure(2)
for i in range(num):
    plt.plot(x_1,NR_train[i], label = show_list[i])
plt.ylim(0,1.1)
plt.xlabel('Epoch')
plt.ylabel('Nonsense Sequence Ratio')
plt.legend()
plt.savefig('cWGAN_Validation_Results/cWGAN_Validation_Images/NonsenseSequenceRatio_NoiseLength_Train.eps')

plt.figure(3)
for i in range(num):
    plt.plot(x_1,NR_new[i], label = show_list[i])
plt.ylim(0,1.1)
plt.xlabel('Epoch')
plt.ylabel('Nonsense Sequence Ratio')
plt.legend()
plt.savefig('cWGAN_Validation_Results/cWGAN_Validation_Images/NonsenseSequenceRatio_NoiseLength_New.eps')

plt.figure(4)
for i in range(num):
    plt.plot(x_50,Nov[i], label = show_list[i])
plt.xlim(xmin = 0)
plt.ylim(0,0.4)
plt.xlabel('Epoch')
plt.ylabel('Sequence Identity')
plt.legend()
plt.savefig('cWGAN_Validation_Results/cWGAN_Validation_Images/SequenceIdentity_NoiseLength.eps')

plt.figure(5)
for i in range(num):
    plt.plot(x_50,PR[i], label = show_list[i])
plt.xlim(xmin = 0)
plt.ylim(0,1.1)
plt.xlabel('Epoch')
plt.ylabel('Padding Ratio')
plt.legend()
plt.savefig('cWGAN_Validation_Results/cWGAN_Validation_Images/PaddingRatio_NoiseLength.eps')







