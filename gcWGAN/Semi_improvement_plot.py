import sys
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')

image_path = sys.argv[1]
weight = sys.argv[2]
if sys.argv[-1] == 'no_label':
    whether_label = False
    label_notation = '_no_label'
else:
    whether_label = True
    label_notation = ''

if not image_path.endswith('/'):
    image_path += '/'

Gen_loss_dic = {}
Cri_loss_dic = {}
Ove_loss_dic = {}

kind_list = ['_semi_diff','_without_semi']
Label_dic = {'_semi_diff':'gcWGAN','_without_semi':'gcWGAN w/out ss'}

for kind in kind_list:
    gen_loss = []
    cri_loss = []
    ove_loss = []

    loss_file = open('Converge_Check_Result/loss_0.0001_5_64_' + str(float(weight)) + kind,'r')
    lines = loss_file.readlines()[1:]
    loss_file.close()
    for line in lines:
        line = line.strip('\n').split('\t')
        gen_loss.append(float(line[0]))
        cri_loss.append(-float(line[1]))
        ove_loss.append(float(line[2]))
   
    Gen_loss_dic[kind] = gen_loss
    Cri_loss_dic[kind] = cri_loss
    Ove_loss_dic[kind] = ove_loss

name = 'Semi_improve_' + weight + '_'

plt.figure(1)
for k in kind_list:
    Label = Label_dic[k]
    plt.plot(Gen_loss_dic[k],label = Label)
plt.xlabel('Epochs',fontsize = 15, weight = 'bold')
plt.ylabel('Generator Loss',fontsize = 15, weight = 'bold')
if whether_label:
    plt.legend(fontsize = 15,prop = {'weight':'bold'})
plt.tight_layout()
plt.savefig(image_path + name + 'generator_loss' + label_notation + '.png')

plt.figure(2)
for k in kind_list:
    Label = Label_dic[k]
    plt.plot(Cri_loss_dic[k],label = Label)
plt.xlabel('Epochs',fontsize = 15, weight = 'bold')
plt.ylabel('Critic Loss',fontsize = 15, weight = 'bold')
if whether_label:
    plt.legend(fontsize = 15,prop = {'weight':'bold'})
plt.tight_layout()
plt.savefig(image_path + name + 'critic_loss' + label_notation + '.png')

plt.figure(3)
for k in kind_list:
    Label = Label_dic[k]
    plt.plot(Ove_loss_dic[k],label = Label)
plt.xlabel('Epochs',fontsize = 15, weight = 'bold')
plt.ylabel('Overall Loss',fontsize = 15, weight = 'bold')
if whether_label:
    plt.legend(fontsize = 15,prop = {'weight':'bold'})
plt.tight_layout()
plt.savefig(image_path + name + 'overall_loss' + label_notation + '.png')
