########################################################
# Plot the loss value based on different sets.
########################################################

import sys
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')

image_path = sys.argv[1]
set_kind = sys.argv[2]

if sys.argv[-1] == 'no_label':
    Weight_list = sys.argv[3:-1]
    whether_label = False
    label_notation = '_no_label'
else:
    Weight_list = sys.argv[3:]
    whether_label = True
    label_notation = ''

if not image_path.endswith('/'):
    image_path += '/'

Gen_loss_dic = {}
Cri_loss_dic = {}
Ove_loss_dic = {}

name = set_kind + '_' + '_'.join(Weight_list) + '_'

for weight in Weight_list:
    gen_loss = []
    cri_loss = []
    ove_loss = []

    loss_file = open('Converge_Check_Result/loss_0.0001_5_64_' + str(float(weight)) + '_semi_diff_' + set_kind,'r')
    lines = loss_file.readlines()[1:]
    loss_file.close()
    for line in lines:
        line = line.strip('\n').split('\t')
        gen_loss.append(float(line[0]))
        cri_loss.append(-float(line[1]))
        ove_loss.append(float(line[2]))
   
    Gen_loss_dic[weight] = gen_loss
    Cri_loss_dic[weight] = cri_loss
    Ove_loss_dic[weight] = ove_loss

max_len = max([len(w) for w in Weight_list])

plt.figure(1)
for w in Weight_list:
    Label = ' ' * (max_len - len(w)) + w
    plt.plot(Gen_loss_dic[w],label = Label)
plt.xlabel('Epoch')
plt.ylabel('Generator Loss')
if whether_label:
    plt.legend()
plt.savefig(image_path + name + 'generator_loss' + label_notation + '.png')

plt.figure(2)
for w in Weight_list:
    Label = ' ' * (max_len - len(w)) + w
    plt.plot(Cri_loss_dic[w],label = Label)
plt.xlabel('Epoch')
plt.ylabel('Critic Loss')
if whether_label:
    plt.legend()
plt.savefig(image_path + name + 'critic_loss' + label_notation + '.png')

plt.figure(3)
for w in Weight_list:
    Label = ' ' * (max_len - len(w)) + w
    plt.plot(Ove_loss_dic[w],label = Label)
plt.xlabel('Epoch')
plt.ylabel('Overall Loss')
if whether_label:
    plt.legend()
plt.savefig(image_path + name + 'overall_loss' + label_notation + '.png')
