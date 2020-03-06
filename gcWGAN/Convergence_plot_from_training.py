import sys
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')

image_path = sys.argv[1]
if not image_path.endswith('/'):
    image_path += '/'

Gen_loss_dic = {}
Cri_loss_dic = {}
Weight_list = []
name = ''

for folder in sys.argv[2:]:
    if not folder.endswith('/'):
        folder += '/'
    gen_loss = []
    cri_loss = []
    model = folder.strip('/').split('/')[-1]
    weight = model.strip('_semi_diff').split('_')[-1]
    name += weight + '_'
    Weight_list.append(weight)
     
    Sample_files = [i for i in os.listdir(folder) if i.startswith('train')]
    loss_dic = {}
    for fil in Sample_files:
        index = int(fil.split('_')[-2])
        if index <= 100:
            s_file = open(folder + fil,'r')
            lines = s_file.readlines()
            s_file.close()   
            loss = [float(j) for j in lines[0].strip('\n').split('\t')]
            loss_dic[index] = loss
    for epo in range(1,101):
        gen_loss.append(loss_dic[epo][0])
        cri_loss.append(-loss_dic[epo][1])
    Gen_loss_dic[weight] = gen_loss
    Cri_loss_dic[weight] = cri_loss

max_len = max([len(w) for w in Weight_list])

plt.figure(1)
for w in Weight_list:
    Label = ' ' * (max_len - len(w)) + w
    plt.plot(Gen_loss_dic[w],label = Label)
plt.xlabel('Epoch')
plt.ylabel('Genertor Loss')
plt.legend()
plt.savefig(image_path + name + 'generator_loss.png')

plt.figure(2)
for w in Weight_list:
    Label = ' ' * (max_len - len(w)) + w
    plt.plot(Cri_loss_dic[w],label = Label)
plt.xlabel('Epoch')
plt.ylabel('Critic Loss')
plt.legend()
plt.savefig(image_path + name + 'critic_loss.png')

