import numpy as np
'''
folds= np.loadtxt("/home/cyppsp/project_deepdesign/Data/version03/subset1/fold_train_compromise", dtype='str')

fold={'a':[],
'b':[],
'c':[],
'd':[],
'e':[],
'f':[],
'g':[]
}

for i in range(len(folds)):
	if(folds[i][1:] not in fold[folds[i][0]]):
		fold[folds[i][0]] .append( folds[i][1:] )


n=0

validation=[]
fold_test=[]
for i in fold:
	arr = np.arange(len(fold[i]))
	np.random.shuffle(arr)

	for j in range(len(arr)):
		if(j*2<len(arr)):
			validation.append(i+fold[i][arr[j]])
		else:
			fold_test.append(i+fold[i][arr[j]])

np.savetxt("/home/cyppsp/project_deepdesign/Data/version03/subset1/fold_val", validation, fmt='%s')
np.savetxt("/home/cyppsp/project_deepdesign/Data/version03/subset1/fold_test", fold_test, fmt='%s')
'''

'''
#  validatae seqs:

deepsf_fold = np.loadtxt("/home/cyppsp/project_deepdesign/Data/version03/subset1/DeepSF_fold", dtype='str')
fold_train= np.loadtxt("/home/cyppsp/project_deepdesign/Data/version03/subset1/fold_train", dtype='str')
fold_com= np.loadtxt("/home/cyppsp/project_deepdesign/Data/version03/subset1/fold_train_compromise", dtype='str')

print deepsf_fold.T[0]


fold_untrain=[]
for i in range(len(fold_train)):
	if(fold_train[i] not in fold_untrain):
		fold_untrain.append(fold_train[i])


new_fold=[]
for i in range(len(fold_com)):
	if(fold_com[i] not in deepsf_fold.T[0]):
	#	print fold_com[i]
		new_fold.append(fold_com[i])

print '\n'
for i in range(len(fold_untrain)):
	if(fold_untrain[i] not in deepsf_fold.T[0]):
		#print fold_untrain[i]
		new_fold.append(fold_untrain[i])
'''



seq={}

fold={'a':[0,0,0],
'b':[0,0,0],
'c':[0,0,0],
'd':[0,0,0],
'e':[0,0,0],
'f':[0,0,0],
'g':[0,0,0]
}
fold_geshu={'a':[0,0,0],
'b':[0,0,0],
'c':[0,0,0],
'd':[0,0,0],
'e':[0,0,0],
'f':[0,0,0],
'g':[0,0,0]
}
fold_train= np.loadtxt("/home/cyppsp/project_deepdesign/Data/version03/subset1/fold_train", dtype='str')
with open("/home/cyppsp/project_deepdesign/Data/version03/subset1/Interval_1.fa", "r") as f:
	while(1):
		lines = f.readline()
		if(lines==''):
			break

		line1 = lines.strip('\n')

		if(line1[1]=='.'):
			line = line1.split()

		#	if(line[0] in new_fold):
			seq[line[0]]=[]


			for i in range(int(line[1])):

				lines=f.readline()
				lines=lines.strip('\n')

# zai xin de fold li:
			#	if(line[0] in new_fold):
				seq[line[0]].append(lines)


fold_train= np.loadtxt("/home/cyppsp/project_deepdesign/Data/version03/subset1/fold_train", dtype='str')
fold_val= np.loadtxt("/home/cyppsp/project_deepdesign/Data/version03/subset1/fold_val", dtype='str')
fold_test= np.loadtxt("/home/cyppsp/project_deepdesign/Data/version03/subset1/fold_test", dtype='str')

fold_geshu_t={'a':[0,0,0],
'b':[0,0,0],
'c':[0,0,0],
'd':[0,0,0],
'e':[0,0,0],
'f':[0,0,0],
'g':[0,0,0]
}
fold_geshu_v={'a':[0,0,0],
'b':[0,0,0],
'c':[0,0,0],
'd':[0,0,0],
'e':[0,0,0],
'f':[0,0,0],
'g':[0,0,0]
}
fold_geshu_te={'a':[0,0,0],
'b':[0,0,0],
'c':[0,0,0],
'd':[0,0,0],
'e':[0,0,0],
'f':[0,0,0],
'g':[0,0,0]
}
for i in seq:
	if(len(seq[i])<=5):
		if(i in fold_train):
			fold_geshu_t[i[0]][2]+=1
		elif(i in fold_val):
			fold_geshu_v[i[0]][2]+=1
		else:
			fold_geshu_te[i[0]][2]+=1

	elif(len(seq[i])<=50):
		if(i in fold_train):
			fold_geshu_t[i[0]][1]+=1
		elif(i in fold_val):
			fold_geshu_v[i[0]][1]+=1
		else:
			fold_geshu_te[i[0]][1]+=1

	else:
		if(i in fold_train):
			fold_geshu_t[i[0]][0]+=1
		elif(i in fold_val):
			fold_geshu_v[i[0]][0]+=1
		else:
			fold_geshu_te[i[0]][0]+=1

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
plt.figure(dpi=800)	
plt.subplots_adjust(wspace=0.2, hspace=0.3)

fig, ax = plt.subplots(nrows=7, ncols=3)

fig.subplots_adjust(wspace=0.2, hspace=0.3)

lis=['a','b','c','d','e','f','g']
c=0
for row in ax:
	t=0
	for col in row:
		sum = fold_geshu_t[lis[c]][t] + fold_geshu_v[lis[c]][t] + fold_geshu_te[lis[c]][t]
		if(sum==0):
			sum=1
		col.bar(1, fold_geshu_t[lis[c]][t]/float(sum))
		col.bar(2, fold_geshu_v[lis[c]][t]/float(sum))
		col.bar(3, fold_geshu_te[lis[c]][t]/float(sum))
		col.text(1, fold_geshu_t[lis[c]][t]/float(sum), "%.2f" % (fold_geshu_t[lis[c]][t]/float(sum)) )
		col.text(2, fold_geshu_v[lis[c]][t]/float(sum), "%.2f" % (fold_geshu_v[lis[c]][t]/float(sum)) )
		col.text(3, fold_geshu_te[lis[c]][t]/float(sum), "%.2f" % (fold_geshu_te[lis[c]][t]/float(sum)) )
		t+=1
	print np.sum(fold_geshu_t[lis[c]]), np.sum(fold_geshu_v[lis[c]]), np.sum(fold_geshu_te[lis[c]])

	plt.setp(row[0], ylabel=lis[c])
	c+=1

plt.setp(ax[0][0], xlabel='EZ')
plt.setp(ax[0][1], xlabel='Med')
plt.setp(ax[0][2], xlabel='Diff')
ax[0][0].xaxis.set_label_position('top')
ax[0][1].xaxis.set_label_position('top')
ax[0][2].xaxis.set_label_position('top')
plt.setp(ax, xticks=[], xticklabels=[], ylim=[0,1])
plt.setp(ax[6], xticks=[1,2,3], xticklabels=["train","val","test"])
plt.show()

ssum=0
for i in seq:
	print i, len(seq[i])
	ssum+=len(seq[i])
	if(len(seq[i])<=5):
		fold[i[0]][2]+= len(seq[i])
		fold_geshu[i[0]][2]+= 1
	elif (len(seq[i])<=50):
		fold[i[0]][1]+=len(seq[i])
		fold_geshu[i[0]][1]+= 1
	else:
		fold[i[0]][0]+=len(seq[i])
		fold_geshu[i[0]][0]+= 1
#	if(i in fold_train):
#		fold[i[0]][0]+=len(seq[i])
emd=[0.,0.,0.]
geshu=[0.,0.,0.]
fold_accu={}
for i in fold:
	for j in range(len(fold[i])):
		emd[j]+=fold[i][j]
		geshu[j]+=fold_geshu[i][j]

	print i, np.sum(fold[i])/float((ssum))/float(np.sum(fold_geshu[i]))


for i in [0,1,2]:
	print emd[i]/float(ssum)/geshu[i]
print ssum


exit(0)

def conver_seq(name, fold_index, seq):
	count=1
	letter_code=['a','c','d','e','f','g','h','i','k','l','m','n','p','q','r','s','t','v','w','y']
	with open("/home/cyppsp/project_deepdesign/DeepSF/"+name+".fea_aa_ss_sa", "w") as f:
		f.write(">%s\n" %(name))
		f.write("%d\t" %(fold_index))
		for i in range(len(seq)):
			for j in range(20):
				if(letter_code[j]==seq[i]):
					f.write("%d:1 " %(count))
				else:
					f.write("%d:0 " %(count))
				count+=1
			for j in range(5):
				f.write("%d:0 " %(count))
				count+=1

name=0

fold_index={
"d.395":  1196,
"g.100":   1197,
"g.98":    1198,
"g.95":  1199,
"a.297":   1200,
"a.301":   1201,
"b.181":   1202,
"b.182":   1203,
"d.391":   1204,
"d.386":   1205,
"d.388":   1206,
"g.97":    1207,
"f.62":    1208,
"b.180":   1209,
"d.392":  	1210,
"d.390":   1211,
"g.101":   1212,
"g.94":    1213,
"g.99":    1214}

name_seq={}
for i in seq:
	name_seq[i]=[]
	for j in seq[i]:
		conver_seq("newseq"+str(name), fold_index[i], j)
		name_seq[i].append("newseq"+str(name))
		name+=1

with open("/home/cyppsp/project_deepdesign/DeepSF/newtrain.list", "w") as ft:
	with open("/home/cyppsp/project_deepdesign/DeepSF/newval.list", "w") as fv:
		for i in seq:
			arr = np.arange(len(seq[i]))
			np.random.shuffle(arr)

			for j in range(len(arr)):
				if( j <= int((float(len(arr)))*0.8)):
					ft.write("%s %d\t%s.1.1\t%s\n" %(name_seq[i][arr[j]], len(seq[i][arr[j]]), i, i))
				else:
					fv.write("%s %d\t%s.1.1\t%s\n" %(name_seq[i][arr[j]], len(seq[i][arr[j]]), i, i))



