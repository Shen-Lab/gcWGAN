# t-SNE
import numpy as np
from numpy import linalg
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import axes3d

# input folds_coordinate

x1 = np.loadtxt("folds_coordinate", dtype='str')
xt = np.loadtxt("../../Model_0/YieldRatio/0.0001_20_128_fold_accuracy_train.fa", dtype='str')
xv = np.loadtxt("../../Model_0/YieldRatio/0.0001_20_128_fold_accuracy_vali.fa", dtype='str')
xte = np.loadtxt("../../Model_0/YieldRatio/0.0001_20_128_fold_accuracy_test.fa", dtype='str')



x=[]
classs=[]
for i in range(len(x1)):
	x.append( map(float, x1[i][1:]) )
	classs.append(x1[i][0])




list1={
	'train' : xt,
	'validation': xv,
	'test' : xte
}

t_embed=TSNE(n_components=3).fit_transform(x)
for name in list1:


	fig=plt.figure()
	ax = fig.gca(projection='3d')



	#	print t_embed.shape

	for i in range(len(list1[name])):

		fold_name=list1[name][i][0]
		t = classs.index(fold_name)
		yield_ratio = np.log(float(list1[name][i][1])+0.1)

		p=ax.scatter(t_embed[t][0],t_embed[t][1],t_embed[t][2], alpha=0.8, vmin=-2.2, vmax=0, cmap='rainbow', c=yield_ratio)


	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	ax.set_zlabel("PC3")
	ax.set_xlim([-30,30])
	ax.set_ylim([-40,40])
	ax.set_zlim([-30,30])
	plt.legend(loc="upper left")
	plt.colorbar(p)
	plt.savefig("yield_ratio1_"+name+".png", dpi=800)


plt.show()
