# t-SNE
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
coor = np.loadtxt("../Results/folds_coordinate", dtype='str')


x1= []
for i in range(len(coor)):
   line = list(map(float, coor[i][1:]))
   x1.append(np.array(line))






#	print t_embed.shape

classs=[]
with open("represent_file", "r") as f:
	for line in f:
		classs.append(line[5])






class1={
	'a':'red',
	'b':'yellow',
	'c':'orange',
	'd':'pink',
	'e':'green',
	'f':'blue',
	'g':'purple'
}


t_embed=TSNE(n_components=2).fit_transform(x1)
#t_embed = x1


fig=plt.figure()
ax = fig

label={
	'a':r'$\alpha$',
	'b':r'$\beta$',
	'c':r'$\alpha/\beta$',
	'd':r'$\alpha+\beta$',
	'e':r'$\alpha and \beta$',
	'f':'Membrane and cell surface',
	'g':'Small proteins'
}

judge={}

print (t_embed[1][0])

for i in range(len(x1)):

	if classs[i] in judge:
		plt.scatter(t_embed[i][0],t_embed[i][1], alpha=0.8, c=class1[classs[i]], s=8)
	else:
		plt.scatter(t_embed[i][0],t_embed[i][1], alpha=0.8, c=class1[classs[i]], label=label[classs[i]], s=8)
		judge[classs[i]]=1

#ax.set_xlabel("PC1")
#ax.set_ylabel("PC2")
#ax.set_zlabel("PC3")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc="upper right")
plt.savefig("tSNE_2d.png", format='png')
plt.savefig("tSNE_2d.eps", format='eps')
plt.show()
