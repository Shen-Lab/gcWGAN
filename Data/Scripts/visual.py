import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d



fold_coor=np.loadtxt("folds_coordinate", dtype='str')

classs=[]
x1=[]
x2=[]
x3=[]
for i in range(len(fold_coor)):
	classs.append(fold_coor[i][0][0])
	x1.append(float(fold_coor[i][1]))
	x2.append(float(fold_coor[i][2]))
	x3.append(float(fold_coor[i][3]))



plt.figure()

c=[]

colormap='rainbow'

class1={
	'a':'red',
	'b':'pink',
	'c':'orange',
	'd':'yellow',
	'e':'green',
	'f':'blue',
	'g':'purple'
}



fig=plt.figure()
ax = fig.gca(projection='3d')

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
for i in range(len(x1)):

	if classs[i] in judge:
		ax.scatter(x1[i],x2[i],x3[i], alpha=0.8, c=class1[classs[i]])
	else:
		ax.scatter(x1[i],x2[i],x3[i], alpha=0.8, c=class1[classs[i]], label=label[classs[i]])
		judge[classs[i]]=1

ax.set_xlabel("PC1", fontsize=12)
ax.set_ylabel("PC2", fontsize=12)
ax.set_zlabel("PC3", fontsize=12)
plt.legend(loc="upper left", prop={'size': 20})


plt.show()

plt.close()
