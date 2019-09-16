import numpy as np
from numpy import linalg
from sklearn import decomposition
from sklearn import preprocessing

x1= np.random.normal(0,1, (10,2))


s1=x1.T[0].mean()
s2=x1.T[1].mean()
print s1, s2 
x=[]

for i in range(len(x1)):
	x.append([])
	for j in range(len(x1[0])):
		x[i].append(x1[i][j]-x1.T[j].mean())







P = decomposition.PCA(n_components=2)

P.fit(x)


print "pca sklearn:"
print P.explained_variance_*0.9
print P.components_



print np.array(x).T[0].mean(), np.array(x).T[1].mean()

K=np.dot(np.array(x), np.array(x).T)

#K=np.dot(np.array(x).T, np.array(x))/10.

w, v = linalg.eig(K)

v=np.transpose(v)

for i in range(len(w)):
	for j in range(i+1, len(w)):
		if(w[i]<w[j]):
			temp=w[i]
			w[i]=w[j]
			w[j]=temp
			v[[i,j]]=v[[j,i]]
print "pca ziji:"
print  w[0]/10
print v