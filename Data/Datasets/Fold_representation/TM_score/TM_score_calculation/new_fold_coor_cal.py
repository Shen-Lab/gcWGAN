import numpy as np
from numpy import linalg
import sys
import TM_helper

class eigen:
	def __init__(self):
		self.x=0
	def center(self, X, tm):

		m,n = len(X), len(X[0])
		
		H1= np.ones((n,n))/float(n)
		H2= np.ones((m,n))/float(n)

		return X - np.matmul(X, H1) - np.matmul(H2,tm) + np.matmul(np.matmul(H2,tm), H1)

	def fit(self, X):
		w, v = linalg.eig(X)
		
		v=np.transpose(v)

		for i in range(len(w)):
			for j in range(i+1, len(w)):
				if(w[i]<w[j]):
					temp=w[i]
					w[i]=w[j]
					w[j]=temp
					v[[i,j]]=v[[j,i]]
		return w, v


#tm = np.loadtxt("/home/cyppsp/project_deepdesign/Data/TM_matrix.fa")
tm = np.loadtxt("../Data/TM_matrix.fa")
tm = tm + np.identity(len(tm))*5.6
tm_test = TM_helper.TM_1200(sys.argv[1], 'pdbs/')

#np.savetxt("1", tm_test)
#tm_test = np.loadtxt("1")

model=eigen()
tm_test_new = model.center(np.array([tm_test]), tm)
fold_basis=np.loadtxt("../Data/scripts/folds_basis")

new_coor=[]

for i in range(1):
	coor_temp=[]
	for j in range(20):
		coor_temp.append(np.dot(tm_test_new[i],fold_basis[j]))

	new_coor.append(coor_temp)

np.savetxt("coordinate", new_coor, fmt='%7.4f')















