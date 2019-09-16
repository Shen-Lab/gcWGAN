import numpy as np
from numpy import linalg
'''
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


tm = np.loadtxt("../TM_matrix.fa")
tm = tm + np.identity(len(tm))*5.6
tm_test_wo = np.loadtxt("/home/cyppsp/project_deepdesign/Data_Process/TM_scores_and_GramMatrix/TM_scores/TM_scores_novel.fa", dtype='str')
tm_test=[]

fold_basis=np.loadtxt("folds_basis")

for i in range(4):
	tm_test.append(map(float, tm_test_wo.T[i+1][1:]))
	print len(tm_test[i])

model=eigen()
tm_test_new = model.center(tm_test, tm)

new_coor=[]

for i in range(4):
	coor_temp=[]
	for j in range(20):
		coor_temp.append(np.dot(tm_test_new[i],fold_basis[j]))

	new_coor.append(coor_temp)

np.savetxt("novel_coordinate", new_coor, fmt='%7.4f')

'''

folds_novel=np.loadtxt("novel_coordinate")
folds_coor=np.loadtxt("folds_coordinate", dtype='str')

z=0

b=0
for i in range(len(folds_coor)):
	temp1 = np.linalg.norm(np.subtract(map(float, folds_coor[i][1:]), folds_novel[0]))
	temp2 = np.linalg.norm(np.subtract(map(float, folds_coor[z][1:]), folds_novel[0]))
	if(temp1<0.3):
		print folds_coor[i][0], temp1
		b+=1

print b
exit(0)


print folds_coor[z][0], np.linalg.norm(np.subtract(map(float, folds_coor[z][1:]), new_coor[0]))












