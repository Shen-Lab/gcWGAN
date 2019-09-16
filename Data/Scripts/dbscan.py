import numpy as np
from sklearn.cluster import DBSCAN

fold={}

s=0
s1=0
for i in range(20):
	with open("Train_Identity_Matrix_"+str(i), "r") as f:
		while(1):
			try:
				line=f.readline()
				if(line==""):
					break
			except:
				break

			lines=line.strip('\n').split()
			
			if(len(lines)>0 and lines[0][1]=='.'):

				num=int(lines[1])

				fname=lines[0]


				ma=[]

				for i in range(num):
					line=f.readline()
					lines=line.strip('\n').split()
					new= map(float, lines)

					ma.append(new)


				ma1=1.-np.array(ma)
				clustering = DBSCAN(metric='precomputed', eps=0.7, min_samples=1).fit(ma1)

				re=clustering.labels_

				index=[]
				
				score=[]
				for j in range(len(ma1)):
					score.append(np.sum(ma1[j][np.where(re==re[j])[0]]))


				for k in range(np.max(re)+1):
					tt=np.where(re==k)[0][0]
					for j in np.where(re==k)[0]:
						if(score[j]<score[tt]):
							tt=j

					index.append(tt)



				fold[fname]=index

				s+=len(np.where(re==-1)[0]) + np.max(re)+1
				s1+=len(index)

				print re

print s, s1

with open("cluster_result_03", "w") as f:
	for i in fold:
		f.write("%s\n" %(i))
		for j in range(len(fold[i])):
			f.write("%d " %(fold[i][j]))

		f.write("\n\n")





