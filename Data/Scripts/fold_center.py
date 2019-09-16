import numpy as np
from numpy import linalg
from represent import delete_fold
from numpy import linalg

folds=[]
f_name=[]
with open("folds_coordinate", "r") as f:
	for lines in f:
		line = lines.strip('\n').split()
		folds.append(map(float,line[1:]))
		f_name.append(line[0])


c = np.mean(folds, axis=0)

print c
