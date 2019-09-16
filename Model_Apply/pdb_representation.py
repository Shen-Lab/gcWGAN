import numpy as np
import TM_helper
from numpy import linalg
import os
import sys

pdb_file = sys.argv[1]
coor_file = sys.argv[2]

fil_coor = open(coor_file,'w')    
cor = TM_helper.coordinate_calculation(pdb_file)
for c in cor[0]: 
   fil_coor.write(' ' + str(c))
fil_coor.write('\n')
fil_coor.close()
        
