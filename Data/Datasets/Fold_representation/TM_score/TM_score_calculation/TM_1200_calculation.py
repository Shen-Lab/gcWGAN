import TM_helper
import sys

pdb = sys.argv[1]

vector = TM_helper.TM_1200(pdb,'pdbs/')

print vector

'''
fil = open(pdb[:-4] + '_TM_Vector','w')
for t in vector:
    fil.write(str(t) + '\n')
fil.close()
'''
