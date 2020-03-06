##########################################################################
# Download a pdb file with the chain and residue constrains.
##########################################################################

import numpy as np
import os
import sys

pdb = sys.argv[1].upper()
link = 'https://files.rcsb.org/download/%s.pdb'%pdb
save_path = sys.argv[2]

if len(sys.argv) >= 4:
    sele_flag = True
    chain = sys.argv[3].upper()
    sele = '(chain %s'%chain
    if len(sys.argv) >= 5:
        resi = sys.argv[4]
        sele += ' and resi %s'%resi
    sele += ')'
    print pdb    
    print sele
else:
    sele_flag = False

if sele_flag:
    index = 0
    while os.path.exists('temp_' + str(index) + '.pdb'):
        index += 1
    file_temp = 'temp_' + str(index) + '.pdb'

    os.system("curl -o %s %s"%(file_temp,link))
    pymol=[]
    pymol.append("load " + file_temp)
    pymol.append("select "+sele)
    pymol.append("save " + save_path + ", sele")
    np.savetxt("load_pdb.pml", pymol, fmt="%s")
    os.system("/opt/coe/pymol/pymol -cq load_pdb.pml")
    os.system("rm " + file_temp)
    os.system("rm load_pdb.pml")

else:
    os.system("curl -o %s %s"%(save_path,link))
