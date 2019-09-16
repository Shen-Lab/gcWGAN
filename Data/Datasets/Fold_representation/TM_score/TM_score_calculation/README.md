To calculate the TM scores between the input pdb file and 1232 known pdb files, you need to put the the input file in this directory and run the command "python TM_1200_calculation.py <file_name.pdb>", and it will generate a file of the TM scores vector. 
Or you can use the function TM_1200 in TM_helper.py to output a vector ranther than to generate a file.

1. To calculate the 20 coordinates for a new pdb, run 'python new_fold_coor_cal.py <pdb_path>'. There will be a new file 'coordinate' generated in the same directory which contains the 20 coordinates. 
Important: Please change the path in new_fold_coor_cal.py:

tm = np.loadtxt("/home/cyppsp/project_deepdesign/Data/TM_matrix.fa") 
fold_basis=np.loadtxt("/home/cyppsp/project_deepdesign/Data/scripts/folds_basis")
to your new path
