#######################################################################################
# Print out the files that are not empty in the given folder with the given charactors.
# 11/02/2019
# Input: path of the given folder
#        the charactors that should be in the file name
#######################################################################################

import os
import sys 

PATH = sys.argv[1]
CHARA = sys.argv[2]

if not PATH.endswith('/'):
    PATH += '/'

file_list = [f for f in os.listdir(PATH) if CHARA in f]

for f in file_list:
    fil = open(PATH + f,'r')
    lines = fil.readlines()
    fil.close()
    for line in lines:
        if line != '' and line != '\n':
            print f
            break
 
