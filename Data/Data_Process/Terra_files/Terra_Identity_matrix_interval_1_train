#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Iden_matrix_interval_1_train
#SBATCH --time=20:00:00              
#SBATCH --ntasks=28      
#SBATCH --mem=40G                  
#SBATCH --output=Output/output_iden_matrix_interval_1_train
#SBATCH --gres=gpu:1                #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=shaowen1994@tamu.edu    #Send all emails to email_address 
#SBATCH --account=122821643660

#First Executable Line
module load Anaconda/2-5.0.1
source activate my_tensorflow-gpu-1.4.1
python Identity_Matrix_interval_1.py ../Datasets/Final_Data/unique_fold_train ../Datasets/Intermediate_Data/Identity_matrix_Interval_1/
source deactivate
