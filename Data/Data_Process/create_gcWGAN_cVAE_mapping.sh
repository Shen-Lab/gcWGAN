
kind=$1
method=$2
path=$3

echo "#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=mapping_${kind}_${method}
#SBATCH --time=5:00:00              
#SBATCH --ntasks=28      
#SBATCH --mem=40G                  
#SBATCH --output=Output/output_mapping_${kind}_${method}
#SBATCH --gres=gpu:1                #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=shaowen1994@tamu.edu    #Send all emails to email_address 
#SBATCH --account=122821643660

#First Executable Line
module load Anaconda/2-5.0.1
python gcWGAN_cVAE_mapping.py ${kind} ${method} ${path}gcWGAN_cVAE_mapping_${kind}_${method}" > Terra_files/Terra_mapping_${kind}_${method}
