
hour=$1
path=$2
epoch=$3
fold=$4

echo "#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=gcyr_${path%/*}_${epoch}_${fold}
#SBATCH --time=${hour}:00:00              
#SBATCH --ntasks=28      
#SBATCH --mem=40G                  
#SBATCH --output=output_gcyr_${path%/*}_${epoch}_${fold}
#SBATCH --gres=gpu:1                #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=shaowen1994@tamu.edu    #Send all emails to email_address 
#SBATCH --account=122821648698

#First Executable Line
module load Anaconda/2-5.0.1
source activate DeepDesign_acc
module load cuDNN/5.1-CUDA-8.0.44
python Yield_Ratio_calculation.py ${path} ${epoch} ${fold}
source deactivate" > Terra_jobs_Yield_Ratio/Terra_gcyr_${path%/*}_${epoch}_${fold}
