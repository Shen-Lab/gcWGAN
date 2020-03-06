
pdb_name=$1
output_name=$2
hour=$3

echo "#BSUB -J design_job
#BSUB -L /bin/bash
#BSUB -W ${hour}:00
#BSUB -n 1
#BSUB -M 40000
#BSUB -R span[ptile=1]
#BSUB -R rusage[mem=40000]
#BSUB -o log_${pdb_name:0:${#pdb_name}-4}
#BSUB -P 082788946244

module load Anaconda/2-5.0.1
source activate myRosetta
module load intel/2017A
./run_res_decomp.sh ${pdb_name} ${output_name}
source deactivate" > ada_res_decomp.job
