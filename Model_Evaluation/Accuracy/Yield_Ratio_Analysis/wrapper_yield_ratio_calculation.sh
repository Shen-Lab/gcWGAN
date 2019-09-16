
hour=03
path=$1
epoch=$2
file=../../../Data/Datasets/Final_Data/unique_fold

mkdir Terra_jobs_Yield_Ratio

for fold in $(cat $file); 
do
   ./create_job_yield_ratio.sh ${hour} ${path} ${epoch} ${fold}
   sbatch Terra_jobs_Yield_Ratio/Terra_gcyr_${path%/*}_${epoch}_${fold}
done
