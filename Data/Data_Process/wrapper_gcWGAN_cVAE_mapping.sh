
kind=$1
method=$2
path=$3

mkdir Terra_files
mkdir Output

./create_gcWGAN_cVAE_mapping.sh ${kind} ${method} ${path}
sbatch Terra_files/Terra_mapping_${kind}_${method}
