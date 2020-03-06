input_pdb=$1
output_file=$2
../../rosetta_bin_linux_2018.33.60351_bundle/main/source/bin/residue_energy_breakdown.linuxgccrelease -database ../../rosetta_bin_linux_2018.33.60351_bundle/main/database -in:file:s ${input_pdb} -out:file:silent ${output_file}
python get_Hbond.py ${output_file}
