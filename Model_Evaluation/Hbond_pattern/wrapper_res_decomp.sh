
hour=1
IFS=$'\n'
for seq in $(cat list.txt);do
   ./create_job_res_decomp.sh ${seq} ${seq:0:${#seq}-4}_res_decomp.out ${hour}
   bsub < ada_res_decomp.job
done
