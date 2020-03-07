# Oracle

This part contains the materials of the oracles. ***DeepSF_modified*** is for our modified oracle, and ***DeepSF_origin*** is for the original DeepSF. ***DeepSF_origin*** contains the scripts we developed to get the the sequence features (PSSM, secondary structure and solvent assessbility) and predict the target fold os a sequence based on the original DeepSF. To run such scripts and the scripts in current directory which are for sequence generation, please download and compile the related packages:
* **[DeepSF](https://github.com/multicom-toolbox/DeepSF)** to ***DeepSF_origin***;
* **[blast-2.2.26](ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.2.26/)**, **[nr90](http://sysbio.rnet.missouri.edu/bdm_download/nr_database/nr90.tar.gz)** and **[nr70](http://sysbio.rnet.missouri.edu/bdm_download/nr_database/nr70.tar.gz)** to ***DeepSF_origin/PSSM***;
* **[SCRATCH-1D_1.2](http://scratch.proteomics.ics.uci.edu/explanation.html)** to ***DeepSF_origin/SS_SA***.

## Operations:
* **Load Environment:**
```
module load Anaconda/2-5.0.1
source activate DeepDesign_acc
module load cuDNN/5.1-CUDA-8.0.44
```
For ***DeepSF_origin/PSSM/PSSM_calculation.py***, also run:

```
module load BLAST+/2.7.1-intel-2017b-Python-2.7.14
```

* **Generate sequences for a known fold (a fold that belongs to those we applied for data representation):**
```
python SeqGenerator_Filter.py <check point path>  <fold name>  <kind>  <sequence number>  <file path>  <job index> (<minimum length>  <maximum length>  <*oracle kind>)
```
*kind* = 'All' or 'Success'; *file path* refers to the path of the file and some charecters to distinguish the generated files; *job index* is for differnt jobs to run in paralell ao that they will not contradict with each other; 'M_DeepSF' and 'O_DeepSF' can independently appear or not in *\*oracle kind*, while they respectively refer to the modified oracle and the original DeepSF.

* **Generate sequences for the novel fold:**
```
python Nov_SeqGenerator_Filter.py <check point path>  <kind>  <sequence number>  <file path>  <job index> (<minimum length>  <maximum length>  <*oracle kind>)
```
* **Generate sequences for the novel fold and take a certain fold to be the target:**
```
python Nov_SeqGenerator_Filter_target.py <check point path>  <kind>  <sequence number>  <target fold> <file path>  <job index> (<minimum length>  <maximum length>  <*oracle kind>)
```
