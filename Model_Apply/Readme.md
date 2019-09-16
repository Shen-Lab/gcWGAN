# Model Application
Given a certain protein structure, you can apply our model to generate related sequences that both pass or not pass the oracle.

## Pre-requisite
* Build the environment ***DeepDesign_acc*** following the instruction in the ***Environments*** folder.
* Download and install the ***TMalign*** software in this folder from  https://zhanglab.ccmb.med.umich.edu/TM-align/.

## Load Environment:
```
module load Anaconda/2-5.0.1
source activate DeepDesign_acc
module load cuDNN/5.1-CUDA-8.0.44
```

## Fold Represenation: 
Given a certain protein structure (***pdb file***), to apply our model to generate sequences, firstly we need to represent the structure with a coordinate. Run the following code and the coordinate will be writen in the ***coordinate file***.
```
python pdb_representation.py <pdb file>  <coordinate file>
```
## Generate Sequences
* **Generate Sequences without Oracle based on Our Best Check Points:**
```
python Random_Generator.py <coordinate file>  <seqeunces number>  <sequence file> 
```
* **Generate Sequences without Oracle based on Other Check Points:**
```
python Random_Generator.py <coordinate file>  <seqeunces number>  <sequence file>  <check point path>  <epoch number>
```
* **Generate Sequences that pass the Oracle based on Our Best Check Points:**
```
python Success_Generator.py <coordinate file>  <seqeunces number>  <sequence file> 
```
* **Generate Sequences that pass the Oracle based on Our Best Check Points:**
```
python Success_Generator.py <coordinate file>  <seqeunces number>  <sequence file>  <check point path>  <epoch number>
```

By changing the ***sequence number*** you can adjust the number of sequences you want, and all the sequences will be shown as Fasta format in ***sequence file***.
