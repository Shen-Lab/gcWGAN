# Oracle

This part contains the materials of the oracles. ***DeepSF_modified*** is for our modified oracle, and ***DeepSF_origin*** is for the original DeepSF.

## Operations:
* **Load Environment:**
```
module load Anaconda/2-5.0.1
source activate tensorflow_training
```
* **To train the model:**
```
python cWGAN_Training.py <learning rate>  <critic iteration number>  <noise length>  
```
   Notice: 3 arguments except the script name.
* **To continue training the model:** 
```
python cWGAN_Continue_Train.py <learning rate>_<critic iteration number>_<noise length>  
```
   Notice: only one argument except the script name.
* **Hyper-parameter Tuning:** Go to the ***Hyper-parameter_Tuning*** folder. Follow the instructions and compare the results to select the hyper-parameter.
