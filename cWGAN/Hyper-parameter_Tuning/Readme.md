# cWGAN Hyper-parameter Tuning
There are 3 hyper-parameters for cWGAN, the initial learning rate, the critic iteration number and the noise length. Due to the calculation burden we tune them sequentially. With the following steps we got the results in the ***cWGAN_Validation_Results*** folder.

## Operations:
* **Load Environment:**
```
module load Anaconda/2-5.0.1
source activate tensorflow_training
```
* **Get the Results for Different Models (on the validation set):**
```
python Nonsense_Ratio.py <learning rate>_<critic iteration number>_<noise length>  
```
```
python Nonsense_Ratio_recalculation.py <learning rate>_<critic iteration number>_<noise length>  vali
```
```
python Iden_and_PR_diff_sets.py <learning rate>_<critic iteration number>_<noise length>  vali
```
```
python cWGAN_loss_calculate.py <check point path>  vali  cWGAN_Validation_Results/ 
```
* **Show the Results** 
```
plot_learning_rate_diffsets.py  vali
```
```
plot_critic_iteration_diffsets.py  vali
```
```
plot_noise_length_diffsets.py  vali
```
