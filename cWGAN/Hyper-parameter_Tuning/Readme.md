# cWGAN Hyper-parameter Tuning
There are 3 hyper-parameters for cWGAN, the initial learning rate, the critic iteration number and the noise length. Due to the calculation burden we tune them sequentially. With the following steps we got the results in the ***cWGAN_Validation_Results*** folder.

## Table of contents:
* **cWGAN_loss_calculate.py:** Calculate the model loss for a certain model on a certain set (training, validation or test).
* **Nonsense_Ratio.py:** Calculate the nonsense sequence ratios for a certain model.
* **Nonsense_Ratio_recalculation.py:** Recalculate the nonsense sequence ratios based on previous samples for a certain model on a certain set.
* **Iden_and_PR_diff_sets.py:** Calculate the sequences identities and padding ratios for a certain model on a certain set.
* **plot_learning_rate_diffsets.py:** Show the results for different initial learning rates.
* **plot_critic_iteration_diffsets.py:** Show the results for different critic iteration numbers.
* **plot_noise_length_diffsets.py:** Show the results for different noise lengths.

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
