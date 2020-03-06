# cWGAN Hyper-parameter Tuning
There are 3 hyper-parameters for cWGAN, the initial learning rate, the critic iteration number and the noise length. Due to the calculation burden we tune them sequentially. With the following steps we got the results in the ***cWGAN_Validation_Results*** folder.

## Table of contents:
* **Nonsense_Ratio.py:** Calculate the scripts to get the nonsense sequence ratios for a certain model.
* **Identity_and_PaddingRatio.py:** Calculate the scripts to get the average sequences identities and padding ratios for a certain model.
* **plot_learning_rate_eps.py:** Show the results for different initial learning rates.
* **plot_critic_iteration_eps.py:** Show the results for different critic iteration numbers.
* **plot_noise_length_eps.py:** Show the results for different noise lengths.

## Operations:
* **Load Environment:**
```
module load Anaconda/2-5.0.1
source activate tensorflow_training
```
* **Get the Results for Different Models:**
```
python Nonsense_Ratio.py <learning rate>_<critic iteration number>_<noise length>  
```
```
python Identity_and_PaddingRatio.py <learning rate>_<critic iteration number>_<noise length>  
```
* **Show the Results** 
```
plot_learning_rate_eps.py
```
```
plot_learning_rate_eps.py
```
```
plot_learning_rate_eps.py
```
