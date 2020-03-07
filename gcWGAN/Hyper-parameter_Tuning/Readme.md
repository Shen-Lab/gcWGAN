# gcWGAN Hyper-parameter Tuning
There are 1 hyper-parameter remaining for gcWGAN, the weight of the feedback penalty. With the following steps we got the results in the ***gWGAN_Validation_Results*** folder.

## Operations:
* **Load Environment:**
```
module load Anaconda/2-5.0.1
source activate DeepDesign_acc
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
