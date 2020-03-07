# gcWGAN Hyper-parameter Tuning
There are 1 hyper-parameter remaining for gcWGAN, the weight of the feedback penalty. With the following steps we got the results in the ***gWGAN_Validation_Results*** folder.

## Operations:
* **Load Environment:**
```
module load Anaconda/2-5.0.1
source activate DeepDesign_acc
```
* **Get the Results for Different Models:**
Calculate nonsense sequence ratio:
```
python Nonsense_Ratio_gcWGAN.py <learning rate>_<critic iteration number>_<noise length>  <set kind>
```
Calculate nonsense sequence ratio on generated samples:
```
python Nonsense_Ratio_recalculation_gcWGAN.py <learning rate>_<critic iteration number>_<noise length>  <set kind>
```
Calculate sequence identities, padding ratios and sequence stabilities:
```
python Iden_PR_Stab_gcWGAN.py <learning rate>_<critic iteration number>_<noise length>  <set kind>
```
* **Show the Results** 
Plot nonsense sequence ratio:
```
plot_NonsenseRatio_diffsets.py <set kind>
```
or (if the nonsense sequence ratios are calculated on genrated sequences):
```
plot_recaNR_diffsets.py <set kind>
```

Plot other criterias:
```
plot_Iden_PR_Stab_diffsets.py <set kind>
```
