# Sequence Diversity and Novelty:

Besides the model accuracy, another goal of our model is to discover new protein sequences for known folds and sequences for novel folds. Therefore we hope that our generated sequences keep a high diversity and high novelty.

* **Load Environment:**
```
module load Anaconda/2-5.0.1
source activate tensorflow-training
```

***

Before the following steps, go to the ***../Sequence_Generation*** folder and generate 100 sequences for the fold to be tested.

* **Calculate the pairwise sequence identities to show the sequence diversity:** 

(fold name = *nov*, *a.1*, ...; model = *cWGAN*, *gcWGAN* or *cVAE*; If model = *cWGAN* or *gcWGAN*, kind = *random* or *success*, otherwise kind is not necessary.)
```
python Diversity_calculation.py <fold name>  <model>  (<kind>)
```

* **Calculate the maximum sequence identities between the generated sequences and natural ones to show the sequence novelty:**

(fold name = *nov*, *a.1*, ...; model = *cWGAN*, *gcWGAN* or *cVAE*; If model = *cWGAN* or *gcWGAN*, kind = *random* or *success*, otherwise kind is not necessary.)
```
python Novelty_calculation.py <fold name>  <model>  (<kind>) 
```

The sequence diversity and novelty results can be found in the folder ***../../Results/Diversity_and_Novelty/*** .

* **Plot the distribution of the sequence identities:**
 
(Before this step, must finish the previous ones. model = *cWGAN* or *gcWGAN*; fold name = *nov*, *a.1*, ...)
```
python Diversity_Novelty_plot.py <model>  <fold name>
```
The figures can be found in the folder ***../../Results/Diversity_and_Novelty/Div_Nov_Image/*** .

* **More flexible version:**
The flexible version just takes the sequence file as the input, so we can do the analysis for more conditions not only the two cases that whether the sequence has been filtered be the oracle.
```
python Diversity_calculation_for_files.py <sequence file> 
python Novelty_calculation_for_files.py <sequence file> 
python Diversity_Novelty_plot_for_files.py  <model>  <fold name>
```
