# Model Accuracy:

Model Accuracy reflects that how likely our model can generate sequences folding into the target folds and how similar the structure prediction of the generated ones can be to the ground truth. We calculated the yield ratio of the folds and applied software [Rosetta](https://www.rosettacommons.org/home) to get the predictions for accuracy measurement. Yield ratio reflects the portion of the sequences that pass the oracle, which can reflect the model accuracy. For comparison we also applied the previous state-of-art model, [cVAE](https://github.com/psipred/protein-vae) to calculate the yield ratios. To calculate the yield ratio, go to the ***Yield_Ratio_Analysis*** folder and follow the instructions. 
 
