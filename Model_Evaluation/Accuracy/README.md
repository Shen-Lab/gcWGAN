# Model Accuracy:

## Yield Ratio:
Yield ratio reflects the portion of the sequences that pass the oracle, which can reflect the model accuracy. For comparison we also applied the previous state-of-art model, [cVAE](https://github.com/psipred/protein-vae) to calculate the yield ratios. To calculate the yield ratio, go to the ***Yield_Ratio_Analysis*** folder and follow the instructions. 
 

***

## Rosetta Analysis:
[Rosetta](https://www.rosettacommons.org/home) is a software that can can be applied for sequence structure prediction. Based on gcWGAN we generated 10 sequences that pass the oracle for 6 selected folds and a novel fold, and then did the same based on cVAE. Then we applied the process in the ***Rosetta_Analysis*** folder for comparison. For more details or apply the process, go to the ***Rosetta_Analysis*** folder.
