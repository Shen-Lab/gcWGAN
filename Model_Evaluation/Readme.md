# Model Evaluation

## Pre-requisite
* Build the environment ***DeepDesign_acc*** following the instruction in the ***Environments*** folder.
* Build the environment ***tensorflow_training*** following the instruction in the ***Environments*** folder.

***

## Yield Ratio Analysis:
Model Accuracy reflects that how likely our model can generate sequences folding into the target folds and how similar the structure prediction of the generated ones can be to the ground truth. We applied two methods to measure the model accuracy, the **Yield Ratio Analysis** and the **Rosetta Analysis**. **Yield Ratio** reflects the portion of the sequences that pass the oracle, which can reflect the model accuracy. For comparison we also applied the previous state-of-art model, [cVAE](https://github.com/psipred/protein-vae) to calculate the yield ratios. Go to the ***Accuracy/Yield_Ratio_Analysis*** folder for more details. [Rosetta](https://www.rosettacommons.org/docs/latest/application_documentation/structure_prediction/abinitio-relax) is a software that can can be applied for sequence structure prediction. Based on gcWGAN we generated 10 sequences that pass the oracle for 6 selected folds and a novel fold, and then did the same based on cVAE. Then we applied the process in the ***Accuracy/Rosetta_Analysis*** folder for comparison. For more details or apply the process, go to the ***Accuracy/Rosetta_Analysis*** folder.

***

## Generate Sequences:
This section contains the scripts to generate sequences according to different conditions for other analysis. It also contains the scripts to measure the sequence genberating rate, which reflect the rate to generate protein sequences that can pass the oracle while gernerating sequences based on cWGAN or gcWGAN. This process shows the improvement from cWGAN to gcWGAN.

***

## Sequence Diversity&Novelty:
Besides the model accuracy, another goal of our model is to discover new protein sequences for known folds and sequences for novel folds. Therefore we hope that our generated sequences keep a high diversity and high novelty. To apply this section, go to the ***Diversity_and_Novelty*** folder for more details.

***

## Result Path
The scripts will autoimatically create a folder named ***Result*** outside this folder if it does not exist, and all the generated model evaluation results will be put inside it. For each section it will create its own result paths, and the result paths are illustrated in the instruction of each section seperately.
