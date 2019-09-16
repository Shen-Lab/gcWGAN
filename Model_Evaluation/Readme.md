# Model Evaluation

## Pre-requisite
* Build the environment ***DeepDesign_acc*** following the instruction in the ***Environments*** folder.
* Build the environment ***tensorflow_training*** following the instruction in the ***Environments*** folder.

***

## Model Accuracy:
Model Accuracy reflects that how likely our model can generate sequences folding into the target folds and how similar the structure prediction of the generated ones can be to the ground truth. For model accuracy measurement we calculted the yield ratio, the portion of the sequences that pass the oracle, and applied [Rosetta](https://www.rosettacommons.org/home) to predict the strcuture of our generated sequences so that we can compare the predictions with the ground truth. Go to folder ***Accuracy** for more details and the scripts we applied for yield ratio analysis.

***

## Generate Sequences:
This section contains the scripts to generate sequences according to different conditions for other analysis. It also contains the scripts to measure the sequence genberating rate, which reflect the rate to generate protein sequences that can pass the oracle while gernerating sequences based on cWGAN or gcWGAN. This process shows the improvement from cWGAN to gcWGAN.

***

## Sequence Diversity&Novelty:
Besides the model accuracy, another goal of our model is to discover new protein sequences for known folds and sequences for novel folds. Therefore we hope that our generated sequences keep a high diversity and high novelty. To apply this section, go to the ***Diversity_and_Novelty*** folder for more details.

***

## Result Path
The scripts will autoimatically create a folder named ***Result*** outside this folder if it does not exist, and all the generated model evaluation results will be put inside it. For each section it will create its own result paths, and the result paths are illustrated in the instruction of each section seperately.
