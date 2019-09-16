# Model Evaluation

## Pre-requisite
* Build the environment ***DeepDesign_acc*** following the instruction in the ***Environments*** folder.
* Build the environment ***tensorflow_training*** following the instruction in the ***Environments*** folder.

***

## Yield Ratio:
Yield ratio reflects the portion of the sequences that pass the oracle, which can reflect the model accuracy. For comparison we also applied the previous state-of-art model, [cVAE](https://github.com/psipred/protein-vae) to calculate the yield ratios. To calculate the yield ratio, go to the ***Yield_Ratio_Analysis*** folder and follow the instructions.

***

## Generate Sequences:
This section contains the scripts to generate sequences according to different conditions for other analysis. It also contains the scripts to measure the sequence genberating rate, which reflect the rate to generate protein sequences that can pass the oracle while gernerating sequences based on cWGAN or gcWGAN. This process shows the improvement from cWGAN to gcWGAN.

***

## Sequence Diversity&Novelty:
Besides the model accuracy, another goal of our model is to discover new protein sequences for known folds and sequences for novel folds. Therefore we hope that our generated sequences keep a high diversity and high novelty. To apply this section, go to the ***Diversity_and_Novelty*** folder for more details.

***

## Result Path
The scripts will autoimatically create a folder named ***Result*** outside this folder if it does not exist, and all the generated model evaluation results will be put inside it. For each section it will create its own result paths, and the result paths are illustrated in the instruction of each section seperately.
