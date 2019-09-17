# Yield Ratio:
Yield ratio reflects the portion of the sequences that pass the oracle, which can reflect the model accuracy. To calculate the yield ratio, read the following steps. 
* **Load and Set Environment**
```
module load Anaconda/2-5.0.1
source activate DeepDesign_acc
module load cuDNN/5.1-CUDA-8.0.44
```
Open the file ***~/.keras/keras.json***, and set "backend" to be "theano", which makes the file looks like:
```
{
    "epsilon": 1e-07,
    "floatx": "float32",
    "image_dim_ordering":"tf",
    "image_data_format": "channels_last",
    "backend": "theano"
}
```
***

## For cWGAN and gcWGAN:

* **Yield Ratio of a Certain Fold base on a Certain Check point:** 
```
python Yield_Ratio_calculation.py <check point path>  <epoch index>  <fold> 
```
The results can be found in the folder ***../../../Results/Accuracy/Yield_Ratio_Result/\<model name\>/model_\<model index\>/*** with a name of the fold name. "model name" is "cWGAN" or "gcWGAN". "model index" is related to the name where you put the check points and the epoch index.

* **Get the Statistics of the Yiled Ratios:** To run this step, you must have calculated the yield ratios for all the folds in the training, the validation and the test set. 
```
python Yield_Ratio_Statistic.py <check point path>  <epoch index>
```
The results can be found in the folder ***../../../Results/Accuracy/Yield_Ratio_Result/\<model name\>/model_\<model index\>/\<set name\>/*** , while "set name" = "train", "vali" or "test".

***

## For cVAE:

For comparison we also applied the previous state-of-art model, [cVAE](https://github.com/psipred/protein-vae) to calculate the yield ratios. To do this, firstly go to the ***Yield_Ratio_Samples/cVAE_Samples*** folder and download the sequences inside it with the link below. Then run the following codes.

Link: https://drive.google.com/drive/folders/1hlaJ3Q5KRcz8ZQ6qyG6GsU8sH7xL6iip

* **Yield Ratio of a Certain Fold base on cVAE:** 
```
python Yield_Ratio_calculation_cVAE.py <fold> 
```
The results can be found in the folder ***../../../Results/Accuracy/Yield_Ratio_Result/cVAE/*** with a name of the fold name. To run the following steps, you must have calculated the yield ratios for all the folds in the training, the validation and the test set based on cVAE.
* **Arrange the Results inside a Dictionary File:**
```
python cVAE_dictionary.py 
```
The dictionary file can be found in the folder ***../../../Results/Accuracy/Yield_Ratio_Result/cVAE/YR_statistic/*** .
* **Get the Statistics of the Yiled Ratios for cVAE:**  
```
python Yield_Ratio_Statistic_cVAE.py 
```
The results can be found in the folder ***../../../Results/Accuracy/Yield_Ratio_Result/cVAE/YR_statistic/\<set name\>/*** , while "set name" = "train", "vali" or "test".
