# gcWGAN

## Traning Process
![Training-Process](Training-Process.png)

## Pre-requisite
* Build the environment ***DeepDesign_acc*** following the instruction in the ***Environments*** folder.

## Table of contents:
* **gcWGAN_training.py:** Main code for gcWGAN Training.

## Operations:
* **Load and Set Environment**
```
module load Anaconda/2-5.0.1
source activate DeepDesign_acc
module load cuDNN/5.1-CUDA-8.0.44
```
Open the file ***~/.keras/keras.json***, and set "backend" to be "tensorflow", which makes the file looks like:
```
{
    "epsilon": 1e-07,
    "floatx": "float32",
    "image_dim_ordering":"tf",
    "image_data_format": "channels_last",
    "backend": "tensorflow"
}
```
* **Warm Start:** Go to the ***WarmStart*** folder. Firstly run 
```
python Pretrain_WGAN.py  
```
Then run
```
python Semi_WGAN  
```
* **To train the model:**
```
python gcWGAN_Training.py <learning rate>  <critic iteration number>  <noise length>  <feedback penalty weight>
```
