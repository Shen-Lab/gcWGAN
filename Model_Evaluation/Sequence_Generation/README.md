# Generate Sequences:

Generate sequences based on our best check points for model evaluation. Go to the ***Pipeline_Sequence_Generation*** folder.
* **Load Environment:**
```
module load Anaconda/2-5.0.1
source activate DeepDesign_acc
module load cuDNN/5.1-CUDA-8.0.44
```

### Sequence Generator
* **Generate Sequences for a Known Fold (a fold that belongs to those we applied for data representation) without the Oracle:**
```
python Known_Random_Generator.py <sequences number>  <fold name>  <model kind>  <minimum length>  <maximum length>
```
* **Generate Sequences for a Known Fold that pass the Oracle:**
```
python Known_Success_Generator.py <sequences number>  <fold name>  <model kind>  <minimum length>  <maximum length>
```
* **Generate Sequences for the Novel Fold without the Oracle:**
```
python Nov_Random_Generator.py <sequences number>  <model kind>  <minimum length>  <maximum length>
```
* **Generate Sequences for the Novel Fold that pass the Oracle:**
```
python Nov_Success_Generator.py <sequences number>  <model kind>  <minimum length>  <maximum length>
```

### Generating Rate
* **Generate Sequences and record the successful indexes:**
```
python GenRate_All.py <model kind>  <fold name>  <sequences number>  <minimum length>  <maximum length>
```
* **Generate Sequences with fixed successful number and record the successful indexes:**
```
python GenRate_Success.py   <model kind>  <fold name>  <success number>  <minimum length>  <maximum length>
```
* **Visualize the successful sequence indexes to show the generating rate:**

Before this step, do the **Generate Sequences and record the successful indexes** for fold *b.2*, *c.56* and *c.94* with *sequence number = 100,000*, and the **Generate Sequences with fixed successful number and record the successful indexes** for fold *a.35*, *d.107* and *g.44* with *success number = 200*.
```
python Slope_Generate_stat_plot.py
```
