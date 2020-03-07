# Data

## Datasets
### Origin_SCOPE
Contain Oringinal protein folds and sequences information we downloaded from SCOPE, including the original 1,232 pdb files.

### Origin_uniref
Contain Oringinal protein sequences information we downloaded from uniref which we applied for the gcWGAN warm start.

### Intermediate_Data
Contain the intermediate data such as TM-score matrix and sequence identity matrices that we generated in order to get the final data applied for the model.

### Final Data
Contain the data we directly applied for the model training, validation and test process.

* **folds_coordinate:** 20-dimensional coordinates for 1232 folds.

* **folds_basis:** The basis vectors of 20-dimensional fold space.

* **pdbs:** Contain the 1,232 pdb files that we deleted the "TER"s inside them to avoid the errors in TM alignment.

***

## Data_Process

Contain the codes to generate the intermediate data and final data during the data process.

***
