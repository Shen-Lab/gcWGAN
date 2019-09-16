# Data

## Fold Representation
### Operations:
* **Run "scripts/data_curation.py" to get representaive pdbs for each fold.**
* **Run "scripts/represent.py" to obtain *folds_coordinate* and *folds_basis*.**

### Result Files:

Check the folder ***Datasets/Fold_representation/verison04/*** .

* **folds_coordinate:** 20-dimensional coordinates for 1232 folds

* **folds_basis:** The basis vectors of 20-dimensional fold space

***

## Training, Test, Validation Sets:

Check the folder ***Datasets/Final_representation/verison04/*** .
    
* **seq_train:** Include all the sequences in the training set. 

* **fold_train:** The corresponding folds for each sequence in 'seq_train' file.

* **fold_val:**  Include all the folds in the validation set.

* **fold_test:** Include all the folds in the test set.

***

## Final Data

Contain the data we directly applied for the model training, validation and test process.

***

## Data Augmentation:

**Link:** https://drive.google.com/open?id=1icWlx31gcA_61bXuogssn7SfbROaGxea

* **Augment data by deleting the beginning and the end of the sequence based on its secondary structure.**
* **Augment b into d and n, z into q and e.**
* **Replace x with a.**
* **Delete those seqs including X or having more than two x.**
