# HFGNN-DTA (Hierarchical Functional-Group GNN for Drug Target Affinity Prediction)
Integrating Functional-Group Context into Graph Neural Networks for Drug-Target Affinity Prediction

**Data Organization**

All preprocessed data are available in the 'data' folder for the Davis and KIBA datasets.

Split datasets for all four strategies (S1 - Standard split, S2 - Cold-drug split, S3 - Cold-target split, S4 - Cold-drug-target split) are provided in the 'data/{dataset}/split_data' folder:

Training: train_SX.csv/
Validation: val_SX.csv/
Test: test_SX.csv
(Replace X with 1, 2, 3, or 4 to indicate the split strategy.)

This structure ensures easy access to all datasets for reproducing experiments across standard and cold-start scenarios.

**Data preprocessing**

First, generate PSSM features and contact map files of proteins using run_pssm.py and run_pconsc4.py files for each dataset.

python run_pssm.py

python run_pconsc4.py

**Train and test the model**

python training.py x y z

python testing.py x z

x - dataset selection [0- Davis, 1- KIBA]

y - number of training epochs (use the values reported in the paper)

z - data  split strategy (S1, S2, S3, S4)
