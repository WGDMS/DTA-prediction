# HFGNN-DTA (Hierarchical Functional-Group GNN for Drug-Target Affinity Prediction)

This repository contains the implementation of the paper:
**HFGNN-DTA: Integrating Functional-Group Context into Graph Neural Networks for Drug-Target Affinity Prediction**

## Data Organization

The preprocessed Davis and KIBA datasets are available in the `data` directory.

Split datasets for all four strategies *(S1 - Standard split, S2 - Cold-drug split, S3 - Cold-target split, S4 - Cold-drug-target split)* are provided in the `data/{dataset}/split_data` folder, where {dataset} is either davis or kiba.

Training: **train_SX.csv**/
Validation: **val_SX.csv**/
Test: **test_SX.csv**
(Replace X with 1, 2, 3, or 4 to indicate the split strategy.)

This structure ensures easy access to all datasets for reproducing experiments across standard and cold-start scenarios.

## Data Preprocessing

Protein position-specific scoring matrix (PSSM) features and residue contact maps can be generated using:
`run_pssm.py`
`run_pconsc4.py`
Run the preprocessing scripts separately for the Davis and KIBA datasets.

*PconsC4 Environment*
Create the Conda environment required for PconsC4-based contact-map generation:
conda env create -f pconsc4_env.yml
Activate the environment:
conda activate pconsc4_env
The pconsc4_env.yml file specifies the dependencies required to run PconsC4.
Generate PSSM Features
python run_pssm.py
Generate Protein Contact Maps
python run_pconsc4.py

 
### Train and test the model
```bash
python training.py x y z

```bash
python testing.py x z

x - dataset selection [0- Davis, 1- KIBA]

y - number of training epochs (as mentioned in the supporting document)

z - data  split strategy (S1, S2, S3, S4)
