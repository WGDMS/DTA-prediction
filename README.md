# DTA-prediction
Drug Target Affinity Prediction

**Data preprocessing**

First, generate PSSM features and contact map files of proteins using run_pssm.py and run_pconsc4.py files for each dataset.


**Train and test the model**

python training.py x y

python testing.py x y

x - database selection [0- Davis, 1- KIBA]

y- no of epochs
