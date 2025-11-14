# DPRnet

## Introduction

DPRnet is a deep learning framework designed to predict candidate drugs that can reverse tumor cell drug resistance. The model integrates molecular graph data, SMILES embeddings, and gene expression profiles, utilizing a multi-head attention mechanism to capture cross-modal relationships. Deeprnet leverages the ChemBERTa-zinc-base-v1 pre-trained model, a state-of-the-art transformer-based model for molecular representations.
This robust and interpretable framework has been validated on independent datasets and demonstrated high prediction accuracy. DPRnet also identifies chemical substructures associated with drug resistance reversal, providing valuable insights for drug discovery.
We also provide an interactive web platform where users can submit their data and perform predictions.

## System Requirements
The source code developed in Python 3.8 using PyTorch 1.7.1. The required python dependencies are given below. 

```
torch>=1.7.1
dgl>=0.9.1
dgllife>=0.2.8
numpy>=1.20.2
scikit-learn>=0.24.2
pandas
rdkit~=2022.3.3
```

## Datasets
Here’s a suggestion for your README:

The `datasets` folder contains the experimental data used in this framework, which is sourced from [ChEMBL](https://www.ebi.ac.uk/chembl/). Additionally, a pre-trained model, [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1), is utilized to process molecular representations for the task.


## Run DPRnet on Our Data to Reproduce Results

To trainDPRnet, where we provide the basic configurations for all hyperparameters in `config.py`. 


```
$ python main.py --cfg "configs/DPRnet.yaml" --data DRtest
```


