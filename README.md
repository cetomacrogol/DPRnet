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

The `datasets` folder contains the experimental data used in this framework, which is sourced from [ChemBL](https://www.ebi.ac.uk/chembl/). Additionally, a pre-trained model, [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1), is utilized to process molecular representations for the task.


## Run DPRnet on Our Data to Reproduce Results

To trainDPRnet, where we provide the basic configurations for all hyperparameters in `config.py`. 


```
$ python main.py --cfg "configs/DPRnet.yaml" --data ${DRtest} 
```


## Comet ML
[Comet ML](https://www.comet.com/site/) is an online machine learning experimentation platform, which help researchers to track and monitor their ML experiments. We provide Comet ML support to easily monitor training process in our code.
This is **optional to use**. If you want to apply, please follow:

- Sign up [Comet](https://www.comet.com/site/) account and install its package using `pip3 install comet_ml`. 
   
- Save your generated API key into `.comet.config` in your home directory, which can be found in your account setting. The saved file format is as follows:

```
[comet]
api_key=YOUR-API-KEY
```

- Set `_C.COMET.USE` to `True` and change `_C.COMET.WORKSPACE` in `config.py` into the one that you created on Comet.




For more details, please refer the [official documentation](https://www.comet.com/docs/python-sdk/advanced/).

## Acknowledgements
This implementation is inspired and partially based on earlier works [2], [4] and [5].

## Citation
Please cite our [paper](https://arxiv.org/abs/2208.02194) if you find our work useful in your own research.
```
    @article{bai2023drugban,
      title   = {Interpretable bilinear attention network with domain adaptation improves drug-target prediction},
      author  = {Peizhen Bai and Filip Miljkovi{\'c} and Bino John and Haiping Lu},
      journal = {Nature Machine Intelligence},
      year    = {2023},
      publisher={Nature Publishing Group},
      doi     = {10.1038/s42256-022-00605-1}
    }
```

## References
    [1] Liu, Tiqing, Yuhmei Lin, Xin Wen, Robert N. Jorissen, and Michael K. Gilson (2007). BindingDB: a web-accessible database of experimentally determined protein–ligand binding affinities. Nucleic acids research, 35(suppl_1), D198-D201.
    [2] Huang, Kexin, Cao Xiao, Lucas M. Glass, and Jimeng Sun (2021). MolTrans: Molecular Interaction Transformer for drug–target interaction prediction. Bioinformatics, 37(6), 830-836.
    [3] Chen, Lifan, et al (2020). TransformerCPI: improving compound–protein interaction prediction by sequence-based deep learning with self-attention mechanism and label reversal experiments. Bioinformatics, 36(16), 4406-4414.
    [4] Kim, Jin-Hwa, Jaehyun Jun, and Byoung-Tak Zhang (2018). Bilinear attention networks. Advances in neural information processing systems, 31.
    [5] Haiping Lu, Xianyuan Liu, Shuo Zhou, Robert Turner, Peizhen Bai, ... & Hao Xu (2022). PyKale: Knowledge-Aware Machine Learning from Multiple Sources in Python. In Proceedings of the 31st ACM International Conference on Information and Knowledge Management (CIKM).
