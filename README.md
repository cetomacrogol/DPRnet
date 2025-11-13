# DPRnet
DPRnet is a deep learning framework designed to predict candidate compounds capable of reversing tumor cell drug resistance. The model integrates molecular graph data, SMILES embeddings, and gene expression profiles, leveraging multi-head attention mechanisms to capture cross-modal relationships. DPRnet provides a robust and interpretable approach for identifying potential resistance reversers. Furthermore, DPRnet identifies chemical substructures associated with resistance reversal, providing valuable insights for reversal agent discovery.
# Requirements
torch>=1.7.1
dgl>=0.7.1
dgllife>=0.2.8
numpy>=1.20.2
scikit-learn>=0.24.2
pandas>=1.2.4
prettytable>=2.2.1
rdkit~=2021.03.2
