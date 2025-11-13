# DPRnet
DPRnet is a deep learning framework designed to predict candidate compounds capable of reversing tumor cell drug resistance. The model integrates molecular graph data, SMILES embeddings, and gene expression profiles, leveraging multi-head attention mechanisms to capture cross-modal relationships. DPRnet provides a robust and interpretable approach for identifying potential resistance reversers.DPRnetrnet leverages the ChemBERTa-zinc-base-v1 pre-trained model, a state-of-the-art transformer-based model for molecular representations, to enhance its performance in predicting drug resistance reversal agents. Furthermore, DPRnet identifies chemical substructures associated with resistance reversal, providing valuable insights for reversal agent discovery.
# Requirements
PyTorch
dgl
dgllife
numpy
scikit-learn
pandas
rdkit
# Training the Model
Prepare your training and validation datasets in the format required by the model. 
cancer_gene_effects_all.csv ontains gene expression data for multiple cancer cell lines. This data serves as cellular feature input for the DPRnet model.
Run the training script:
