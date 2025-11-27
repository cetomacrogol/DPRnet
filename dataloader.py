import pandas as pd
import torch.utils.data as data
import torch
import os
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290, data_type=None):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.data_type = data_type
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

        self.cell_expression_data = pd.read_csv('cancer_gene_effects_all.csv')
        self.cell_expression_data.set_index('gene', inplace=True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        compound_smiles = self.df.iloc[index]['compound_smiles']
        compound_graph = self.fc(smiles=compound_smiles, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        compound_graph = self._pad_graph(compound_graph)

        drug_smiles = self.df.iloc[index]['drug_smiles']
        drug_graph = self.fc(smiles=drug_smiles, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        drug_graph = self._pad_graph(drug_graph)

        y = self.df.iloc[index]["label"]

        cell = self.df.iloc[index]['cell']

        cell_expression = self.cell_expression_data[cell].values
        cell_expression_tensor = torch.tensor(cell_expression, dtype=torch.float32)

        return compound_graph, compound_smiles, drug_graph, drug_smiles, y, cell_expression_tensor

    def _pad_graph(self, graph):
        actual_node_feats = graph.ndata['h'].clone()
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        graph.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        graph.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        graph = graph.add_self_loop()
        return graph


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
