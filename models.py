import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dgllife.model.gnn import GCN
from torch.nn.utils.weight_norm import weight_norm
from transformers import AutoModel, AutoTokenizer


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class DRPNet(nn.Module):
    def __init__(self, **config):
        super(DRPNet, self).__init__()
        compound_in_feats = config["COMPOUND"]["NODE_IN_FEATS"]
        compound_embedding = config["COMPOUND"]["NODE_IN_EMBEDDING"]
        compound_hidden_feats = config["COMPOUND"]["HIDDEN_LAYERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        compound_padding = config["COMPOUND"]["PADDING"]
        smiles_padding = config["SMILES"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        chemberta_model_path = config["CHEMBERTA"]["MODEL_PATH"]
        custom_output_dim = config["CHEMBERTA"]["OUTPUT_DIM"]
        cell_expression_in_feats = config["CELL_EXPRESSION"]["IN_FEATS"]
        cell_expression_hidden_feats = config["CELL_EXPRESSION"]["HIDDEN_LAYERS"]

    
        self.compound_extractor = MolecularGCN(in_feats=compound_in_feats, dim_embedding=compound_embedding,
                                               padding=compound_padding, hidden_feats=compound_hidden_feats)
        self.smiles_extractor = ChemBERTaCompoundModel(chemberta_model_path, custom_output_dim=custom_output_dim)

        self.drug_extractor = MolecularGCN(in_feats=compound_in_feats, dim_embedding=compound_embedding,
                                           padding=compound_padding, hidden_feats=compound_hidden_feats)
        self.drug_smiles_extractor = ChemBERTaCompoundModel(chemberta_model_path, custom_output_dim=custom_output_dim)

        self.compound_total_dim = custom_output_dim + compound_hidden_feats[-1]
        self.drug_total_dim = custom_output_dim + compound_hidden_feats[-1]
        self.compound_fc = nn.Linear(self.compound_total_dim, 128)
        self.drug_fc = nn.Linear(self.drug_total_dim, 128)

        self.cell_feature_extractor = CellFeatureExtractor(in_feats=cell_expression_in_feats,
                                                           hidden_feats=cell_expression_hidden_feats)
        self.multihead_attention = MultiheadAttentionModule(
            atom_embed_dim=128,  
            global_embed_dim=384,  
            num_heads=8,  
            dropout=0.1 
        )

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, compound_graph, compound_smiles, drug_graph, drug_smiles, cell_expression_tensor, mode="train"):
        atom_features = self.compound_extractor(compound_graph)  # (batch, num_atoms, 128)
        smiles_features = self.smiles_extractor(compound_smiles)  # (batch, 128)

        drug_gcn_features = self.drug_extractor(drug_graph)  # (batch, num_drug_atoms, 128)
        drug_gcn_features = torch.mean(drug_gcn_features, dim=1)  # (batch, 128)
        drug_smiles_features = self.drug_smiles_extractor(drug_smiles)  # (batch, 128)
        drug_processed = self.drug_fc(torch.cat([drug_gcn_features, drug_smiles_features], dim=1))  # (batch, 128)

        cell_features = self.cell_feature_extractor(cell_expression_tensor)  # (batch, 128)

        global_features = torch.cat([smiles_features, cell_features, drug_processed], dim=-1)  # (batch, 384)

        compound_fused, atom_weights = self.multihead_attention(
            atom_features,  # (batch, num_atoms, 128)
            global_features  # (batch, 384)
        )

        score = self.mlp_classifier(compound_fused)  # (batch, 1)

        return score, atom_weights

class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata['h'].clone()
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class ChemBERTaCompoundModel(nn.Module):
    def __init__(self, chemberta_model_path, custom_output_dim):
        super(ChemBERTaCompoundModel, self).__init__()
        self.chemberta_model = AutoModel.from_pretrained(chemberta_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(chemberta_model_path)
        self.projection_layer = nn.Linear(self.chemberta_model.config.hidden_size, custom_output_dim)

    def forward(self, tokenized_smiles):
        outputs = self.chemberta_model(**tokenized_smiles)
        cls_token_output = outputs.last_hidden_state[:, 0, :]
        projected_output = self.projection_layer(cls_token_output)
        return projected_output


class CellFeatureExtractor(nn.Module):
    def __init__(self, in_feats, hidden_feats, dropout=0.2):
        super(CellFeatureExtractor, self).__init__()
        layers = []
        in_dim = in_feats
        for out_dim in hidden_feats:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(128, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class MLP(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super(MLP, self).__init__()
        layers = []
        in_dim = in_feats
        for out_dim in hidden_feats:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MultiheadAttentionModule(nn.Module):
    def __init__(self, atom_embed_dim=128, global_embed_dim=384, num_heads=8, dropout=0.1):
        super(MultiheadAttentionModule, self).__init__()
        self.atom_embed_dim = atom_embed_dim
        self.global_embed_dim = global_embed_dim
        self.num_heads = num_heads

        self.atom_q_proj = nn.Linear(atom_embed_dim, atom_embed_dim)
        self.atom_k_proj = nn.Linear(atom_embed_dim, atom_embed_dim)

        self.global_v_proj = nn.Linear(global_embed_dim, atom_embed_dim)

        self.atom_importance = nn.Sequential(
            nn.Linear(atom_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=atom_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.fusion = nn.Sequential(
            nn.Linear(atom_embed_dim * 2, atom_embed_dim),
            nn.ReLU(),
            nn.LayerNorm(atom_embed_dim)
        )

    def forward(self, atom_features, global_features):
        batch_size, num_atoms, _ = atom_features.shape

        q = self.atom_q_proj(atom_features)  # (batch, num_atoms, 128)
        k = self.atom_k_proj(atom_features)  # (batch, num_atoms, 128)

        v = self.global_v_proj(global_features).unsqueeze(1)  # (batch, 1, 128)
        v = v.expand(-1, num_atoms, -1)  # (batch, num_atoms, 128)

        attn_output, attn_weights = self.multihead_attn(
            q, k, v,
            need_weights=True
        )  # attn_weights: (batch, num_atoms, num_atoms)

        atom_importance = attn_weights.mean(dim=1)  # (batch, num_atoms)

        atom_importance = atom_importance + torch.sigmoid(self.atom_importance(atom_features)).squeeze(-1)
        atom_importance = atom_importance / atom_importance.sum(dim=1, keepdim=True)  # 归一化

        weighted_atoms = (atom_features * atom_importance.unsqueeze(-1)).sum(dim=1)  # (batch, 128)

        fused_features = self.fusion(torch.cat([
            weighted_atoms,
            self.global_v_proj(global_features)
        ], dim=-1))  

        return fused_features, atom_importance
    



