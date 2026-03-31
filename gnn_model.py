
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from rdkit import Chem
import torch

class GCNModel(nn.Module):
    def __init__(self):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(6, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def combine_graphs(drug_smiles, exc_smiles, label=0):
    def mol_to_graph(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        atom_features = []
        for atom in mol.GetAtoms():
            at = atom.GetAtomicNum()
            feat = [0]*6
            feat[at % 6] = 1
            atom_features.append(feat)
        x = torch.tensor(atom_features, dtype=torch.float)
        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        if len(edges) == 0:
            edge_index = torch.empty((2,0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return x, edge_index

    x1, edge1 = mol_to_graph(drug_smiles)
    x2, edge2 = mol_to_graph(exc_smiles)
    if x1 is None or x2 is None:
        return None
    x = torch.cat([x1, x2], dim=0)
    edge2_shifted = edge2 + x1.shape[0]
    edge_index = torch.cat([edge1, edge2_shifted], dim=1)
    y = torch.tensor([label], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data
