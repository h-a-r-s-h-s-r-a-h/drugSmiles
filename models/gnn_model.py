import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import torch.nn as nn

def create_molecular_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
        
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            _normalize_atomic_num(atom.GetAtomicNum()),
            _normalize_degree(atom.GetDegree()),
            _normalize_formal_charge(atom.GetFormalCharge()),
            _normalize_hybridization(atom.GetHybridization()),
            atom.GetIsAromatic(),
            _normalize_valence(atom.GetTotalValence()),
            _normalize_num_h(atom.GetTotalNumHs()),
            _normalize_radical_electrons(atom.GetNumRadicalElectrons())
        ]
        atom_features.append(features)
        
    # Get edge indices and features
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]  # Add both directions
        
        # Normalize bond features
        bond_type = _normalize_bond_type(bond.GetBondType())
        edge_features.extend([bond_type] * 2)  # Same features for both directions
        
    # Convert to PyTorch tensors
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float).view(-1, 1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class DrugGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(DrugGNN, self).__init__()
        
        # Single GNN layer with batch norm
        self.conv = GCNConv(num_features, hidden_channels)
        self.bn = nn.BatchNorm1d(hidden_channels)
        
        # Simple output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, num_classes),
            nn.BatchNorm1d(num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Very conservative initialization
            nn.init.uniform_(module.weight, -0.01, 0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, GCNConv):
            # Initialize GCNConv parameters
            nn.init.uniform_(module.lin.weight, -0.01, 0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 0.1)
            nn.init.zeros_(module.bias)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GNN layer with regularization
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Global mean pooling
        x = global_mean_pool(x, batch)
        
        # Output layer
        x = self.output(x)
        
        return x

# Helper functions for feature normalization
def _normalize_atomic_num(atomic_num):
    return atomic_num / 118.0  # Maximum atomic number

def _normalize_degree(degree):
    return degree / 6.0  # Typical maximum degree

def _normalize_formal_charge(charge):
    return (charge + 5) / 10.0  # Normalize to [0,1] range

def _normalize_hybridization(hybridization):
    hybridization_types = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4
    }
    return hybridization_types.get(hybridization, 0) / 4.0

def _normalize_valence(valence):
    return valence / 8.0  # Typical maximum valence

def _normalize_num_h(num_h):
    return num_h / 4.0  # Typical maximum number of hydrogens

def _normalize_radical_electrons(num_radical):
    return num_radical / 4.0  # Typical maximum number of radical electrons

def _normalize_bond_type(bond_type):
    bond_types = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }
    return bond_types.get(bond_type, 0) / 3.0  # Normalize to [0,1] range 