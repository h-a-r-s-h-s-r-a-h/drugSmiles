import torch
import numpy as np
import argparse
from models.gnn_model import DrugGNN, create_molecular_graph

def predict_properties(smiles):
    """
    Predict molecular properties for a given SMILES string
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model with same architecture as training
    model = DrugGNN(
        num_features=8,  # Number of atom features
        hidden_channels=64,  # Hidden channels from train.py
        num_classes=1  # Single output for logP
    ).to(device)
    
    # Load model weights
    try:
        model.load_state_dict(torch.load('drug_gnn_model.pt', weights_only=True))
    except FileNotFoundError:
        return "Error: Model file 'drug_gnn_model.pt' not found. Please train the model first."
    
    # Load scaler parameters if available
    try:
        scaler_mean = np.load('models/logp_scaler_mean.npy')
        scaler_scale = np.load('models/logp_scaler_scale.npy')
    except FileNotFoundError:
        print("Warning: Scaler files not found. Using default values.")
        scaler_mean = np.array([0.0])
        scaler_scale = np.array([1.0])
    
    model.eval()
    
    # Create graph from SMILES
    graph = create_molecular_graph(smiles)
    if graph is None:
        return "Invalid SMILES string"
    
    # Make prediction
    with torch.no_grad():
        graph = graph.to(device)
        prediction = model(graph)
        
        # Inverse transform the prediction
        scaled_pred = prediction.cpu().numpy().reshape(-1)
        original_pred = (scaled_pred * scaler_scale[0]) + scaler_mean[0]
        
    return {
        'logP': float(original_pred[0])
    }

def main():
    parser = argparse.ArgumentParser(description='Predict molecular properties')
    parser.add_argument('--input', type=str, required=True, help='SMILES string of the molecule')
    args = parser.parse_args()
    
    # Make prediction
    results = predict_properties(args.input)
    
    if isinstance(results, str):
        print(results)
    else:
        print("\nPredicted Properties:")
        print(f"LogP: {results['logP']:.2f}")

if __name__ == "__main__":
    main() 