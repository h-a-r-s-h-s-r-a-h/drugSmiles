import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.gnn_model import DrugGNN, create_molecular_graph
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        pred = pred.squeeze()
        target = target.squeeze()
        return torch.sqrt(torch.mean((pred - target) ** 2))

def load_data():
    """
    Load and prepare the dataset
    """
    # Load processed data
    df = pd.read_csv('data/pubchem_processed.csv')
    
    # Scale the logP values
    scaler = StandardScaler()
    logp_values = df['logp'].values.reshape(-1, 1)
    scaled_logp = scaler.fit_transform(logp_values)
    
    # Save scaler parameters
    np.save('models/logp_scaler_mean.npy', scaler.mean_)
    np.save('models/logp_scaler_scale.npy', scaler.scale_)
    
    # Convert SMILES to graphs
    graphs = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating graphs"):
        graph = create_molecular_graph(row['smiles'])
        if graph is not None:
            # Set the scaled target value
            graph.y = torch.tensor([scaled_logp[idx]], dtype=torch.float)
            graphs.append(graph)
    
    # Split data into train, validation, and test sets
    train_data, temp_data = train_test_split(graphs, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    criterion = RMSELoss()
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            predictions.extend(out.cpu().numpy())
            actuals.extend(batch.y.cpu().numpy())
    
    rmse = total_loss / len(loader)
    r2 = np.corrcoef(np.array(predictions).flatten(), np.array(actuals).flatten())[0, 1] ** 2
    
    return rmse, r2

def train(model, train_loader, val_loader, test_loader, device, num_epochs=100):
    model = model.to(device)
    
    # Use SGD with momentum for better stability
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-4,
        momentum=0.9,
        weight_decay=1e-5,
        nesterov=True
    )
    
    # Use MSE loss with gradient clipping
    criterion = torch.nn.MSELoss()
    
    # Conservative learning rate scheduling
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=15,
        verbose=True,
        min_lr=1e-8
    )
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 25
    min_delta = 1e-6
    min_epochs = 50
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        # Training loop
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass with gradient clipping
            out = model(batch)
            loss = criterion(out, batch.y)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch + 1}")
                continue
            
            # Clip gradients before backward pass
            for param in model.parameters():
                if param.grad is not None:
                    torch.nn.utils.clip_grad_value_(param, 1.0)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
        
        if num_train_batches > 0:
            avg_train_loss = total_train_loss / num_train_batches
        else:
            avg_train_loss = float('nan')
        train_losses.append(avg_train_loss)
        
        # Validation loop
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                val_loss = criterion(out, batch.y)
                
                if not torch.isnan(val_loss):
                    total_val_loss += val_loss.item()
                    num_val_batches += 1
        
        if num_val_batches > 0:
            avg_val_loss = total_val_loss / num_val_batches
        else:
            avg_val_loss = float('nan')
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        if not torch.isnan(torch.tensor(avg_val_loss)):
            scheduler.step(avg_val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.6f}')
        print(f'Validation Loss: {avg_val_loss:.6f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.8f}')
        
        # Early stopping logic
        if epoch >= min_epochs and not torch.isnan(torch.tensor(avg_val_loss)):
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, 'best_model_checkpoint.pt')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
        elif not torch.isnan(torch.tensor(avg_val_loss)):
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, 'best_model_checkpoint.pt')
    
    # Final evaluation on test set
    model.eval()
    total_test_loss = 0
    num_test_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            test_loss = criterion(out, batch.y)
            if not torch.isnan(test_loss):
                total_test_loss += test_loss.item()
                num_test_batches += 1
    
    if num_test_batches > 0:
        avg_test_loss = total_test_loss / num_test_batches
    else:
        avg_test_loss = float('nan')
    print(f'\nFinal Test Loss: {avg_test_loss:.6f}')
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()
    
    return train_losses, val_losses, avg_test_loss

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load and preprocess data
    train_loader, val_loader, test_loader = load_data()
    
    # Initialize model
    model = DrugGNN(
        num_features=8,  # Number of atom features
        hidden_channels=64,
        num_classes=1
    )
    
    # Train model
    train_losses, val_losses, test_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=100
    )
    
    # Save trained model
    torch.save(model.state_dict(), 'drug_gnn_model.pt')
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == '__main__':
    main() 