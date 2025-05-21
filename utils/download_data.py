import os
import pandas as pd
import requests
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

def download_pubchem_dataset():
    """
    Download a subset of drug-like molecules from PubChem.
    """
    print("Downloading PubChem dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # List of common drug SMILES (diverse set of drugs)
    drug_smiles = [
        # Pain and Inflammation
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CC(C)CC1=CC=C(C=C1)[C@H](C)C(=O)O",  # Ibuprofen
        "CC1=CC=C(C=C1)NC(=O)CC(=O)O",  # Acetaminophen/Paracetamol
        
        # Antibiotics
        "CC1=C(C=C(C=C1)Cl)N=C(C2=CC=CC=C2)OCC(=O)O",  # Clofibrate
        "CC1=C(C(=O)C2=C(O1)C(=O)C=C(C2=O)OC)C",  # Griseofulvin
        "CC1=C(N=C(C(=N1)N)N)CCOCc2ccc(cc2)OC",  # Trimethoprim
        
        # Cardiovascular
        "CCCC1=NN(C2=C1N=C(N=C2N)C)C3=CC=C(C=C3)S(=O)(=O)N",  # Sildenafil
        "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",  # Warfarin
        "CCOC(=O)C1=C(NC(=C(C1C2=CC=CC=C2)C(=O)OC)C)CCCN3CCCCC3",  # Amlodipine
        
        # CNS Drugs
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)(C)NCC(O)C1=CC(O)=CC(O)=C1",  # Salbutamol
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C3=CC=CC=C3",  # Alprazolam
        
        # Hormones
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
        "CC12CCC3C(C1CCC2(O)C(=O)CO)CCC4=CC(=O)CCC34C",  # Cortisone
        "CC12CCC3C(C1CC(C2O)O)CCC4=C3C=CC(=O)C4",  # Estrone
        
        # Anti-inflammatory
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(N)(=O)=O)C(F)(F)F",  # Celecoxib
        "CC1=CC2=C(C=C1)N(C(=O)C3=CC=C(C=C3)Cl)C4=CC=CC=C4C2=O",  # Diclofenac
        "CC1=CC=C(C=C1)C(C)C(=O)O",  # Naproxen
        
        # Antidepressants
        "CNCCC(OC1=CC=C(C=C1)CCC2=CC=CC=C2)C3=CC=CC=C3",  # Fluoxetine
        "CN(C)CCCN1C2=CC=CC=C2CCC3=CC=CC=C31",  # Amitriptyline
        
        # Anti-cancer
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",  # Imatinib
        "CC1=C(C(=CC=C1)Cl)NC2=NC=CC(=N2)C3=CN=CC=C3",  # Gefitinib
        
        # Antivirals
        "CC(C)(C)NCC(O)C1=CC(O)=CC(O)=C1",  # Oseltamivir
        "C1CN(CCN1)C2=NC(=NC3=C2N=CN3C4C(C(C(O4)CO)O)O)N",  # Acyclovir
        
        # Antidiabetics
        "CN(C)C(=N)NC(=N)N",  # Metformin
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(N)(=O)=O)C(F)(F)F"  # Rosiglitazone
    ]
    
    # Process the molecules
    processed_data = []
    for smiles in tqdm(drug_smiles, desc="Processing molecules"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Generate Morgan fingerprint
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                
                # Calculate logP
                logp = Descriptors.MolLogP(mol)
                
                # Calculate QED (Quantitative Estimate of Drug-likeness)
                qed = Descriptors.qed(mol)
                
                processed_data.append({
                    'smiles': smiles,
                    'fingerprint': list(fp.ToBitString()),
                    'logp': logp,
                    'qed': qed
                })
        except:
            continue
    
    # Save processed data
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv('data/pubchem_processed.csv', index=False)
    print(f"Processed {len(processed_data)} molecules successfully!")

if __name__ == "__main__":
    download_pubchem_dataset() 