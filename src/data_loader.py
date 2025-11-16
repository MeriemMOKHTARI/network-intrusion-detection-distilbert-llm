import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

def load_and_preprocess_data(train_path, test_path, sample_frac=0.05):
    """Charge et prétraite les données"""
    train_df = pd.read_parquet(train_path).sample(frac=sample_frac, random_state=42)
    test_df = pd.read_parquet(test_path).sample(frac=sample_frac, random_state=42)
    
    # Votre code de preprocessing ici
    cols_for_text = ['proto','service','state','dur']
    def make_text(row):
        return " | ".join([f"{c}={row[c]}" for c in cols_for_text if c in row])
    
    train_df["log_text"] = train_df.apply(make_text, axis=1)
    test_df["log_text"] = test_df.apply(make_text, axis=1)
    
    train_df["y"] = train_df["attack_cat"].apply(lambda x: 0 if str(x).lower() in ["normal","benign"] else 1)
    test_df["y"] = test_df["attack_cat"].apply(lambda x: 0 if str(x).lower() in ["normal","benign"] else 1)
    
    return train_df, test_df

def create_hf_datasets(train_df, test_df):
    """Crée les datasets Hugging Face"""
    combined = pd.concat([train_df[['log_text','y']], test_df[['log_text','y']]], ignore_index=True)
    train_val, test_split = train_test_split(combined, test_size=0.2, stratify=combined["y"], random_state=42)
    train_split, val_split = train_test_split(train_val, test_size=0.15, stratify=train_val["y"], random_state=42)
    
    hf = DatasetDict({
        "train": Dataset.from_pandas(train_split.reset_index(drop=True)).rename_column("y", "label"),
        "validation": Dataset.from_pandas(val_split.reset_index(drop=True)).rename_column("y", "label"),
        "test": Dataset.from_pandas(test_split.reset_index(drop=True)).rename_column("y", "label"),
    })
    
    return hf