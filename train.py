import os
import sys
import random
import string
import shutil
import gzip
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import GPT2Config, GPT2LMHeadModel, AdamW

import requests

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log")
    ]
)

# Constants and Configuration
DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
LOSS_DATA_DIR = Path("loss_data")
PLOTS_DIR = Path("plots")
FINAL_MODEL_PATH = Path("final_gpt_model.pth")

UNIPROTKB_URL = "https://www.kaggle.com/api/v1/datasets/download/sauers/uniprotkb-human"
SEAMUS_URL = "https://www.kaggle.com/api/v1/datasets/download/sauers/seamus"

FASTA_FILENAME = "uniprotkb_Human_AND_model_organism_9606_2024_10_13.fasta.zip"
TEXT_FILENAME = "seamus.txt.zip"

RANDOM_SEED = 42
VOCAB_SIZE = 28
BATCH_SIZE = 64
BLOCK_SIZE = 32
NUM_TRIALS = 4
NUM_EPOCHS = 1
LEARNING_RATE = 5e-5
CHECKPOINT_FLAG = True
MAX_BATCHES = 10000

for directory in [DATA_DIR, CHECKPOINT_DIR, LOSS_DATA_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def download_file(url, dest_path):
    """
    Downloads a file from the specified URL to the destination path.
    """
    if dest_path.exists():
        logging.info(f"File '{dest_path}' already exists. Skipping download.")
        return
    logging.info(f"Downloading '{url}' to '{dest_path}'...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        logging.info(f"Downloaded '{dest_path}'.")
    except Exception as e:
        logging.error(f"Failed to download '{url}': {e}")
        sys.exit(1)

def download_datasets():
    """
    Download the specified datasets if they are not already present in the data directory.
    """
    datasets = {
        UNIPROTKB_URL: FASTA_FILENAME,
        SEAMUS_URL: TEXT_FILENAME
    }
    
    for url, filename in datasets.items():
        file_path = DATA_DIR / filename
        download_file(url, file_path)

def manage_checkpoints(delete_checkpoints=True):
    """
    Manage the checkpoints directory by listing and optionally deleting checkpoint files.
    """
    if not CHECKPOINT_DIR.exists():
        logging.info(f"Checkpoint directory '{CHECKPOINT_DIR}' does not exist.")
        return
    
    checkpoint_files = list(CHECKPOINT_DIR.glob("*.pth"))
    
    if checkpoint_files:
        logging.info(f"Checkpoint files in '{CHECKPOINT_DIR}':")
        for file in checkpoint_files:
            creation_time = datetime.fromtimestamp(file.stat().st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f"File: {file.name}, Full path: {file.resolve()}, Created on: {creation_time}")
        
        if delete_checkpoints:
            for file in checkpoint_files:
                try:
                    file.unlink()
                    logging.info(f"Deleted: {file}")
                except Exception as e:
                    logging.error(f"Failed to delete '{file}': {e}")
            logging.info("All checkpoint files have been deleted.")
    else:
        logging.info(f"No checkpoint files found in '{CHECKPOINT_DIR}'.")

def set_random_seeds(seed=RANDOM_SEED):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seeds set to {seed}.")

def get_device():
    """
    Check for GPU availability and return the appropriate device.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    return device

def load_fasta(file_path):
    """
    Load amino acid sequences from a FASTA file by concatenating lines between headers.
    """
    logging.info(f"Loading amino acid sequences from: {file_path}")
    sequences = []
    open_func = gzip.open if file_path.suffix.endswith('.gz') else open
    try:
        with open_func(file_path, 'rt') as f:
            current_seq = []
            for line in f:
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                        current_seq = []
                    continue  # Skip header lines
                seq_line = line.strip().replace('\n', '').replace('\r', '')
                if seq_line:
                    current_seq.append(seq_line)
            if current_seq:
                sequences.append(''.join(current_seq))
    except Exception as e:
        logging.error(f"Error loading FASTA file '{file_path}': {e}")
        sys.exit(1)
    
    logging.info(f"Total amino acid sequences loaded: {len(sequences)}")
    return sequences

def load_text(file_path):
    """
    Load and preprocess English text from a file.
    """
    logging.info(f"Loading English text from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        logging.error(f"Error loading text file '{file_path}': {e}")
        sys.exit(1)
    
    logging.info(f"Total characters loaded: {len(text)}")
    # Preprocess text: lowercase and keep alphanumerics, period, and space
    allowed_chars = set(string.ascii_lowercase + string.digits + ' .')
    processed_text = ''.join([c.lower() for c in text if c.lower() in allowed_chars])
    logging.info(f"Total characters after preprocessing: {len(processed_text)}")
    logging.info(f"Sample Text: {processed_text[:200]}")
    return processed_text

def create_token_mappings():
    """
    Create random token mappings for amino acids and English characters.
    """
    # Define characters for English text
    english_chars = list(string.ascii_lowercase) + [' ', '.']  # 26 letters + space + period = 28 tokens
    
    # Define characters for amino acids
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY') + ['X', 'B', 'U', 'Z']  # 24 amino acids
    special_token = '>'  # 1 end-of-sequence token
    empty_tokens = ['<PAD1>', '<PAD2>', '<PAD3>']  # 3 empty tokens
    
    # Total AA characters: 24 AAs + '>' + 3 empty tokens = 28 tokens
    aa_characters = amino_acids + [special_token] + empty_tokens
    
    logging.info(f"Total characters for AA tokenizer: {len(aa_characters)}")  # 28
    
    if len(aa_characters) != VOCAB_SIZE:
        logging.error("AA characters + special tokens do not match VOCAB_SIZE")
        sys.exit(1)
    
    # Assign tokens randomly to AA characters
    aa_token_indices = list(range(VOCAB_SIZE))
    random.shuffle(aa_token_indices)
    aa_token_mapping = {char: idx for char, idx in zip(aa_characters, aa_token_indices)}
    
    logging.info("\nAA Token Mapping:")
    for char, token in aa_token_mapping.items():
        logging.info(f"'{char}': {token}")
    
    # Define characters for English tokenizer
    english_token_indices = list(range(VOCAB_SIZE))
    random.shuffle(english_token_indices)
    english_token_mapping = {char: idx for char, idx in zip(english_chars, english_token_indices)}
    
    logging.info("\nEnglish Token Mapping:")
    for char, token in english_token_mapping.items():
        logging.info(f"'{char}': {token}")
    
    return aa_token_mapping, english_token_mapping, aa_characters, english_chars

def tokenize_aa_sequences(sequences, mapping, block_size=128, subset_fraction=0.1):
    """
    Tokenize amino acid sequences with length filtering and optional subset selection.
    """
    logging.info("\nTokenizing amino acid sequences...")
    tokenized_sequences = []
    skipped_sequences = 0
    
    # Filter sequences to have len(seq) >= block_size +1
    filtered_sequences = [seq for seq in sequences if len(seq) >= block_size +1]
    logging.info(f"Total sequences with len >= {block_size +1}: {len(filtered_sequences)}")
    
    if not filtered_sequences:
        logging.warning("No sequences meet the length requirement. Exiting tokenization.")
        return tokenized_sequences
    
    # Select a random subset (e.g., 10%) for efficiency
    subset_size = max(1, int(len(filtered_sequences) * subset_fraction))
    subset_indices = random.sample(range(len(filtered_sequences)), subset_size)
    subset_sequences = [filtered_sequences[i] for i in subset_indices]
    logging.info(f"Selected {subset_size} sequences for tokenization.")
    
    for seq in tqdm(subset_sequences, desc="Tokenizing AA Sequences"):
        tokens = []
        for char in seq:
            token = mapping.get(char, -1)
            tokens.append(token)
        if -1 in tokens:
            skipped_sequences += 1
            continue  # Skip sequences with unmapped characters
        tokenized_sequences.append(tokens)
    
    logging.info(f"Total tokenized amino acid sequences: {len(tokenized_sequences)}")
    logging.info(f"Total sequences skipped due to unmapped characters: {skipped_sequences}")
    
    # Display sample tokenized sequences
    for i in range(min(3, len(tokenized_sequences))):
        logging.info(f"Tokenized Sequence {i+1}: {tokenized_sequences[i][:50]}...")
    
    return tokenized_sequences

def create_fake_aa_sequences(tokenized_real_aa, aa_token_mapping):
    """
    Create fake (shuffled) amino acid sequences using the tokenized real sequences.
    """
    logging.info("\nCreating fake (shuffled) amino acid sequences...")
    fake_aa_sequences = []
    tokens_to_chars = {v: k for k, v in aa_token_mapping.items()}
    
    for tokens in tokenized_real_aa:
        # Convert tokens back to characters for shuffling
        char_seq = ''.join([tokens_to_chars.get(token, 'A') for token in tokens])  # Default to 'A' if token not found
        shuffled_seq = ''.join(random.sample(char_seq, len(char_seq)))
        fake_aa_sequences.append(shuffled_seq)
    
    logging.info(f"Total fake amino acid sequences created: {len(fake_aa_sequences)}")
    return fake_aa_sequences

def tokenize_text(text, mapping):
    """
    Tokenize English text.
    """
    logging.info("\nTokenizing English text...")
    tokens = []
    skipped_chars = 0
    for char in text:
        token = mapping.get(char, -1)
        if token != -1:
            tokens.append(token)
        else:
            skipped_chars += 1  # Optionally, handle unknown tokens here
    logging.info(f"Total tokens in English text: {len(tokens)}")
    logging.info(f"Total characters skipped in English text: {skipped_chars}")
    logging.info(f"Sample Tokenized Text: {tokens[:50]}")
    return tokens

class SequenceDataset(Dataset):
    """
    PyTorch Dataset for amino acid sequences.
    """
    def __init__(self, sequences, block_size=128):
        logging.info(f"\nInitializing SequenceDataset with block size {block_size}...")
        self.inputs = []
        self.labels = []
        for seq in sequences:
            if len(seq) < block_size +1:
                continue  # Skip short sequences
            for i in range(len(seq) - block_size):
                input_seq = seq[i:i+block_size]
                label_seq = seq[i+1:i+block_size+1]
                self.inputs.append(input_seq)
                self.labels.append(label_seq)
        logging.info(f"Total samples in SequenceDataset: {len(self.inputs)}")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class TextDataset(Dataset):
    """
    PyTorch Dataset for English text.
    """
    def __init__(self, tokens, block_size=128):
        logging.info(f"\nInitializing TextDataset with block size {block_size}...")
        self.inputs = []
        self.labels = []
        seq = tokens
        if len(seq) < block_size +1:
            logging.warning("English text sequence is shorter than block_size +1. No samples created.")
            return
        for i in range(len(seq) - block_size):
            input_seq = seq[i:i+block_size]
            label_seq = seq[i+1:i+block_size+1]
            self.inputs.append(input_seq)
            self.labels.append(label_seq)
        logging.info(f"Total samples in TextDataset: {len(self.inputs)}")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def load_checkpoint(model, optimizer, checkpoint_dir):
    """
    Load the most recent checkpoint from the checkpoint directory.
    """
    if not checkpoint_dir.exists():
        logging.info(f"Checkpoint directory '{checkpoint_dir}' does not exist.")
        return 0, 1, 0  # Return starting values if no checkpoints

    checkpoint_files = list(checkpoint_dir.glob("*.pth"))
    if checkpoint_files:
        latest_checkpoint = max(
            checkpoint_files, 
            key=lambda x: x.stat().st_ctime
        )
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            trial = checkpoint['trial']
            batch = checkpoint['batch']
            logging.info(f"Resumed from checkpoint: {latest_checkpoint}")
            return epoch, trial, batch
        except Exception as e:
            logging.error(f"Failed to load checkpoint '{latest_checkpoint}': {e}")
    else:
        logging.info("No checkpoint found. Starting from scratch.")
    
    return 0, 1, 0

def plot_loss(trial, epoch, losses, dataset_name, window_size=50):
    """
    Plot and save the loss curve.
    """
    # Apply a moving average to smooth the loss curve
    smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')

    # Create a new figure for plotting
    plt.figure(figsize=(10,6))
    plt.plot(smoothed_losses, label=f'{dataset_name} Loss (Smoothed)')
    plt.title(f'Trial {trial} | Epoch {epoch} | {dataset_name} | Smoothed Loss Curve')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the plots directory
    plot_filename = PLOTS_DIR / f'trial_{trial}_epoch_{epoch}_{dataset_name}_loss.png'
    plt.savefig(plot_filename)
    plt.close()
    logging.info(f"Loss plot saved to {plot_filename}")

def train_model(model, dataloader, optimizer, epoch, trial, dataset_name, checkpoint_flag, checkpoint_dir=CHECKPOINT_DIR, max_batches=MAX_BATCHES):
    """
    Train the model on the given dataloader.
    """
    model.train()
    total_loss = 0
    batch_losses = []
    
    for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc=f"Trial {trial} | Epoch {epoch} | {dataset_name}")):
        if batch_idx + 1 > max_batches:
            logging.info(f"Reached the limit of {max_batches} batches. Stopping.")
            break

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        batch_losses.append(loss.item())
        
        if checkpoint_flag and (batch_idx + 1) % 100 == 0:
            checkpoint_path = checkpoint_dir / f'trial_{trial}_epoch_{epoch}_batch_{batch_idx+1}.pth'
            torch.save({
                'trial': trial,
                'epoch': epoch,
                'batch': batch_idx+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            logging.info(f"Checkpoint saved at {checkpoint_path}")
            
            # Plot and save loss curves
            plot_loss(trial, epoch, batch_losses, dataset_name)
    
    avg_loss = total_loss / len(dataloader)
    logging.info(f"Trial {trial} | Epoch {epoch} | {dataset_name} | Average Loss: {avg_loss}")
    return avg_loss, batch_losses

def main():
    # Download datasets
    download_datasets()
    
    # Manage checkpoints
    manage_checkpoints(delete_checkpoints=True)
    
    # Set random seeds
    set_random_seeds(seed=RANDOM_SEED)
    
    # Check device
    global device
    device = get_device()
    
    # Load data
    fasta_path = DATA_DIR / FASTA_FILENAME
    text_path = DATA_DIR / TEXT_FILENAME
    
    aa_sequences = load_fasta(fasta_path)
    english_text = load_text(text_path)
    
    # Create token mappings
    aa_token_mapping, english_token_mapping, aa_characters, english_chars = create_token_mappings()
    
    # Tokenize amino acid sequences
    tokenized_real_aa = tokenize_aa_sequences(aa_sequences, aa_token_mapping, block_size=BLOCK_SIZE, subset_fraction=0.1)
    
    # Create and tokenize fake amino acid sequences
    if tokenized_real_aa:
        fake_aa_sequences = create_fake_aa_sequences(tokenized_real_aa, aa_token_mapping)
        tokenized_fake_aa = tokenize_aa_sequences(fake_aa_sequences, aa_token_mapping, block_size=BLOCK_SIZE, subset_fraction=1.0)
    else:
        logging.warning("No real amino acid sequences were tokenized. Skipping fake sequences creation.")
        tokenized_fake_aa = []
    
    # Tokenize English text
    tokenized_text = tokenize_text(english_text, english_token_mapping)
    
    # Define PyTorch Datasets
    logging.info("\nCreating datasets for real amino acids, fake amino acids, and English text...")
    dataset_real_aa = SequenceDataset(tokenized_real_aa, block_size=BLOCK_SIZE) if tokenized_real_aa else None
    dataset_fake_aa = SequenceDataset(tokenized_fake_aa, block_size=BLOCK_SIZE) if tokenized_fake_aa else None
    dataset_text = TextDataset(tokenized_text, block_size=BLOCK_SIZE) if tokenized_text else None
    
    # Verify dataset sizes
    if dataset_real_aa:
        logging.info(f"\nReal AA Dataset Size: {len(dataset_real_aa)}")
    else:
        logging.info("\nReal AA Dataset Size: 0")
    
    if dataset_fake_aa:
        logging.info(f"Fake AA Dataset Size: {len(dataset_fake_aa)}")
    else:
        logging.info("Fake AA Dataset Size: 0")
    
    if dataset_text:
        logging.info(f"English Text Dataset Size: {len(dataset_text)}")
    else:
        logging.info("English Text Dataset Size: 0")
    
    # Create DataLoaders
    if dataset_real_aa and len(dataset_real_aa) > 0:
        loader_real_aa = DataLoader(dataset_real_aa, batch_size=BATCH_SIZE, shuffle=True)
    else:
        logging.warning("Real AA Dataset is empty. DataLoader not created.")
        loader_real_aa = None
    
    if dataset_fake_aa and len(dataset_fake_aa) > 0:
        loader_fake_aa = DataLoader(dataset_fake_aa, batch_size=BATCH_SIZE, shuffle=True)
    else:
        logging.warning("Fake AA Dataset is empty. DataLoader not created.")
        loader_fake_aa = None
    
    if dataset_text and len(dataset_text) > 0:
        loader_text = DataLoader(dataset_text, batch_size=BATCH_SIZE, shuffle=True)
    else:
        logging.warning("English Text Dataset is empty. DataLoader not created.")
        loader_text = None
    
    logging.info("\nDataLoaders created.")
    
    # Define GPT-like Transformer Model
    logging.info("Defining GPT-like Transformer model...")
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_embd=256,
        n_layer=4,
        n_head=8,
        dropout=0.1,
        bos_token_id=0,
        eos_token_id=1
    )
    
    model = GPT2LMHeadModel(config).to(device)
    logging.info("Model initialized.")
    
    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Optionally resume from checkpoint
    start_epoch, start_trial, start_batch = load_checkpoint(model, optimizer, CHECKPOINT_DIR)
    
    # Initialize lists to store all losses
    all_losses_real = []
    all_losses_fake = []
    all_losses_text = []
    
    # Start training trials
    for trial in range(start_trial, NUM_TRIALS + 1):
        logging.info(f"\n=== Starting Trial {trial} ===")
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            logging.info(f"\n--- Trial {trial} | Epoch {epoch} ---")
            
            # Train on real amino acid sequences
            logging.info("Training on real amino acid sequences...")
            if loader_real_aa:
                avg_loss_real, losses_real = train_model(
                    model, loader_real_aa, optimizer, epoch, trial, 
                    'Real_AA', CHECKPOINT_FLAG
                )
                all_losses_real.extend(losses_real)
            else:
                logging.warning("Real AA DataLoader is not available.")
            
            # Train on fake amino acid sequences
            logging.info("Training on fake amino acid sequences...")
            if loader_fake_aa:
                avg_loss_fake, losses_fake = train_model(
                    model, loader_fake_aa, optimizer, epoch, trial, 
                    'Fake_AA', CHECKPOINT_FLAG
                )
                all_losses_fake.extend(losses_fake)
            else:
                logging.warning("Fake AA DataLoader is not available.")
            
            # Post-train on English text
            logging.info("Post-training on English text...")
            if loader_text:
                avg_loss_text, losses_text = train_model(
                    model, loader_text, optimizer, epoch, trial, 
                    'English_Text', CHECKPOINT_FLAG
                )
                all_losses_text.extend(losses_text)
            else:
                logging.warning("English Text DataLoader is not available.")
            
            logging.info("=== Epoch Completed ===")
        
        logging.info(f"=== Trial {trial} Completed ===")
    
    logging.info("\n=== All Training Trials Completed ===")
    
    # Save the final model
    try:
        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        logging.info(f"Final model saved at '{FINAL_MODEL_PATH}'.")
    except Exception as e:
        logging.error(f"Failed to save the final model: {e}")

if __name__ == "__main__":
    main()
