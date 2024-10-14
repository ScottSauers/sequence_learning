from datetime import datetime

checkpoint_dir = 'checkpoints'
delete_checkpoints = True  # Set to True if you want to delete all checkpoint files

if not os.path.exists(checkpoint_dir):
    print(f"Checkpoint directory '{checkpoint_dir}' does not exist.")
else:
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if checkpoint_files:
        print(f"Checkpoint files in '{checkpoint_dir}':")
        for file in checkpoint_files:
            full_path = os.path.join(checkpoint_dir, file)
            creation_time = os.path.getctime(full_path)
            readable_time = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
            print(f"File: {file}, Full path: {full_path}, Created on: {readable_time}")
        
        if delete_checkpoints:
            for file in checkpoint_files:
                full_path = os.path.join(checkpoint_dir, file)
                os.remove(full_path)
                print(f"Deleted: {full_path}")
            print("All checkpoint files have been deleted.")
    else:
        print(f"No checkpoint files found in '{checkpoint_dir}'.")

import os
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, AdamW
import gzip

# Set random seeds for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Function to load FASTA file correctly by concatenating lines between headers
def load_fasta(file_path):
    print(f"Loading amino acid sequences from: {file_path}")
    sequences = []
    open_func = gzip.open if file_path.endswith('.gz') else open
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
    print(f"Total amino acid sequences loaded: {len(sequences)}")
    # Display sample sequences
    #for i in range(len(sequences)):
        #print(f"Sample Sequence {i}: {sequences[i][:500]}...")
    return sequences

# Function to load and preprocess English text
def load_text(file_path):
    print(f"Loading English text from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Total characters loaded: {len(text)}")
    # Preprocess text: lowercase and keep alphanumerics, period, and space
    allowed_chars = set(string.ascii_lowercase + string.digits + ' .')
    processed_text = ''.join([c.lower() for c in text if c.lower() in allowed_chars])
    print(f"Total characters after preprocessing: {len(processed_text)}")
    print(f"Sample Text: {processed_text[:200]}")
    return processed_text

# Paths to data
fasta_path = '/kaggle/input/uniprotkb-human/uniprotkb_Human_AND_model_organism_9606_2024_10_13.fasta'
text_path = '/kaggle/input/seamus/seamus.txt'

# Load data
aa_sequences = load_fasta(fasta_path)
english_text = load_text(text_path)

import random
import string
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# Define vocabulary size
vocab_size = 28
print(f"Total tokens in tokenizer: {vocab_size}")

# Define characters for English text
english_chars = list(string.ascii_lowercase) + [' ', '.']  # 26 letters + space + period = 28 tokens

# Define characters for amino acids
amino_acids = list('ACDEFGHIKLMNPQRSTVWY') + ['X', 'B', 'U', 'Z']  # 24 amino acids
special_token = '>'  # 1 end-of-sequence token
empty_tokens = ['<PAD1>', '<PAD2>', '<PAD3>']  # 3 empty tokens

# Total AA characters: 24 AAs + '>' + 3 empty tokens = 28 tokens
aa_characters = amino_acids + [special_token] + empty_tokens

print(f"Total characters for AA tokenizer: {len(aa_characters)}")  # 28

assert len(aa_characters) == vocab_size, "AA characters + special tokens do not match vocab_size"

# Assign tokens randomly to AA characters
aa_token_indices = list(range(vocab_size))
random.shuffle(aa_token_indices)
aa_token_mapping = {char: idx for char, idx in zip(aa_characters, aa_token_indices)}

print("\nAA Token Mapping:")
for char, token in aa_token_mapping.items():
    print(f"'{char}': {token}")

# No need to assert that '>' and '.' have unique tokens; overlapping is allowed

# Define characters for English tokenizer
english_token_indices = list(range(vocab_size))
random.shuffle(english_token_indices)
english_token_mapping = {char: idx for char, idx in zip(english_chars, english_token_indices)}

print("\nEnglish Token Mapping:")
for char, token in english_token_mapping.items():
    print(f"'{char}': {token}")

# Function to tokenize amino acid sequences with length filtering
def tokenize_aa_sequences(sequences, mapping, block_size=128, subset_fraction=0.1):
    print("\nTokenizing amino acid sequences...")
    tokenized_sequences = []
    skipped_sequences = 0
    
    # Filter sequences to have len(seq) >= block_size +1
    filtered_sequences = [seq for seq in sequences if len(seq) >= block_size +1]
    print(f"Total sequences with len >= {block_size +1}: {len(filtered_sequences)}")
    
    if len(filtered_sequences) == 0:
        print("No sequences meet the length requirement. Exiting tokenization.")
        return tokenized_sequences
    
    # Select a random subset (e.g., 10%) for efficiency
    subset_size = max(1, int(len(filtered_sequences) * subset_fraction))
    subset_indices = random.sample(range(len(filtered_sequences)), subset_size)
    subset_sequences = [filtered_sequences[i] for i in subset_indices]
    print(f"Selected {subset_size} sequences for tokenization.")
    
    for seq in tqdm(subset_sequences, desc="Tokenizing AA Sequences"):
        tokens = []
        for char in seq:
            token = mapping.get(char, -1)
            tokens.append(token)
        if -1 in tokens:
            skipped_sequences += 1
            continue  # Skip sequences with unmapped characters
        tokenized_sequences.append(tokens)
    
    print(f"Total tokenized amino acid sequences: {len(tokenized_sequences)}")
    print(f"Total sequences skipped due to unmapped characters: {skipped_sequences}")
    
    # Display sample tokenized sequences
    for i in range(min(3, len(tokenized_sequences))):
        print(f"Tokenized Sequence {i+1}: {tokenized_sequences[i][:50]}...")
    
    return tokenized_sequences

# Function to tokenize English text
def tokenize_text(text, mapping):
    print("\nTokenizing English text...")
    tokens = []
    skipped_chars = 0
    for char in text:
        token = mapping.get(char, -1)
        if token != -1:
            tokens.append(token)
        else:
            skipped_chars += 1  # Optionally, handle unknown tokens here
    print(f"Total tokens in English text: {len(tokens)}")
    print(f"Total characters skipped in English text: {skipped_chars}")
    print(f"Sample Tokenized Text: {tokens[:50]}")
    return tokens

# Define block size
block_size = 32

# Tokenize amino acid sequences (using subset with length filtering)
tokenized_real_aa = tokenize_aa_sequences(aa_sequences, aa_token_mapping, block_size=block_size, subset_fraction=0.1)

# Check if tokenized_real_aa is empty
if not tokenized_real_aa:
    print("Error: No tokenized amino acid sequences were created. Please check the sequence lengths and token mapping.")
else:
    # Create fake (shuffled) amino acid sequences using the same subset
    print("\nCreating fake (shuffled) amino acid sequences...")
    fake_aa_sequences = []
    for seq in tokenized_real_aa:
        # Convert tokens back to characters for shuffling
        tokens_to_chars = {v: k for k, v in aa_token_mapping.items()}
        char_seq = ''.join([tokens_to_chars.get(token, 'A') for token in seq])  # Default to 'A' if token not found
        shuffled_seq = ''.join(random.sample(char_seq, len(char_seq)))
        fake_aa_sequences.append(shuffled_seq)
    print(f"Total fake amino acid sequences created: {len(fake_aa_sequences)}")
    
    # Tokenize fake amino acid sequences (using subset with length filtering)
    tokenized_fake_aa = tokenize_aa_sequences(fake_aa_sequences, aa_token_mapping, block_size=block_size, subset_fraction=1.0)  # Process all fake sequences

# Tokenize English text
tokenized_text = tokenize_text(english_text, english_token_mapping)

# Define PyTorch Dataset
class SequenceDataset(Dataset):
    def __init__(self, sequences, block_size=128):
        print(f"\nInitializing SequenceDataset with block size {block_size}...")
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
        print(f"Total samples in dataset: {len(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Create PyTorch Dataset for English text
class TextDataset(Dataset):
    def __init__(self, tokens, block_size=128):
        print(f"\nInitializing TextDataset with block size {block_size}...")
        self.inputs = []
        self.labels = []
        seq = tokens
        if len(seq) < block_size +1:
            print("Warning: English text sequence is shorter than block_size +1. No samples created.")
            return
        for i in range(len(seq) - block_size):
            input_seq = seq[i:i+block_size]
            label_seq = seq[i+1:i+block_size+1]
            self.inputs.append(input_seq)
            self.labels.append(label_seq)
        print(f"Total samples in TextDataset: {len(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Create datasets
print("\nCreating datasets for real amino acids, fake amino acids, and English text...")
if tokenized_real_aa:
    dataset_real_aa = SequenceDataset(tokenized_real_aa, block_size=block_size)
else:
    dataset_real_aa = None

if 'fake_aa_sequences' in locals() and fake_aa_sequences:
    dataset_fake_aa = SequenceDataset(tokenized_fake_aa, block_size=block_size)
else:
    dataset_fake_aa = None

dataset_text = TextDataset(tokenized_text, block_size=block_size)  # Single large sequence

# Verify dataset sizes
if dataset_real_aa:
    print(f"\nReal AA Dataset Size: {len(dataset_real_aa)}")
else:
    print(f"\nReal AA Dataset Size: 0")

if dataset_fake_aa:
    print(f"Fake AA Dataset Size: {len(dataset_fake_aa)}")
else:
    print(f"Fake AA Dataset Size: 0")

print(f"English Text Dataset Size: {len(dataset_text)}")

# Create DataLoaders
batch_size = 64
if dataset_real_aa and len(dataset_real_aa) > 0:
    loader_real_aa = DataLoader(dataset_real_aa, batch_size=batch_size, shuffle=True)
else:
    print("Warning: Real AA Dataset is empty. DataLoader not created.")
if dataset_fake_aa and len(dataset_fake_aa) > 0:
    loader_fake_aa = DataLoader(dataset_fake_aa, batch_size=batch_size, shuffle=True)
else:
    print("Warning: Fake AA Dataset is empty. DataLoader not created.")
if len(dataset_text) > 0:
    loader_text = DataLoader(dataset_text, batch_size=batch_size, shuffle=True)
else:
    print("Warning: English Text Dataset is empty. DataLoader not created.")

print("\nDataLoaders created.")


from IPython.display import display, clear_output
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, AdamW
from tqdm import tqdm
import numpy as np

%matplotlib inline

# Define GPT-like Transformer Model
print("Defining GPT-like Transformer model...")

config = GPT2Config(
    vocab_size=vocab_size,
    n_embd=256,
    n_layer=4,
    n_head=8,
    dropout=0.1,
    bos_token_id=0,
    eos_token_id=1
)

model = GPT2LMHeadModel(config).to(device)
print("Model initialized.")

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# Flag to resume from checkpoint
resume_from_checkpoint = True
checkpoint_dir = 'checkpoints'

# Function to load the most recent checkpoint
def load_checkpoint(model, optimizer, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory '{checkpoint_dir}' does not exist.")
        return 0, 1, 0  # Return starting values if no checkpoints

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if checkpoint_files:
        latest_checkpoint = max(
            checkpoint_files, 
            key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x))
        )
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            trial = checkpoint['trial']
            batch = checkpoint['batch']
            print(f"Resumed from checkpoint: {checkpoint_path}")
            return epoch, trial, batch
        else:
            print(f"Checkpoint file '{latest_checkpoint}' does not exist.")
    else:
        print("No checkpoint found. Starting from scratch.")
    
    return 0, 1, 0

# Optionally resume from the most recent checkpoint
start_epoch, start_trial, start_batch = load_checkpoint(model, optimizer, checkpoint_dir)

# Define training function with real-time plotting
def train_model(model, dataloader, optimizer, epoch, trial, dataset_name, checkpoint_flag, checkpoint_dir='checkpoints', max_batches=10000):
    model.train()
    total_loss = 0
    batch_losses = []
    
    for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc=f"Trial {trial} | Epoch {epoch} | {dataset_name}")):
        if batch_idx + 1 > max_batches:
            print(f"Reached the limit of {max_batches} batches. Stopping.")
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
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f'trial_{trial}_epoch_{epoch}_batch_{batch_idx+1}.pth'
            )
            torch.save({
                'trial': trial,
                'epoch': epoch,
                'batch': batch_idx+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
            
            # Plot and display loss curves in real-time
            plot_loss(trial, epoch, batch_losses, dataset_name)
    
    avg_loss = total_loss / len(dataloader)
    print(f"Trial {trial} | Epoch {epoch} | {dataset_name} | Average Loss: {avg_loss}")
    return avg_loss, batch_losses

def plot_loss(trial, epoch, losses, dataset_name, window_size=50):
    clear_output(wait=True)  # Clear the previous output to refresh the plot

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
    
    # Display the updated figure in real-time in Kaggle
    display(plt.gcf())
    plt.close()

# Training parameters
num_trials = 4
num_epochs = 1 
checkpoint_flag = True

# Initialize lists to store all losses
all_losses_real = []
all_losses_fake = []
all_losses_text = []

# Create directory for saving loss data
os.makedirs('loss_data', exist_ok=True)

# Start training trials
for trial in range(start_trial, num_trials + 1):
    print(f"\n=== Starting Trial {trial} ===")
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n--- Trial {trial} | Epoch {epoch} ---")
        
        # Train on real amino acid sequences
        print("Training on real amino acid sequences...")
        if loader_real_aa:
            avg_loss_real, losses_real = train_model(
                model, loader_real_aa, optimizer, epoch, trial, 
                'Real AA', checkpoint_flag
            )
            all_losses_real.extend(losses_real)
        else:
            print("Real AA DataLoader is not available.")
        
        # Train on fake amino acid sequences
        print("Training on fake amino acid sequences...")
        if loader_fake_aa:
            avg_loss_fake, losses_fake = train_model(
                model, loader_fake_aa, optimizer, epoch, trial, 
                'Fake AA', checkpoint_flag
            )
            all_losses_fake.extend(losses_fake)
        else:
            print("Fake AA DataLoader is not available.")
        
        # Post-train on English text
        print("Post-training on English text...")
        if loader_text:
            avg_loss_text, losses_text = train_model(
                model, loader_text, optimizer, epoch, trial, 
                'English Text', checkpoint_flag
            )
            all_losses_text.extend(losses_text)
        else:
            print("English Text DataLoader is not available.")
        
        print("=== Epoch Completed ===")
    
    print(f"=== Trial {trial} Completed ===")

print("\n=== All Training Trials Completed ===")

# Save the final model
final_model_path = 'final_gpt_model.pth' 
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved at {final_model_path}")
