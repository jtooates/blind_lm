"""
Phase 1: Data Loader
Loads synthetic sentences from Phase 0 for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Optional, Dict
import json


class SyntheticSentenceDataset(Dataset):
    """Dataset of synthetic sentences for Phase 1 training"""

    def __init__(
        self,
        sentences: List[str],
        tokenizer,
        max_length: int = 64
    ):
        """
        Args:
            sentences: List of sentence strings
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        # Tokenize
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': sentence
        }


class TextFileDataset(Dataset):
    """Dataset that loads sentences from a text file"""

    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: int = 64,
        max_sentences: Optional[int] = None
    ):
        """
        Args:
            file_path: Path to text file with one sentence per line
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            max_sentences: Optional limit on number of sentences to load
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load sentences from file
        with open(file_path, 'r') as f:
            self.sentences = [line.strip() for line in f if line.strip()]

        if max_sentences is not None:
            self.sentences = self.sentences[:max_sentences]

        print(f"Loaded {len(self.sentences)} sentences from {file_path}")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        # Tokenize
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': sentence
        }


class JSONLDataset(Dataset):
    """Dataset that loads from JSONL with augmentations"""

    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: int = 64,
        use_paraphrases: bool = False,
        max_scenes: Optional[int] = None
    ):
        """
        Args:
            file_path: Path to JSONL file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            use_paraphrases: Whether to include paraphrases
            max_scenes: Optional limit on number of scenes to load
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load sentences
        self.sentences = []

        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if max_scenes is not None and i >= max_scenes:
                    break

                data = json.loads(line)

                # Add original
                self.sentences.append(data['original'])

                # Add paraphrases if requested
                if use_paraphrases and 'paraphrases' in data:
                    self.sentences.extend(data['paraphrases'])

        print(f"Loaded {len(self.sentences)} sentences from {file_path}")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        # Tokenize
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': sentence
        }


class FixedEvaluationSet:
    """Fixed set of sentences for consistent evaluation"""

    def __init__(self, sentences: List[str], tokenizer, max_length: int = 64):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-tokenize
        self.encodings = []
        for sentence in sentences:
            encoding = tokenizer(
                sentence,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.encodings.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'text': sentence
            })

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.encodings[idx]

    def get_batch(self, device='cpu'):
        """Get all sentences as a single batch"""
        input_ids = torch.stack([e['input_ids'] for e in self.encodings]).to(device)
        attention_mask = torch.stack([e['attention_mask'] for e in self.encodings]).to(device)
        texts = [e['text'] for e in self.encodings]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text': texts
        }


def create_dataloaders(
    train_file: str,
    val_file: Optional[str] = None,
    tokenizer_name: str = 'gpt2',
    batch_size: int = 256,
    max_length: int = 64,
    num_workers: int = 4,
    file_format: str = 'txt'
) -> Dict[str, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_file: Path to training data file
        val_file: Optional path to validation data file
        tokenizer_name: HuggingFace tokenizer name
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of dataloader workers
        file_format: 'txt' or 'jsonl'

    Returns:
        dict with 'train' and optionally 'val' dataloaders
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
    if file_format == 'txt':
        train_dataset = TextFileDataset(train_file, tokenizer, max_length)
        val_dataset = TextFileDataset(val_file, tokenizer, max_length) if val_file else None
    elif file_format == 'jsonl':
        train_dataset = JSONLDataset(train_file, tokenizer, max_length, use_paraphrases=False)
        val_dataset = JSONLDataset(val_file, tokenizer, max_length, use_paraphrases=False) if val_file else None
    else:
        raise ValueError(f"Unknown file format: {file_format}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    dataloaders = {'train': train_loader}

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        dataloaders['val'] = val_loader

    return dataloaders, tokenizer


def create_fixed_eval_set(tokenizer, num_sentences: int = 16) -> FixedEvaluationSet:
    """
    Create a fixed evaluation set for consistent visualization.

    Args:
        tokenizer: HuggingFace tokenizer
        num_sentences: Number of sentences in evaluation set

    Returns:
        FixedEvaluationSet
    """
    # Fixed sentences spanning different patterns
    fixed_sentences = [
        "the red block is on the blue cube",
        "the green box is next to the yellow block",
        "the blue cube is under the red block",
        "the yellow box is left of the green cube",
        "the purple block is on top of the orange box",
        "the red cube is right of the blue block",
        "the green block is beneath the yellow cube",
        "the orange box is beside the purple block",
        "the blue block is near the green cube",
        "the yellow cube is on the red box",
        "the green box is above the blue block",
        "the red block is touching the yellow cube",
        "the purple cube is far from the orange block",
        "the blue box is next to the green block",
        "the yellow block is under the purple cube",
        "the orange block is on the blue box",
    ]

    # Trim to requested number
    fixed_sentences = fixed_sentences[:num_sentences]

    return FixedEvaluationSet(fixed_sentences, tokenizer)


if __name__ == "__main__":
    # Test the dataloader
    import sys
    sys.path.append('..')

    print("Testing dataloader...")

    # Generate some test data first
    from generate_sentences import generate_sentences
    import tempfile
    import os

    # Generate a small dataset
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_file = f.name

    generate_sentences(
        num_sentences=100,
        complexity=1,
        use_variety=False,
        seed=42,
        output_file=temp_file,
        use_complex=False
    )

    # Create dataloader
    dataloaders, tokenizer = create_dataloaders(
        train_file=temp_file,
        tokenizer_name='gpt2',
        batch_size=8,
        num_workers=0,
        file_format='txt'
    )

    # Test iteration
    print("\nTesting train dataloader:")
    train_loader = dataloaders['train']
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  Example text: {batch['text'][0]}")
        if i >= 2:
            break

    # Test fixed evaluation set
    print("\nTesting fixed evaluation set:")
    eval_set = create_fixed_eval_set(tokenizer, num_sentences=4)
    print(f"Eval set size: {len(eval_set)}")
    batch = eval_set.get_batch()
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Example text: {batch['text'][0]}")

    # Cleanup
    os.unlink(temp_file)
    print("\nDataloader tests passed!")
