"""
Test that reconstruction display works during evaluation
"""

import torch
import sys
from pathlib import Path

print("="*70)
print("Testing Reconstruction Display")
print("="*70)

from model import create_model
from decoder import create_decoder
from losses import ImagePriorLoss
from dataloader import create_fixed_eval_set
from transformers import AutoTokenizer

# Create encoder and decoder
encoder_config = {
    'vocab_size': 50257,
    'max_seq_len': 64,
    'hidden_size': 384,
    'num_layers': 6,
    'num_heads': 8,
    'ffn_size': 1536,
    'dropout': 0.1,
    'grid_size': 32,
    'num_channels': 6,
    'use_rope': True,
    'use_smooth_head': True
}

decoder_config = {
    'vocab_size': 50257,
    'max_seq_len': 64,
    'hidden_size': 384,
    'num_layers': 4,
    'num_heads': 8,
    'ffn_size': 1536,
    'dropout': 0.1,
    'use_rope': True
}

encoder = create_model(encoder_config)
decoder = create_decoder(decoder_config)

# Create tokenizer and eval set
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
eval_set = create_fixed_eval_set(tokenizer, num_sentences=5)

print(f"\n✓ Created encoder, decoder, and eval set with {len(eval_set)} sentences")

# Get batch
eval_batch = eval_set.get_batch()

print("\n" + "="*70)
print("Original Sentences:")
print("="*70)
for i, text in enumerate(eval_batch['text']):
    print(f"{i+1}. {text}")

# Forward pass
with torch.no_grad():
    latents = encoder(eval_batch['input_ids'], eval_batch['attention_mask'])
    logits = decoder(latents, eval_batch['input_ids'], eval_batch['attention_mask'])

# Decode predictions
predicted_ids = logits.argmax(dim=-1)
predicted_texts = []
for i in range(predicted_ids.shape[0]):
    pred_text = tokenizer.decode(predicted_ids[i], skip_special_tokens=True)
    predicted_texts.append(pred_text)

print("\n" + "="*70)
print("Predicted Sentences (Untrained Model):")
print("="*70)
for i, text in enumerate(predicted_texts):
    print(f"{i+1}. {text}")

# Check if they match (they won't for untrained model)
print("\n" + "="*70)
print("Comparison:")
print("="*70)
exact_matches = 0
for i, (orig, pred) in enumerate(zip(eval_batch['text'], predicted_texts)):
    match = "✓" if orig == pred else "✗"
    if orig == pred:
        exact_matches += 1
    print(f"{i+1}. {match}")
    print(f"   Original:  {orig}")
    print(f"   Predicted: {pred}")

accuracy = exact_matches / len(eval_batch['text']) * 100
print(f"\nExact match accuracy: {exact_matches}/{len(eval_batch['text'])} ({accuracy:.1f}%)")

print("\n" + "="*70)
print("✓ Reconstruction display test complete!")
print("="*70)
print("\nNote: Untrained model produces gibberish (expected).")
print("After training with reconstruction loss, accuracy should increase.")
print("="*70)
