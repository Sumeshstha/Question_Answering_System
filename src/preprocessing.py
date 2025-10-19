"""
SQuAD dataset preprocessing script.
This script downloads and preprocesses the SQuAD dataset for question answering.
"""
import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

def preprocess_squad(model_name="bert-large-uncased", save_path="data/processed", 
                     max_length=384, stride=128, squad_version="v1.1"):
    """
    Preprocess SQuAD dataset for question answering.
    
    Args:
        model_name: Pretrained model name for tokenizer
        save_path: Path to save processed dataset
        max_length: Maximum sequence length
        stride: Stride for sliding window
        squad_version: SQuAD version ("v1.1" or "v2.0")
    """
    print(f"Loading SQuAD {squad_version} dataset...")
    
    # Load dataset from Hugging Face or local files
    try:
        if squad_version == "v1.1":
            dataset = load_dataset("squad")
        else:
            dataset = load_dataset("squad_v2")
        print("Dataset loaded from Hugging Face")
    except Exception as e:
        print(f"Error loading from Hugging Face: {e}")
        print("Attempting to load from local files...")
        # Implement local loading if needed
        raise NotImplementedError("Local loading not implemented yet")
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Preprocessing function for dataset mapping
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        contexts = [c.strip() for c in examples["context"]]
        
        # Tokenize inputs
        inputs = tokenizer(
            questions,
            contexts,
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        # Map original indices to new features
        sample_map = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs.pop("offset_mapping")
        
        # Initialize answer positions
        inputs["start_positions"] = []
        inputs["end_positions"] = []
        
        for i, offsets in enumerate(offset_mapping):
            # Get index of the original example this feature came from
            sample_idx = sample_map[i]
            
            # Get answer for this example
            answer = examples['answers'][sample_idx]
            
            # If no answers, set positions to CLS token (0)
            if len(answer["answer_start"]) == 0:
                inputs["start_positions"].append(0)
                inputs["end_positions"].append(0)
                continue
                
            # Get start character of answer in context
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            
            # Find token indices that contain the answer
            token_start_index = 0
            token_end_index = len(offsets) - 1
            
            # Find start position
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            token_start_index -= 1
            
            # Find end position
            while token_end_index > 0 and offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            token_end_index += 1
            
            # Check for valid indices
            if token_start_index < 0 or token_start_index >= len(offsets) or token_end_index < 0 or token_end_index >= len(offsets):
                inputs["start_positions"].append(0)
                inputs["end_positions"].append(0)
            # If answer is truncated, set positions to CLS token
            elif offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
                inputs["start_positions"].append(0)
                inputs["end_positions"].append(0)
            else:
                inputs["start_positions"].append(token_start_index)
                inputs["end_positions"].append(token_end_index)
        
        return inputs
    
    print("Preprocessing training set...")
    processed_train = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    print("Preprocessing validation set...")
    processed_validation = dataset["validation"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )
    
    # Create processed dataset
    processed_dataset = {
        "train": processed_train,
        "validation": processed_validation
    }
    
    # Save processed dataset
    os.makedirs(save_path, exist_ok=True)
    save_dir = os.path.join(save_path, f"squad_{squad_version.replace('.', '')}")
    print(f"Saving processed dataset to {save_dir}")
    for split, dataset_split in processed_dataset.items():
        dataset_split.save_to_disk(os.path.join(save_dir, split))
    
    print("Preprocessing complete!")
    return processed_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SQuAD dataset")
    parser.add_argument("--model_name", default="bert-large-uncased", help="Model name for tokenizer")
    parser.add_argument("--save_path", default="data/processed", help="Path to save processed dataset")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--stride", type=int, default=128, help="Stride for sliding window")
    parser.add_argument("--squad_version", default="v1.1", choices=["v1.1", "v2.0"], help="SQuAD version")
    
    args = parser.parse_args()
    preprocess_squad(
        model_name=args.model_name,
        save_path=args.save_path,
        max_length=args.max_length,
        stride=args.stride,
        squad_version=args.squad_version
    )