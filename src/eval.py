"""
Evaluation script for Question Answering model on SQuAD dataset.
Computes Exact Match (EM) and F1 scores.
"""
import os
import argparse
import json
import pandas as pd
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import evaluate
import torch
import numpy as np

def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer,
                              n_best_size=20, max_answer_length=30):
    """
    Post-process model predictions to get readable answers from the SQuAD dataset.
    """
    all_start_logits, all_end_logits = raw_predictions
    
    # Build a map from example to its corresponding features
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = {}
    for i, feature in enumerate(features):
        example_index = example_id_to_index[feature["example_id"]]
        features_per_example.setdefault(example_index, []).append(i)
    
    # The dictionaries to be returned
    predictions = {}
    
    # Loop over all examples
    for example_index, example in enumerate(examples):
        # Get features associated with this example
        feature_indices = features_per_example[example_index]
        
        min_null_score = float('inf')
        valid_answers = []
        
        # Loop through all features for this example
        for feature_index in feature_indices:
            # Get logits and offsets
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            
            # Update minimum null prediction
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if feature_null_score < min_null_score:
                min_null_score = feature_null_score
            
            # Go through all possible start and end positions
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip invalid predictions
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                        or end_index < start_index 
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    
                    # Get answer text
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    answer_text = example["context"][start_char:end_char]
                    
                    # Compute score
                    answer_score = start_logits[start_index] + end_logits[end_index]
                    
                    valid_answers.append({
                        "score": answer_score,
                        "text": answer_text,
                        "start_char": start_char,
                        "end_char": end_char
                    })
        
        # Select best answer
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            predictions[example["id"]] = best_answer["text"]
        else:
            predictions[example["id"]] = ""
    
    return predictions

def evaluate_model(args):
    """
    Evaluate a fine-tuned QA model on SQuAD dataset.
    """
    print(f"Loading model and tokenizer from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir)
    
    # Load SQuAD dataset
    print("Loading SQuAD dataset...")
    if args.squad_version == "v1.1":
        dataset = load_dataset("squad", split="validation")
    else:
        dataset = load_dataset("squad_v2", split="validation")
    
    # Tokenize validation dataset
    def prepare_validation_features(examples):
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=args.max_length,
            stride=args.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        # Map features to examples (one example can give several features)
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        
        # Create a mapping between original example IDs and features
        tokenized_examples["example_id"] = []
        for i in range(len(tokenized_examples["input_ids"])):
            sample_idx = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_idx])
        
        return tokenized_examples
    
    # Tokenize validation dataset
    print("Tokenizing validation dataset...")
    validation_features = dataset.map(
        prepare_validation_features,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Create dataloader
    validation_features.set_format(type="torch", columns=["input_ids", "attention_mask", "example_id"])
    
    # Evaluate
    print("Running evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_start_logits = []
    all_end_logits = []
    
    # Process in batches
    for i in range(0, len(validation_features), args.batch_size):
        batch = validation_features[i:i+args.batch_size]
        batch = {k: v.to(device) for k, v in batch.items() if k != "example_id"}
        
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits.cpu().numpy()
            end_logits = outputs.end_logits.cpu().numpy()
            
            all_start_logits.extend(start_logits)
            all_end_logits.extend(end_logits)
    
    # Get predictions
    print("Post-processing predictions...")
    raw_predictions = (all_start_logits, all_end_logits)
    predictions = postprocess_qa_predictions(
        dataset, 
        validation_features, 
        raw_predictions, 
        tokenizer
    )
    
    # Format predictions for evaluation
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset]
    
    # Compute metrics
    print("Computing metrics...")
    metric = evaluate.load("squad")
    results = metric.compute(predictions=formatted_predictions, references=references)
    
    print(f"Exact Match: {results['exact_match']:.2f}")
    print(f"F1 Score: {results['f1']:.2f}")
    
    # Save predictions to CSV
    print(f"Saving predictions to {args.output_file}")
    predictions_df = []
    for ex in dataset:
        predictions_df.append({
            "id": ex["id"],
            "question": ex["question"],
            "context": ex["context"],
            "gold_answer": ex["answers"]["text"][0] if len(ex["answers"]["text"]) > 0 else "",
            "predicted_answer": predictions.get(ex["id"], ""),
        })
    
    pd.DataFrame(predictions_df).to_csv(args.output_file, index=False)
    print(f"Predictions saved to {args.output_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QA model on SQuAD")
    parser.add_argument("--model_dir", default="models/bert-large-squad", help="Directory with fine-tuned model")
    parser.add_argument("--squad_version", default="v1.1", choices=["v1.1", "v2.0"], help="SQuAD version")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--stride", type=int, default=128, help="Stride for sliding window")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--output_file", default="predictions.csv", help="File to save predictions")
    
    args = parser.parse_args()
    evaluate_model(args)