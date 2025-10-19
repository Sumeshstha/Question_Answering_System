"""
Training script for fine-tuning BERT-large model on SQuAD dataset.
"""
import os
import argparse
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
import evaluate

# Load SQuAD metrics
metric = evaluate.load("squad")

def compute_metrics(eval_pred):
    """
    Compute SQuAD metrics (Exact Match and F1 score).
    """
    logits, labels = eval_pred
    
    # Get start and end logits
    start_logits, end_logits = logits
    start_positions, end_positions = labels
    
    # Compute predictions
    predictions = {
        'id': [],
        'prediction_text': [],
        'no_answer_probability': []
    }
    
    references = {
        'id': [],
        'answers': []
    }
    
    # Process predictions and references
    # This is a simplified version - in practice, you'd need to map back to original text
    # using offset mapping and dataset information
    
    # Return metrics
    return metric.compute(predictions=predictions, references=references)

def train(args):
    """
    Fine-tune a BERT model on SQuAD dataset.
    """
    print(f"Loading processed dataset from {args.data_dir}")
    try:
        train_dataset = load_from_disk(os.path.join(args.data_dir, "train"))
        eval_dataset = load_from_disk(os.path.join(args.data_dir, "validation"))
    except Exception as e:
        print(f"Error loading processed dataset: {e}")
        print("Please run preprocessing.py first to prepare the dataset.")
        return
    
    print(f"Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb/tensorboard reporting
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        # compute_metrics=compute_metrics,  # Uncomment for full metrics computation
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training complete!")
    
    # Evaluate the model
    print("Evaluating model...")
    results = trainer.evaluate()
    print(f"Evaluation results: {results}")
    
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT on SQuAD")
    parser.add_argument("--model_name", default="bert-large-uncased", help="Pretrained model name")
    parser.add_argument("--data_dir", default="data/processed/squad_v11", help="Directory with processed dataset")
    parser.add_argument("--output_dir", default="models/bert-large-squad", help="Directory to save model")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    
    args = parser.parse_args()
    train(args)