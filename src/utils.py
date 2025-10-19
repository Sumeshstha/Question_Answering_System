"""
Utility functions for the Question Answering system.
"""
import os
import json
import torch
import logging
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_model_and_tokenizer(model_path: str) -> Tuple[AutoModelForQuestionAnswering, AutoTokenizer]:
    """
    Load model and tokenizer from the specified path.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model and tokenizer from {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {e}")
        raise

def get_answer(question: str, context: str, model, tokenizer, device: str = None) -> Dict:
    """
    Get answer for a question based on the provided context.
    
    Args:
        question: Question text
        context: Context text
        model: Question answering model
        tokenizer: Tokenizer for the model
        device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
        
    Returns:
        Dictionary with answer, score, and position
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tokenize input
    inputs = tokenizer(
        question,
        context,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model output
    with torch.no_grad():
        model = model.to(device)
        outputs = model(**inputs)
    
    # Get start and end logits
    start_logits = outputs.start_logits[0].cpu().numpy()
    end_logits = outputs.end_logits[0].cpu().numpy()
    
    # Get answer span
    answer_start = int(torch.argmax(outputs.start_logits))
    answer_end = int(torch.argmax(outputs.end_logits)) + 1
    
    # Convert tokens to answer text
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    answer = tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end])
    
    # Clean up answer (remove special tokens, etc.)
    answer = answer.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").strip()
    
    # Calculate confidence score
    score = float(torch.max(outputs.start_logits).cpu().detach().numpy())
    score = max(0, min(1, score / 10))  # Normalize score between 0 and 1
    
    # Get character positions in original text
    char_start, char_end = get_char_positions(
        context, answer, inputs["offset_mapping"][0].cpu().numpy(), answer_start, answer_end
    )
    
    return {
        "answer": answer,
        "score": score,
        "start": char_start,
        "end": char_end
    }

def get_char_positions(context: str, answer: str, offset_mapping: List, answer_start: int, answer_end: int) -> Tuple[int, int]:
    """
    Get character positions of the answer in the original context.
    
    Args:
        context: Original context text
        answer: Answer text
        offset_mapping: Token to character mapping
        answer_start: Start token index
        answer_end: End token index
        
    Returns:
        Tuple of (start_char, end_char)
    """
    # Default positions if mapping fails
    start_char, end_char = 0, 0
    
    try:
        # Get character positions from offset mapping
        if answer_start < len(offset_mapping) and answer_end < len(offset_mapping):
            start_char = int(offset_mapping[answer_start][0])
            end_char = int(offset_mapping[answer_end-1][1])
            
            # Adjust if positions seem incorrect
            if end_char <= start_char:
                # Try to find the answer in the context
                answer_pos = context.lower().find(answer.lower())
                if answer_pos != -1:
                    start_char = answer_pos
                    end_char = start_char + len(answer)
    except Exception as e:
        logger.error(f"Error getting character positions: {e}")
    
    return start_char, end_char

def save_predictions(predictions: List[Dict], output_file: str) -> None:
    """
    Save predictions to a JSON file.
    
    Args:
        predictions: List of prediction dictionaries
        output_file: Path to output file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Predictions saved to {output_file}")

def calculate_metrics(predictions: List[Dict], references: List[Dict]) -> Dict:
    """
    Calculate Exact Match (EM) and F1 score for predictions.
    
    Args:
        predictions: List of prediction dictionaries
        references: List of reference dictionaries
        
    Returns:
        Dictionary with metrics
    """
    exact_match = 0
    f1_total = 0
    
    for pred, ref in zip(predictions, references):
        pred_text = pred["prediction_text"].lower().strip()
        ref_texts = [r.lower().strip() for r in ref["answers"]["text"]]
        
        # Calculate Exact Match
        if any(pred_text == ref_text for ref_text in ref_texts):
            exact_match += 1
        
        # Calculate F1 score
        best_f1 = 0
        for ref_text in ref_texts:
            f1 = compute_f1(pred_text, ref_text)
            best_f1 = max(best_f1, f1)
        f1_total += best_f1
    
    total = len(predictions)
    return {
        "exact_match": 100.0 * exact_match / total if total > 0 else 0,
        "f1": 100.0 * f1_total / total if total > 0 else 0
    }

def compute_f1(prediction: str, reference: str) -> float:
    """
    Compute F1 score between prediction and reference.
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
        
    Returns:
        F1 score
    """
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return int(pred_tokens == ref_tokens)
    
    # Count common tokens
    common = sum(1 for token in pred_tokens if token in ref_tokens)
    
    # If no common tokens, F1 = 0
    if common == 0:
        return 0
    
    # Calculate precision and recall
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    
    # Calculate F1
    f1 = 2 * precision * recall / (precision + recall)
    return f1