"""
FastAPI backend for serving the Question Answering model.

This version will try to load a Hugging Face question-answering pipeline from a
local model directory (`models/bert-squad/`). If transformers or the local
model are not available, it falls back to a lightweight mock pipeline (same
API) so the server stays responsive for development.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import random
from typing import Optional

app = FastAPI(title="Question Answering API", 
              description="API for answering questions based on provided context",
              version="1.1.0")


# --- Mock pipeline (fallback) -------------------------------------------------
class MockQAPipeline:
    def __init__(self):
        print("Initialized mock QA pipeline")
    
    def __call__(self, question, context, handle_impossible_answer=False):
        words = context.split()
        if len(words) < 5:
            start_idx = 0
            end_idx = len(words)
        else:
            start_idx = random.randint(0, max(0, len(words) - 5))
            end_idx = min(start_idx + random.randint(1, 5), len(words))
        answer_words = words[start_idx:end_idx]
        answer = " ".join(answer_words)
        start = context.find(answer)
        end = start + len(answer) if start >= 0 else 0
        return {"answer": answer, "score": random.uniform(0.7, 0.99), "start": start, "end": end}


# --- Try to load real Hugging Face pipeline ----------------------------------
qa_pipeline = None
model_info_cache = {"model_path": "Mock QA Model", "device": "CPU", "pipeline_type": "question-answering"}

@app.on_event("startup")
async def load_model():
    """Attempt to load a local Hugging Face QA pipeline; fall back to mock."""
    global qa_pipeline, model_info_cache
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "bert-squad"))
    try:
        # local import to keep startup fast when transformers aren't installed
        from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
        print("transformers available, attempting to load local model if present...")

        if os.path.isdir(model_dir) and os.listdir(model_dir):
            print(f"Found local model directory: {model_dir}")
            # Load tokenizer and model from local files (CPU by default)
            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            model = AutoModelForQuestionAnswering.from_pretrained(model_dir, local_files_only=True)
            qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=-1)
            model_info_cache = {"model_path": model_dir, "device": "CPU", "pipeline_type": "question-answering"}
            print("Loaded local Hugging Face QA model from models/bert-squad/")
        else:
            print(f"Local model not found at {model_dir} or directory empty. Falling back to default HF model from hub... (requires internet)")
            # Try to load a small default model from the hub to keep memory small
            try:
                qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=-1)
                model_info_cache = {"model_path": "distilbert-base-uncased-distilled-squad (hub)", "device": "CPU", "pipeline_type": "question-answering"}
                print("Loaded DistilBERT QA model from Hugging Face Hub")
            except Exception as e:
                print(f"Failed to load hub model: {e}. Using mock pipeline.")
                qa_pipeline = MockQAPipeline()

    except Exception as e:
        # transformers not installed or other import error
        print(f"Could not import transformers or load model: {e}. Using mock pipeline.")
        qa_pipeline = MockQAPipeline()


# --- Request/response models -----------------------------------------------
class QARequest(BaseModel):
    question: str
    context: str

class QAResponse(BaseModel):
    answer: str
    score: float
    start: int
    end: int


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Question Answering API is running"}


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Question Answering API is running"}


def _call_pipeline(question: str, context: str) -> dict:
    """Call the QA pipeline and normalize the output to our response format."""
    global qa_pipeline
    result = qa_pipeline(question=question, context=context)

    # Hugging Face pipeline sometimes returns dict directly, or nested keys
    if isinstance(result, dict) and "answer" in result:
        return {"answer": result.get("answer", ""),
                "score": float(result.get("score", 0.0)),
                "start": int(result.get("start", -1)),
                "end": int(result.get("end", -1))}

    # If result is a list (some versions), take the top item
    if isinstance(result, list) and result:
        top = result[0]
        return {"answer": top.get("answer", ""),
                "score": float(top.get("score", 0.0)),
                "start": int(top.get("start", -1)),
                "end": int(top.get("end", -1))}

    # Fallback: try to coerce
    return {"answer": str(result), "score": 0.0, "start": -1, "end": -1}


@app.post("/answer", response_model=QAResponse)
def answer(req: QARequest):
    if qa_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not req.question or not req.context:
        raise HTTPException(status_code=400, detail="Question and context are required")

    try:
        result = _call_pipeline(req.question, req.context)
        return {"answer": result["answer"], "score": result["score"], "start": result["start"], "end": result["end"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/model-info")
def model_info():
    if qa_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_info_cache


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)