# Question Answering System

A complete end-to-end Question Answering (QA) system built with Hugging Face Transformers, PyTorch, FastAPI, and Streamlit. This system can answer questions based on provided context paragraphs or retrieve information from Wikipedia.

## Project Overview

This project implements a Question Answering system with the following components:

- **Model**: BERT-large-uncased fine-tuned on SQuAD dataset
- **Backend**: FastAPI REST API for serving predictions
- **Frontend**: Streamlit web interface for user interaction
- **Deployment**: Docker configuration for easy deployment

## Features

- Extract precise answers from context paragraphs
- Confidence scoring for answers
- Optional Wikipedia retrieval for finding relevant context
- Modern, responsive UI with Streamlit
- Dockerized deployment for both API and frontend
- Support for multilingual models (optional)

## Directory Structure

```
qa-project/
├─ data/
│  └─ squad/                  # SQuAD dataset files
├─ src/
│  ├─ preprocessing.py        # Data preprocessing script
│  ├─ train.py                # Model training script
│  ├─ eval.py                 # Evaluation script
│  ├─ serve.py                # FastAPI backend
│  ├─ retrieval.py            # Wikipedia retrieval module
│  └─ utils.py                # Utility functions
├─ frontend/
│  └─ streamlit_app.py        # Streamlit frontend
├─ models/                    # Directory for saved models
├─ docker/
│  ├─ Dockerfile.api          # Dockerfile for API
│  └─ Dockerfile.frontend     # Dockerfile for frontend
├─ docker-compose.yml         # Docker Compose configuration
├─ requirements.txt           # Python dependencies
└─ README.md                  # This file
```

## Setup Instructions

### Environment Setup

1. Create a virtual environment:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Data Preprocessing

Download and preprocess the SQuAD dataset:

```bash
# For SQuAD v1.1
python src/preprocessing.py --squad_version v1.1

# For SQuAD v2.0
python src/preprocessing.py --squad_version v2.0
```

### Model Training

Fine-tune the BERT model on SQuAD:

```bash
python src/train.py --model_name bert-large-uncased --data_dir data/processed/squad_v11 --output_dir models/bert-large-squad --epochs 3
```

### Model Evaluation

Evaluate the fine-tuned model:

```bash
python src/eval.py --model_dir models/bert-large-squad --output_file predictions.csv
```

## Running the Application

### Local Development

1. Start the backend API:

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
```

2. Start the Streamlit frontend:

```bash
streamlit run frontend/streamlit_app.py
```

3. Open your browser at http://localhost:8501

### Using Docker

1. Build and start the containers:

```bash
docker-compose up --build
```

2. Access the application:
   - Frontend: http://localhost:8501
   - API: http://localhost:8000
   - API documentation: http://localhost:8000/docs

## API Usage

### Example with cURL

```bash
curl -X POST "http://localhost:8000/answer" \
     -H "Content-Type: application/json" \
     -d '{"question": "Who won the Super Bowl?", "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title."}'
```

### Example with Python

```python
import requests

response = requests.post(
    "http://localhost:8000/answer",
    json={
        "question": "Who won the Super Bowl?",
        "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title."
    }
)

print(response.json())
```

## Deployment Options

### Option A: Hugging Face Spaces

1. Create a new Space on Hugging Face with Streamlit SDK
2. Upload the following files:
   - `frontend/streamlit_app.py` (renamed to `app.py`)
   - `requirements.txt`
3. Modify the app to load the model directly from Hugging Face Hub

### Option B: Render/Heroku

1. Push your Docker image to a registry
2. Configure the deployment platform to use your Docker image
3. Set environment variables as needed

## Performance Metrics

The BERT-large model fine-tuned on SQuAD v1.1 typically achieves:

- Exact Match (EM): ~80-85%
- F1 Score: ~88-92%

Performance may vary based on training parameters and hardware.

## Multilingual Support

For multilingual support, replace `bert-large-uncased` with:

- `bert-base-multilingual-cased` for 104 languages
- `xlm-roberta-large` for even better multilingual performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for Transformers library
- SQuAD dataset creators
- FastAPI and Streamlit for the web frameworks