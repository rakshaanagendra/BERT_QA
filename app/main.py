from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import mlflow
from datetime import datetime

# 1️⃣ Define the FastAPI app
app = FastAPI(title="BERT QA API", description="Ask questions with a fine-tuned DistilBERT model")

# 2️⃣ Define the input schema (for incoming JSON requests)
class QARequest(BaseModel):
    question: str
    context: str

# 3️⃣ Load model and tokenizer once (for efficiency)
model_path = r"C:\Users\nagen\Desktop\ML&DLProjects\BERT_QA\model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# 4️⃣ Root endpoint (simple test)
@app.get("/")
def home():
    return {"message": "Welcome to the BERT QA API! Use /predict to ask questions."}

# 5️⃣ Prediction endpoint
@app.post("/predict")
def predict(request: QARequest):
    """
    Accepts a question and context, returns the predicted answer + confidence.
    """

    # Tokenize inputs
    inputs = tokenizer.encode_plus(request.question, request.context, return_tensors="pt", truncation=True)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    # Run model inference (no gradients)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Determine best start/end positions
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits) + 1

    # Convert token indices back to text
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    answer = tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end])

    # Compute confidence score
    start_prob = torch.nn.functional.softmax(start_logits, dim=-1)[0, answer_start].item()
    end_prob = torch.nn.functional.softmax(end_logits, dim=-1)[0, answer_end - 1].item()
    confidence = round((start_prob + end_prob) / 2, 4)

    # Log to MLflow
    mlflow.set_experiment("BERT_QA_API_Logs")
    with mlflow.start_run(run_name=f"API_Run_{datetime.now().strftime('%H-%M-%S')}"):
        mlflow.log_param("question", request.question)
        mlflow.log_param("context_length", len(request.context))
        mlflow.log_metric("confidence", confidence)
        mlflow.log_text(answer, "predicted_answer.txt")

    # Return JSON response
    return {
        "question": request.question,
        "predicted_answer": answer,
        "confidence": confidence,
        "start_index": answer_start.item(),
        "end_index": answer_end.item(),
    }
