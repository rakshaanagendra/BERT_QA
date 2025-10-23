import mlflow
import mlflow.pytorch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from datetime import datetime

# 1️⃣ Set up MLflow experiment
mlflow.set_experiment("BERT_QA_Experiments")

# 2️⃣ Load model + tokenizer once
model_path = r"C:\Users\nagen\Desktop\ML&DLProjects\BERT_QA\model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# 3️⃣ Define multiple question–context pairs
qa_pairs = [
    (
        "Who created Python?",
        "Python was created by Guido van Rossum in 1991 and is widely used for AI, ML, and automation."
    ),
    (
        "When was Python created?",
        "Python was created by Guido van Rossum in 1991 and is widely used for AI, ML, and automation."
    ),
    (
        "What is Python used for?",
        "Python was created by Guido van Rossum in 1991 and is widely used for AI, ML, and automation."
    ),
    (
        "Who invented Java?",
        "Java was developed by James Gosling and released by Sun Microsystems in 1995."
    ),
]

# 4️⃣ Loop through each pair
for i, (question, context) in enumerate(qa_pairs, start=1):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits) + 1

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    answer = tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end])

    # Confidence (average of start and end probs)
    start_prob = torch.nn.functional.softmax(start_logits, dim=-1)[0, answer_start].item()
    end_prob = torch.nn.functional.softmax(end_logits, dim=-1)[0, answer_end - 1].item()
    confidence = round((start_prob + end_prob) / 2, 4)

    # 5️⃣ Start a new MLflow run for each question
    with mlflow.start_run(run_name=f"Run_{i}_{datetime.now().strftime('%H-%M-%S')}"):
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("model_type", model.config.architectures[0])
        mlflow.log_param("question", question)
        mlflow.log_param("context_length", len(context))
        mlflow.log_metric("start_index", answer_start.item())
        mlflow.log_metric("end_index", answer_end.item())
        mlflow.log_metric("confidence", confidence)
        mlflow.log_text(answer, f"predicted_answer_{i}.txt")

    print(f"\n✅ Logged Run {i}:")
    print(f"Question: {question}")
    print(f"Predicted Answer: {answer}")
    print(f"Confidence: {confidence}")
