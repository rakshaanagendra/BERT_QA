# 5_train_bert_qa.py
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
)
import numpy as np
import evaluate

# 1️⃣ Load dataset and tokenizer
dataset = load_dataset("squad")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

max_length = 128
doc_stride = 32

# 2️⃣ Prepare tokenized datasets (reuse same logic)
def prepare_train_features(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[i]
        answer = examples["answers"][sample_index]
        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = tokenized.sequence_ids(i)
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
            start_positions.append(0)
            end_positions.append(0)
        else:
            token_start = context_start
            while token_start < len(offsets) and offsets[token_start][0] <= start_char:
                token_start += 1
            token_start -= 1

            token_end = context_end
            while token_end >= 0 and offsets[token_end][1] >= end_char:
                token_end -= 1
            token_end += 1

            start_positions.append(token_start)
            end_positions.append(token_end)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

tokenized_datasets = dataset.map(
    prepare_train_features,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# 3️⃣ Load pretrained model for QA
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# 4️⃣ Define training arguments
training_args = TrainingArguments(
    output_dir="./models/bert_qa_model",
    eval_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
)

# 5️⃣ Define evaluation metric (Exact Match + F1)
metric = evaluate.load("squad")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    start_logits, end_logits = predictions
    start_positions, end_positions = labels

    # Get predicted start/end token indices
    start_preds = np.argmax(start_logits, axis=1)
    end_preds = np.argmax(end_logits, axis=1)

    # Compute Exact Match and F1
    exact_match = np.mean(start_preds == start_positions) * 100
    f1 = np.mean(end_preds == end_positions) * 100

    return {"exact_match": exact_match, "f1": f1}


# 6️⃣ Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = tokenized_datasets["train"].select(range(20000)),
    eval_dataset  = tokenized_datasets["validation"].select(range(2000)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 7️⃣ Train
trainer.train()
trainer.save_model("./models/bert_qa_model")
