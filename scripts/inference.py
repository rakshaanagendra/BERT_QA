from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import pandas as pd

# 1️⃣ Load the model and tokenizer
model_path = r"C:\Users\nagen\Desktop\ML&DLProjects\BERT_QA\model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# 2️⃣ Example input
context = "Python was created by Guido van Rossum in 1991 and is widely used for AI, ML, and automation."
question = "Who created Python?"

# 3️⃣ Tokenize
inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True)
if "token_type_ids" in inputs:
    del inputs["token_type_ids"]

# 4️⃣ Run the model (no gradient tracking)
with torch.no_grad():
    outputs = model(**inputs)

# 5️⃣ Extract start and end logits
start_logits = outputs.start_logits[0]
end_logits = outputs.end_logits[0]

# 6️⃣ Get predicted start and end indices
answer_start = torch.argmax(start_logits)
answer_end = torch.argmax(end_logits) + 1

# 7️⃣ Convert token IDs back to readable tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# 8️⃣ Build a table to visualize token scores
# Convert logits to probabilities for easier interpretation
start_probs = torch.nn.functional.softmax(start_logits, dim=0)
end_probs = torch.nn.functional.softmax(end_logits, dim=0)

data = []
for i, token in enumerate(tokens):
    data.append({
        "Index": i,
        "Token": token,
        "Start_Prob": round(start_probs[i].item(), 4),
        "End_Prob": round(end_probs[i].item(), 4),
        "Chosen": "⬅️ Start" if i == answer_start else ("➡️ End" if i == answer_end - 1 else "")
    })

df = pd.DataFrame(data)
print("\n=== Token Probabilities ===")
print(df.to_string(index=False))

# 9️⃣ Reconstruct the final predicted answer text
answer = tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end])

# 🔟 Print final answer summary
print("\nQuestion:", question)
print("Predicted Answer:", answer)
print("Start Index:", answer_start.item())
print("End Index:", answer_end.item())
