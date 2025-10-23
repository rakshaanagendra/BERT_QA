from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# 1️⃣ Use the official fine-tuned model
model_name = "distilbert-base-cased-distilled-squad"

# 2️⃣ Download both tokenizer and model
print(f"Downloading {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 3️⃣ Save them locally
save_path = r"C:\Users\nagen\Desktop\ML&DLProjects\BERT_QA\model"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"✅ Model and tokenizer saved to {save_path}")
