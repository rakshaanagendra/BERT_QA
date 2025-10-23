# _tokenize_dataset.py
from datasets import load_dataset
from transformers import AutoTokenizer

# 1️⃣ Load dataset
dataset = load_dataset("squad")

# 2️⃣ Load pretrained tokenizer (BERT base, cased)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# 3️⃣ Define some config parameters
max_length = 384     # BERT's max input length
doc_stride = 128     # overlap for long contexts

# 4️⃣ Example: show tokenization for one question-context pair
sample = dataset["train"][0]
question = sample["question"]
context  = sample["context"]

print("\nQUESTION:", question)
print("\nCONTEXT:", context[:200] + "...")  # show only first 200 chars

# 5️⃣ Tokenize
inputs = tokenizer(
    question,
    context,
    max_length=max_length,
    truncation="only_second",   # truncate context only
    stride=doc_stride,
    return_overflowing_tokens=True,  # create multiple windows if needed
    return_offsets_mapping=True,     # map tokens back to original text
    padding="max_length",            # pad to max_length
)

# 6️⃣ Print token details
print("\nNumber of windows:", len(inputs["input_ids"]))

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("\nFirst 30 tokens:\n", tokens[:30])
