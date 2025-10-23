# 4_preprocess_dataset.py
from datasets import load_dataset
from transformers import AutoTokenizer

# 1️⃣ Load SQuAD and tokenizer
dataset = load_dataset("squad")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

max_length = 384
doc_stride = 128

# 2️⃣ Function to prepare each example
def prepare_train_features(examples):
    # Tokenize question–context pairs
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",          # truncate context only
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,    # handle long contexts
        return_offsets_mapping=True,       # char↔token mapping
        padding="max_length",
    )

    # bookkeeping for long contexts
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    # 3️⃣ For each tokenized chunk
    for i, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[i]
        answer = examples["answers"][sample_index]
        # If no answer text (rare), label as CLS token index 0
        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # Tokens belonging to the context portion
        sequence_ids = tokenized.sequence_ids(i)
        # Find where context tokens start & end
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        # If answer not inside this chunk → mark as CLS
        if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
            start_positions.append(0)
            end_positions.append(0)
        else:
            # find token where answer starts
            token_start = context_start
            while token_start < len(offsets) and offsets[token_start][0] <= start_char:
                token_start += 1
            start_positions.append(token_start - 1)

            # find token where answer ends
            token_end = context_end
            while token_end >= 0 and offsets[token_end][1] >= end_char:
                token_end -= 1
            end_positions.append(token_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

# 4️⃣ Apply preprocessing to the training and validation splits
train_dataset = dataset["train"].map(
    prepare_train_features,
    batched=True,
    remove_columns=dataset["train"].column_names,
)
valid_dataset = dataset["validation"].map(
    prepare_train_features,
    batched=True,
    remove_columns=dataset["validation"].column_names,
)

print("Train columns:", train_dataset.column_names)
print("Example start/end positions:", train_dataset[0]["start_positions"], train_dataset[0]["end_positions"])
print("Validation columns:", valid_dataset.column_names)

tokens = tokenizer.convert_ids_to_tokens(train_dataset[0]["input_ids"])
print("\nStart index:", train_dataset[0]["start_positions"])
print("End index:", train_dataset[0]["end_positions"])
print("Answer tokens:", tokens[train_dataset[0]["start_positions"]:train_dataset[0]["end_positions"] + 1])
