# _load_dataset.py
from datasets import load_dataset

#  Load the SQuAD v1.1 dataset from Hugging Face
dataset = load_dataset("squad")

# Print the available splits
print(dataset)

# Show one example from the training set
example = dataset["train"][0]
print("\nKeys:", example.keys())        # show what fields are present
print("\nContext:", example["context"]) # the paragraph
print("\nQuestion:", example["question"])
print("\nAnswers:", example["answers"])
