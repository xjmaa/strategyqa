# preprocess_fever.py
from datasets import load_dataset
import json

# Load the FEVER dataset
fever = load_dataset("fever", "v1.0")

# Convert to StrategyQA format
def fever_to_strategyqa_format(dataset, output_file, max_samples=10000):
    formatted_data = []
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        question = example["claim"]  # FEVER's claim is the question
        label = example["label"]      # Label: "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"
        
        # Convert labels to yes/no format for consistency
        answer = "yes" if label == "SUPPORTS" else "no"

        formatted_data.append({"question": question, "answer": answer})

    # Save as JSON
    with open(output_file, "w") as f:
        json.dump(formatted_data, f, indent=2)

# Convert train and validation sets
fever_to_strategyqa_format(fever["train"], "data/fever_train.json")
fever_to_strategyqa_format(fever["paper_test"], "data/fever_dev.json")
print("success")
