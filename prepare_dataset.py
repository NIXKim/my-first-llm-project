import json
from datasets import Dataset

def load_and_prepare_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    dataset = Dataset.from_list(raw_data)
    
    def preprocess(example):
        input_text = f"Instruction:: {example['instruction']}\nInput: {example['input']}"
        return {
            "input_text": input_text,
            "target_text": example["output"]
        }
        
    return dataset.map(preprocess)