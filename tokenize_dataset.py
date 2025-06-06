from transformers import AutoTokenizer

def tokenize_dataset(dataset, model_name="google/flan-t5-base", max_input_length=512, max_target_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(example):
        model_input = tokenizer(
            example["input_text"],
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            example["target_text"],
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
        )
        model_input["labels"] = labels["input_ids"]
        return model_input
    
    return dataset.map(tokenize, remove_columns=dataset.column_names), tokenizer