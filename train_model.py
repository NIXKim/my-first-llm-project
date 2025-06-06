from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from prepare_dataset import load_and_prepare_dataset
from tokenize_dataset import tokenize_dataset

model_name = "google/flan-t5-base"
dataset_path = "inst_dataset_100k.json"

# Step 1: Load and Preprocess dataset
dataset = load_and_prepare_dataset(dataset_path)

# Step 2: Tokenize dataset
tokenized_dataset, tokenizer = tokenize_dataset(dataset, model_name)

# Ensure tokenizer and model have a pad_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Step 3: Load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

# Step 4: Training Setup
training_args = Seq2SeqTrainingArguments(
    output_dir = "./flan_tuned_model_v100k",  
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_dir="./logs",
    disable_tqdm=False,
    save_total_limit=1,
    save_steps=500,
    predict_with_generate=True,
    # fp16=True --> CUDA GPU 전용 기능이라 Mac에서는 지원이 안돼서 주석처리
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
)

# Step 5: Train
trainer.train()
trainer.save_model("./flan_tuned_model_v100k")