import json
import evaluate
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import torch

# 모델 경로 및 토크나이저 로드(inst_model_path"mistralai/Mistral-7B-Instruct-v0.1")
model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# 평가 데이터셋 로드
with open("eval_instruction_dataset_1000.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    
# 일부 샘플로 제한하여 평가 (에러 방지를 위해 최대 50개)
samples = random.sample(data, 20)

# 결과 저장 리스트
predictions = []
references = []

for sample in samples:
    instruction = sample["instruction"]
    input_text = sample["input"]
    reference = sample["output"]
    
    prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        num_beams=1
    )
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    predictions.append(decoded_output)
    references.append(reference)
    
# BERTScore
bertscore = evaluate.load("bertscore")
bert_result = bertscore.compute(predictions=predictions, references=references, lang="en")
avg_bert_f1 = sum(bert_result["f1"]) / len(bert_result["f1"])
print(f"BERTScore (F1): {avg_bert_f1:.4f}")