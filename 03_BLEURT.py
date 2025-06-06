import json
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

# 모델 경로 및 토크나이저 로드(inst_model_path"./flan_tuned_model_v10k", "./flan_tuned_model_v50k", "./flan_tuned_model_v100k", base_model_path"google/flan-t5-base")
model_path = "./flan_tuned_model_v100k"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 평가 데이터셋 로드
with open("eval_instruction_dataset_1000.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 샘플 선택
samples = random.sample(data, 100)

predictions = []
references = []

# Zero-shot 프롬프트 기반 추론
for sample in samples:
    instruction = sample["instruction"]
    input_text = sample["input"]
    reference = sample["output"]

    prompt = f"Instruction: {instruction}\nInput: {input_text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs, max_length=128)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    predictions.append(decoded)
    references.append(reference)

# BLEURT 평가
bleurt = evaluate.load("bleurt", config_name="BLEURT-20")
scores = bleurt.compute(predictions=predictions, references=references)

avg_bleurt = sum(scores["scores"]) / len(scores["scores"])
print(f"Average BLEURT Score: {avg_bleurt:.4f}")
