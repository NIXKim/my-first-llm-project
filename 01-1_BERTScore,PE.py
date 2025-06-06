import json
import evaluate
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random

# Prompt Engineering 방식 선택: "zeroshot", "role", "format", "fewshot"
prompt_style = "fewshot"

# Few-shot 예제 정의
fewshot_examples = [
    {
        "instruction": "Provide emergency steps for earthquake safety.",
        "input": "You are indoors during an earthquake.",
        "output": "Drop to the ground, take cover under sturdy furniture, and hold on until the shaking stops."
    },
    {
        "instruction": "Summarize the following flood warning message.",
        "input": "Heavy rainfall is expected overnight, and river levels are rising rapidly. Residents near the river should prepare for possible evacuation.",
        "output": "Rising river levels due to heavy rain may require evacuation for nearby residents."
    },
    {
        "instruction": "Translate this emergency message into Korean.",
        "input": "A tsunami is expected to hit the coastal areas within the next hour. Seek higher ground immediately.",
        "output": "A tsunami is expected to hit the coastal areas within the next hour. Seek higher ground immediately."
    },
    {
        "instruction": "List general safety tips for wildfire emergencies.",
        "input": "",
        "output": "Prepare an emergency kit, follow evacuation orders, keep indoor air clean, and stay informed through official alerts."
    }
]

# 모델 경로 및 토크나이저 로드(inst_model_path"./flan_tuned_model_v10k", "./flan_tuned_model_v50k", "./flan_tuned_model_v100k", base_model_path"google/flan-t5-base")
model_path = "./flan_tuned_model_v100k"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 평가 데이터셋 로드
with open("eval_instruction_dataset_1000.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    
# 일부 샘플로 제한하여 평가 (에러 방지를 위해 최대 50개)
samples = random.sample(data, 100)

predictions = []
references = []

for sample in samples:
    instruction = sample["instruction"]
    input_text = sample["input"]
    reference = sample["output"]

    if prompt_style == "zeroshot":
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
    elif prompt_style == "role":
        prompt = f"You are a helpful assistant.\nFollow the instruction:\nInstruction: {instruction}\nInput: {input_text}"
    elif prompt_style == "format":
        prompt = f"<Instruction>: {instruction}\n<Input>: {input_text}\n<Output>:"
    elif prompt_style == "fewshot":
        prompt = ""
        for ex in fewshot_examples:
            prompt += f"Instruction: {ex['instruction']}\nInput: {ex['input']}\nOutput: {ex['output']}\n\n"
        prompt += f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs, max_length=128)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    predictions.append(decoded_output)
    references.append(reference)

    print(f"\nInstruction: {instruction}")
    print(f"Input: {input_text}")
    print(f"Model Output: {decoded_output}")
    print(f"Reference Output: {reference}")
    
# BERTScore
bertscore = evaluate.load("bertscore")
bert_result = bertscore.compute(predictions=predictions, references=references, lang="en")
avg_bert_f1 = sum(bert_result["f1"]) / len(bert_result["f1"])
print(f"BERTScore (F1): {avg_bert_f1:.4f}")
