import json
import time
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 모델 경로 및 토크나이저 로드(inst_model_path"./flan_tuned_model_v10k", "./flan_tuned_model_v50k", "./flan_tuned_model_v100k", base_model_path"google/flan-t5-base")
model_path = "./flan_tuned_model_v100k"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 평가 데이터셋 로드
with open("eval_instruction_dataset_1000.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 평가 횟수 설정
total_runs = 320  # 필요시 줄이거나 늘릴 수 있음
if total_runs > len(data):
    print(f"경고: 데이터셋 크기({len(data)})보다 평가 횟수({total_runs})가 많습니다. 중복 사용됩니다.")

latencies = []

for i in range(total_runs):
    sample = random.choice(data)  # 중복 허용
    instruction = sample["instruction"]
    input_text = sample["input"]

    prompt = f"Instruction: {instruction}\nInput: {input_text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

    start_time = time.time()
    _ = model.generate(**inputs, max_length=128)
    latency = time.time() - start_time
    latencies.append(latency)

    if (i + 1) % 500 == 0:
        print(f"{i + 1}개 완료")

avg_latency = sum(latencies) / len(latencies)
print(f"\n총 {total_runs}회 실행 후 평균 Latency: {avg_latency:.4f}초")