import json
import random
import time
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from comet import download_model, load_from_checkpoint

# Few-shot examples for prompts
fewshot_examples = [
    {
        "instruction": "Translate the emergency alert into English.",
        "input": "A wildfire is approaching the residential area. Evacuate immediately.",
        "output": "A wildfire is approaching the residential area. Evacuate immediately."
    },
    {
        "instruction": "Summarize the emergency broadcast.",
        "input": "Due to a powerful typhoon, all schools and public offices will be closed tomorrow.",
        "output": "All schools and offices will close tomorrow due to the typhoon."
    },
    {
        "instruction": "Provide first aid steps for a burn injury.",
        "input": "",
        "output": "Cool the burn with water, cover with a sterile bandage, and seek medical help."
    }
]

# Tokenizers 경고 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 모델 경로 및 토크나이저 로드(inst_model_path"./flan_tuned_model_v10k", "./flan_tuned_model_v50k", "./flan_tuned_model_v100k", base_model_path"google/flan-t5-base")
model_path = "./flan_tuned_model_v100k"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 평가 데이터셋 로드
with open("eval_instruction_dataset_1000.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 일부 샘플로 제한하여 평가 (예: 50개)
samples = random.sample(data, 100)

# 결과 저장 리스트
predictions = []
references = []
sources = []
latencies = []

for sample in samples:
    instruction = sample["instruction"]
    input_text = sample["input"]
    reference = sample["output"]

    # Prompt Engineering 방식 선택: "zeroshot", "role", "format", "fewshot"
    prompt_strategy = "fewshot"  # Change this to zero-shot, format, or role as needed

    if prompt_strategy == "zeroshot":
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
    elif prompt_strategy == "format":
        prompt = f"[TASK] {instruction}\n[CONTEXT] {input_text}\n[RESPONSE]"
    elif prompt_strategy == "role":
        prompt = f"You are a disaster response expert.\nYour task is to {instruction.lower()}.\nInput: {input_text}\nOutput:"
    elif prompt_strategy == "fewshot":
        examples = "\n\n".join(
            [f"Instruction: {ex['instruction']}\nInput: {ex['input']}\nOutput: {ex['output']}" for ex in fewshot_examples[:3]]
        )
        prompt = f"{examples}\n\nInstruction: {instruction}\nInput: {input_text}\nOutput:"
    else:
        raise ValueError("Invalid prompt strategy.")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    start_time = time.time()
    outputs = model.generate(**inputs, max_length=128)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    latency = time.time() - start_time
    latencies.append(latency)

    predictions.append(decoded_output)
    references.append(reference)
    sources.append(prompt)

    print(f"\nInstruction: {instruction}")
    print(f"Input: {input_text}")
    print(f"Model Output: {decoded_output}")
    print(f"Reference Output: {reference}")
    print(f"Latency: {latency:.4f} seconds")

# COMET 평가
print("\nEvaluating with COMET...")
comet_ckpt_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_ckpt_path)

data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, predictions, references)]

# COMET 튜플 반환값 언패킹 수정
prediction_result = comet_model.predict(
    data,
    batch_size=8,
    gpus=1 if comet_model.device.type == "cuda" else 0,
    num_workers=1
)
score_list = prediction_result.scores
avg_score = prediction_result.system_score

try:
    score_float = float(avg_score)
    print(f"\nAverage COMET Score: {score_float:.4f}")
except (ValueError, TypeError):
    print(f"\nCOMET 평균 점수 형식이 잘못되었습니다: {avg_score}")

# 평균 Latency 출력
avg_latency = sum(latencies) / len(latencies)
print(f"\nAverage Latency: {avg_latency:.4f} seconds")