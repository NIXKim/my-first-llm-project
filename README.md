# my-llm-project
[한국어]
# 🌍 Disaster-Aware LLM Optimization with Instruction Tuning and Prompt Engineering

본 프로젝트는 재난 대응을 위한 특화 언어모델(Large Language Model, LLM)을 개발하고, Instruction Tuning과 Prompt Engineering 기법을 활용하여 모델 성능을 최적화하는 연구입니다.

## 🧠 Research Overview

- 전 세계적으로 발생하는 재난 상황에 실시간으로 대응할 수 있는 LLM 필요성 증가
- 범용 LLM의 한계를 보완하기 위해 재난 특화 Instruction Dataset으로 모델 튜닝
- 다양한 Prompt Engineering 기법을 적용해 LLM의 응답 품질을 추가 개선

## 📚 Model Structure

- **Base Model**: `google/flan-t5-base` (0.25B parameter)
- **Tuning Dataset**: 재난 대응 Instruction Dataset (10K / 50K / 100K)
- **Prompt Strategies**:
  - Baseline Prompt
  - Zero-shot Prompt
  - Role Prompt
  - Format Prompt
  - Few-shot Prompt
- **Expert MoE 모델** (지진 / 홍수 / 화재 유형별 분화)

## 📊 Evaluation Metrics

| Category       | Metric        | Purpose                          |
|----------------|---------------|----------------------------------|
| 정량 평가      | BERTScore     | 의미 유사도 기반 정밀 평가      |
|                | COMET         | 문장 수준 번역 품질 평가        |
|                | BLEURT        | 정교한 문장 비교                 |
| 정성 평가      | GPT-Judge     | Correctness / Helpfulness / Safety / Overall Score |
| 기타           | Latency       | 응답 속도 측정                   |

## 🧪 Key Results

- **Instruction Tuning 모델**, 최대 **BERTScore 0.88** 달성 (SOTA 0.95에 근접)
- Role / Format Prompt 적용 시 GPT-Judge 평가 기준에서 **정확도 및 유용성 향상**
- Few-shot Prompt에서는 일부 실험에서 응답 누락 문제 확인
- 데이터셋 규모 증가 시 BLEU/ROUGE/COMET 등 **정량 지표 향상** 경향 확인

## ✨ Contributions

- Instruction Tuning + Prompt Engineering **연속 결합 방식**을 실험적으로 입증
- 소형 모델(0.25B) 기반으로도 **도메인 특화 성능 극대화** 가능성 제시
- **정성 평가 자동화** (GPT-Judge 기반)를 통한 실용성 중심 LLM 분석 도입
- 재난 유형별 전문가 모델(MoE) 구조 실험

## 🔭 Future Work

- 실시간 정보 검색 결합 (RAG)
- 다국어 재난 대응 모델 확장
- Prompt 자동 생성 및 최적화 연구
- 모델 안전성 강화를 위한 응답 필터링 기법 연구

## 📝 Citation

> Kim, Hwangmin. “Disaster-Aware LLM Optimization with Instruction Tuning and Prompt Engineering.” Graduation Thesis, Yonsei University, 2025.

---

## 📂 Repository Structure

[English]
# 🌍 Disaster-Aware LLM Optimization with Instruction Tuning and Prompt Engineering

This project focuses on optimizing Large Language Models (LLMs) tailored for disaster response by applying **Instruction Tuning** and **Prompt Engineering**. The goal is to build a lightweight, efficient, and domain-specific LLM capable of generating helpful, safe, and accurate responses in emergency situations.

## 🧠 Research Overview

- The increasing frequency of global disasters demands real-time, reliable AI support
- General-purpose LLMs are insufficient in disaster domains
- This study builds disaster-specific instruction datasets and applies tuning & prompt techniques to enhance model performance

## 📚 Model Architecture

- **Base Model**: `google/flan-t5-base` (0.25B parameters)
- **Training Data**: Custom disaster instruction datasets (sizes: 10K / 50K / 100K)
- **Prompt Strategies**:
  - Baseline Prompt
  - Zero-shot Prompt
  - Role Prompt
  - Format Prompt
  - Few-shot Prompt
- **Expert MoE (Mixture of Experts)**: Specialized for Earthquake, Flood, Fire scenarios

## 📊 Evaluation Metrics

| Category     | Metric        | Description                                     |
|--------------|---------------|-------------------------------------------------|
| Quantitative | BERTScore     | Semantic similarity evaluation                  |
|              | COMET         | Sentence-level generation quality               |
|              | BLEURT        | Fine-grained textual similarity scoring         |
| Qualitative  | GPT-Judge     | Assesses Correctness, Helpfulness, Safety, Overall |
| Other        | Latency       | Measures response time                          |

## 🧪 Key Results

- Instruction-tuned models achieved **BERTScore of 0.88**, close to SOTA (0.95) with only 0.25B parameters
- Role and Format prompts significantly improved **GPT-Judge scores**
- Few-shot prompts sometimes caused incomplete or empty responses
- Performance consistently improved across BLEU/ROUGE/COMET as dataset size increased

## ✨ Contributions

- First experimental study combining Instruction Tuning **followed by Prompt Engineering**
- Demonstrated high performance in **domain-specific tasks with small-scale models**
- Introduced **LLM-based qualitative evaluation** using GPT-Judge
- Explored **expert-specific MoE models** for different disaster types

## 🔭 Future Work

- Integrate RAG (Retrieval-Augmented Generation) for real-time updates
- Extend to multilingual disaster LLMs
- Automate prompt generation and optimization
- Enhance response safety and consistency with post-filtering methods

## 📝 Citation

> Kim, Hwangmin. “Disaster-Aware LLM Optimization with Instruction Tuning and Prompt Engineering.” Graduation Thesis, Yonsei University, 2025.

---

## 📂 Repository Structure
