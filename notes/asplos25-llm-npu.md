---
title: "Fast On-device LLM Inference with NPUs (llm.npu)"
venue: ASPLOS 2025
authors: Daliang Xu, Hao Zhang, Liming Yang, Ruiqi Liu, Gang Huang, Mengwei Xu, Xuanzhe Liu
affiliations: Peking University, Beijing University of Posts and Telecommunications
tags: [llm-inference, mobile, npu, edge-ai, quantization, prefill, on-device, chunking]
read_date: 2026-06-28
status: read
url: https://doi.org/10.1145/3669940.3707239
pdf: ../pdfs/asplos25-llm-npu.pdf
---

## TL;DR

- **문제**:
  - 모바일 LLM에서 prefill이 전체 지연의 **88.3–98.8%** 차지 (UI 자동화 기준)
  - NPU는 INT8 MatMul에 최적화되어 있지만 세 가지 근본적 갭이 존재:
    - NPU는 정적 그래프만 지원 → 동적 길이 프롬프트마다 그래프 재구축 필요 (Gemma-2B 기준 11초+)
    - Per-group 양자화(AWQ 등)는 NPU에서 그룹 크기 단위 서브텐서 MatMul로 분할 필요 → 최대 10.7× 성능 저하
    - LayerNorm·Attention 등 FP 연산은 NPU에서 INT8 대비 **193–759× 느림**
- **접근**:
  - 3단계 알고리즘-시스템 공동 설계
  - Prompt-level: 프롬프트를 고정 크기 청크로 분할, 정적 서브그래프 공유로 그래프 구축 오버헤드 제거
  - Tensor-level: 이상치(0.1–0.3% 채널)만 CPU에서 병렬 처리, 나머지는 NPU INT8로 실행
  - Block-level: NPU 버블을 줄이기 위한 Out-of-order 서브그래프 스케줄링
- **결과**:
  - Prefill: llama.cpp 대비 **18–43×**, PowerInfer-v2 대비 **3.28–5.32×**
  - 에너지: llama.cpp/MLC-LLM 대비 **35–59×**, GPU(TFLite) 대비 **1.85–4.32×**
  - 역사적 첫 **1,000+ tok/s** prefill 달성 (Qwen1.5-1.8B, 10억 규모 모델)
  - 정확도 손실 **<1%** (FP16 대비)

---

## Background

- 기존 모바일 LLM 연구는 decoding 가속에 집중 → prefill 병목 미해결
- NPU는 모바일 SoC에서 INT8 MatMul에 특화 (73 TOPS)이나 FP 연산 지원이 극히 취약
- 기존 NPU 추론 시스템(PowerInfer-v2)이 있으나 정적 그래프 제약·per-group 양자화·FP 연산 문제를 모두 해결하지 못함
- 이 논문의 기여:
  - 세 가지 NPU 갭을 체계적으로 분석·정의
  - 각 갭을 해결하는 3단계 최적화를 하나의 시스템(llm.npu)으로 통합
  - 10억 규모 모델에서 처음으로 1,000+ tok/s prefill 달성

---

## Key Insights

1. **NPU의 정적 그래프 제약은 프롬프트를 청크로 쪼개면 우회할 수 있다.**
   - Decoder-only 구조에서 i번째 토큰의 결과는 이전 토큰에만 의존
   - 따라서 프롬프트를 256토큰 청크로 순차 처리해도 결과 동일
   - 청크 내에서 Linear/LayerNorm 등 정적 연산자는 서브그래프 공유 가능 → 메모리 75% 절감

2. **이상치 채널(0.1–0.3%)만 별도 처리하면 per-group 양자화 없이 정확도를 유지할 수 있다.**
   - `x = [clip(x/s)] + [outlier_part]` 로 분리
   - INT8 부분 → NPU, 이상치 부분 → CPU FP32 병렬 실행
   - 상위 3% 채널이 80% 이상치를 생성하므로 해당 채널 가중치만 메모리에 상주
   - 불필요한 85% 레이어의 이상치는 오프라인 프로파일링으로 제거 → 동기화 비용 29.7% 절감

3. **NPU가 병목이므로 CPU/GPU 작업을 NPU 실행 중에 숨기는 Out-of-order 스케줄링이 효과적이다.**
   - 순차 실행 시 NPU 버블율 37%
   - 각 서브그래프의 기여도 C를 계산해 항상 가장 유리한 것부터 실행
   - 마이크로초 단위 온라인 스케줄링으로 18–44% 추가 지연 감소

4. **FP 연산(LayerNorm, Attention)은 반드시 CPU/GPU에 배치해야 한다.**
   - NPU FP16 MatMul은 INT8 대비 193–759× 느림
   - 하드웨어 친화성에 따라 레이어를 정적으로 NPU/CPU 배분

---

### §3 NPU 갭 분석

| 갭 | 현상 | 수치 |
|---|---|---|
| **Gap-1: 동적 길이** | 길이마다 그래프 재구축 필요 | Gemma-2B: 11초+ 오버헤드 |
| **Gap-2: 양자화 불일치** | Per-group 양자화 → 서브텐서 MatMul 필요 | 최대 10.7× 성능 저하 |
| **Gap-3: FP 연산** | NPU는 INT8 특화, FP16은 극도로 느림 | FP16 MatMul이 INT8 대비 193–759× |

---

### §4 Design

**원칙**:
- NPU가 임계 경로 → NPU 활용도 극대화
- CPU/GPU는 보조 역할 (이상치 처리, FP 연산, NPU 버블 채우기)

#### §4.1 Chunk-Sharing Graph 실행 (Prompt-level)

- 프롬프트를 256토큰 청크로 분할하여 순차 처리
- 동일 크기 청크들은 동일 서브그래프 구조를 공유
  - **정적 연산자** (Linear, LayerNorm): 청크 간 서브그래프 재사용
  - **동적 연산자** (Attention): 청크마다 KV 차원이 달라 개별 서브그래프 필요
- Qwen1.5-1.8B: 144개 서브그래프 중 120개 공유, 메모리 **75% 절감** (7.2GB @ 1024토큰)
- 온라인 그래프 구축 지연 제거

#### §4.2 Shadow Outlier 실행 (Tensor-level)

**분해 공식**:
```
X·W ≈ [INT8(X)] · W  (NPU)
     + [outlier(X)] · W_hot  (CPU, 병렬)
```

- 이상치 판별: per-tensor 스케일 s로 나눴을 때 클리핑 범위를 벗어나는 성분
- `W_hot`: 상위 3% "핫 채널" 가중치만 상주, 나머지는 디스크
- 오프라인 프로파일링으로 불필요 레이어(85%)의 이상치 처리 스킵 → CPU-NPU 동기화 제거
- SmoothQuant 대비 최대 **32.9% 정확도 향상**

#### §4.3 Out-of-order 서브그래프 스케줄링 (Block-level)

**의존성 구조**:
- 청크 내: G(i,j) ← G(i,j-1) (이전 레이어)
- 청크 간: G(i,j) ← G(0..i, j-1) (모든 이전 청크의 j-1번 레이어)

**스케줄링 기준**:
- CPU/GPU 서브그래프 g의 기여도 C = 이후 NPU 서브그래프들의 총 대기 시간
- NPU 서브그래프 g의 기여도 C = 이후 CPU 서브그래프들의 총 대기 시간의 음수
- 매 단계 최대 C인 서브그래프 실행 → NPU 버블 최소화

---

### §5 Evaluation

**실험 환경**:
- Redmi K70 Pro (Snapdragon 8 Gen3, Hexagon NPU, 24GB)
- Redmi K60 Pro (Snapdragon 8 Gen2, 16GB)
- 모델: Qwen1.5-1.8B, Gemma-2B, Phi-2-2.7B, LLaMA-2-7B, Mistral-7B

**Prefill 속도 (프롬프트 1024토큰, K70 Pro)**

| 비교 대상 | 가속 배율 |
|---|---|
| llama.cpp (CPU) | **18.17–38.4×** |
| MLC-LLM (GPU) | **32.5–43.6×** |
| TFLite (GPU) | **1.27–2.34×** |
| PowerInfer-v2 (NPU) | **3.28–5.32×** |

- Qwen1.5-1.8B: **~1,100 tok/s** (역사적 첫 >1,000 tok/s)

**에너지 (K60 Pro, 1024토큰)**

| 비교 대상 | 절감 배율 |
|---|---|
| llama.cpp (CPU) | **35.63–59.52×** |
| MLC-LLM (GPU) | **35.21–59.25×** |
| TFLite (GPU) | **1.85–4.32×** |

**End-to-End (LongBench, 1024토큰 입력)**
- Qwen1.5-1.8B: 전체 1.7초 (prefill 1.49s), llama.cpp 26.7초 → **6.2–26.8×**
- Gemma-2B: 전체 1.9초, llama.cpp 34.6초 → **최대 41.3×**

**정확도 (FP16 대비 평균 손실 <1%)**
- SmoothQuant: 5.1–14.9% 손실, K-Quant: 31.3% 손실과 대비

**Ablation (Qwen1.5-1.8B, 프롬프트 512토큰)**

| 단계 | CPU 대비 성능 |
|---|---|
| 기본 NPU (최적화 없음) | **2.55–2.68× 느림** |
| + Chunk-sharing | **1.46–5.09× 빠름** |
| + Shadow outlier | **추가 3.91–8.68×** |
| + Out-of-order | **추가 18–44%** |
| **llm.npu (전체)** | **18–43× 빠름** |
