---
title: "Hybe: GPU-NPU Hybrid System for Efficient LLM Inference with Million-Token Context Window"
venue: ISCA 2025
authors: Seungjae Moon, Junseo Cha, Hyunjun Park, Joo-Young Kim
affiliations: HyperAccel, Seoul, Republic of Korea
tags: [llm-inference, heterogeneous-compute, gpu, npu, kv-cache, long-context, decode, prefill-decode-disaggregation]
read_date: 2026-05-26
status: read
url: https://dl.acm.org/doi/10.1145/3695053.3731051
pdf: ../pdfs/isca25-hybe.pdf
---

## TL;DR

- **문제**: Context window가 100K 토큰을 넘으면 KV cache가 GPU 메모리를 지배 → batching 불가 → GPU decode 단계에서 underutilization 심화.
- **접근**: Prefill은 기존 GPU(H100 + vLLM), Decode는 경량 NPU로 분리. NPU는 compute가 아닌 **memory bandwidth를 최대한 활용**하도록 설계. Fine-grained KV transmission으로 GPU→NPU KV offload를 prefill 도중에 즉시 시작, GPU 메모리 요구량 감소. Stage-wise pipelining 스케줄러로 idle HW 최소화.
- **결과**: H100 GPU 동일 대수 대비 throughput 향상 + Llama-3 energy efficiency 개선.

---

## Background

### LLM 추론의 두 단계 비대칭

```
Prefill:  input 전체를 한 번에 처리 → compute-intensive
           → GPU의 대규모 코어가 효율적

Decode:   토큰 하나씩 생성 (autoregressive) → memory-intensive
           → 매 step마다 KV cache 전체를 DRAM에서 로드
           → GPU 코어 대부분 idle (memory-bound)
```

### Long Context에서 문제가 심화되는 이유

기존 해법: batching — 여러 요청을 동시에 처리해 GPU 활용률 보완.

100K+ 토큰에서 batching 불가능해지는 이유:
```
KV cache per request = 2 × L × H × d × S × 2 byte

S=100K, Llama-3-70B:
  레이어 80개, head 8개 (GQA), d=128, FP16
  → 토큰당 ~320KB → 100K 토큰 = ~32 GB / request

H100 HBM = 80 GB → 요청 2개도 불가 → batch size = 1
```

Batch = 1이면 GPU 코어 활용률은 decode 내내 1% 미만.

### 기존 hybrid 시스템의 한계

- GPU-only: decode 단계 underutilization
- 기존 hybrid (CPU offloading 등): KV 전송 latency가 bottleneck, 동기화 overhead

---

## Key Insights

1. **Decode 전용 NPU는 compute가 아니라 메모리 대역폭에 최적화해야 한다.**
   - Decode matmul은 matrix-vector (batch=1) → compute-bound 아님
   - 병목 = KV cache를 HBM에서 읽는 속도
   - 따라서 NPU에 많은 compute core 불필요 → 같은 면적에 HBM + 필요한 최소 연산 유닛만

2. **KV를 prefill이 끝난 뒤 전송하면 이미 늦다.**
   - Prefill이 길면 (100K 토큰) KV가 GPU HBM을 채우는 시점에 이미 OOM 위험
   - Fine-grained transmission: prefill 도중 layer별로 KV가 생성될 때마다 즉시 NPU로 offload
   - GPU HBM에 KV가 쌓이지 않음 → GPU가 더 많은 요청의 prefill 처리 가능

3. **Stage-wise pipelining으로 GPU-NPU idle 최소화.**
   - GPU (prefill)와 NPU (decode)가 각자 큐를 처리
   - 한쪽이 비면 스케줄러가 dynamically 재배정

---

## 아키텍처

### 전체 구조

```
GPU (H100)                    Hybe NPU (×N)
  vLLM 기반 prefill            Decode 전용
  ────────────────             ────────────
  input 처리                   HBM (GPU와 동일 spec)
  KV 생성 → 즉시 전송 →→→→→→ KV cache 보관
                               attention + FFN (경량)
                               output token 생성
```

### NPU 설계 원칙 (§5)

- **Hardware-aware memory mapping**: HBM 용량과 bandwidth를 decode workload에 맞게 배분
- **Model parallelism**: 여러 NPU에 모델 분산 (tensor parallelism)
- **LLM-specific hardware design**: softmax, RoPE 등 LLM 전용 연산 유닛 포함

### Fine-grained KV Transmission (§6.1)

```
기존 방식:
  prefill 전체 완료 → KV 전부 GPU HBM에 쌓임 → 한 번에 NPU로 전송
  → GPU HBM이 일시적으로 꽉 참

Hybe:
  prefill layer N 완료 → KV_N 즉시 NPU로 전송 (streaming)
  → GPU HBM의 KV는 생성 직후 해제
  → GPU HBM에 항상 여유 공간 → 더 많은 prefill 처리 가능
```

### Stage-wise Pipelining (§6.2)

```
GPU 큐:  [prefill A] [prefill B] [prefill C] ...
NPU 큐:  [decode A]  [decode B]  [decode C]  ...

스케줄러:
  - GPU가 prefill 완료 → NPU 큐에 decode 추가
  - NPU idle → 대기 중 요청 우선 배정
  - 두 하드웨어가 동시에 유휴 상태가 되지 않도록 동적 조정
```

---

## Evaluation (§8)

**실험 환경:**
- GPU: NVIDIA H100 (HBM 80GB)
- NPU: Hybe NPU (H100과 동일 HBM spec, 경량 compute)
- 비교 대상: 동일 대수의 H100 GPU-only

**결과:**

| 지표 | 결과 |
|---|---|
| Throughput | H100 GPU-only 대비 향상 (Hi-3, long context) |
| Energy efficiency | Llama-3 long context에서 GPU-only 대비 개선 |
| System efficiency | GPU HBM 활용률 향상, decode underutilization 해소 |

---

## HeteroInfer와 비교

| | HeteroInfer (SOSP 2025) | Hybe (ISCA 2025) |
|---|---|---|
| 하드웨어 | 모바일 SoC (Snapdragon) | 데이터센터 (H100 + 커스텀 NPU) |
| 분할 방식 | Tensor-level 동시 실행 | Stage-level (prefill=GPU, decode=NPU) |
| 핵심 문제 | 모바일 BW 미활용 | Long context decode underutilization |
| NPU 타입 | SoC 내장 NPU | 커스텀 설계 NPU (HBM 탑재) |
| KV 처리 | UMA 공유 | Fine-grained streaming 전송 |
