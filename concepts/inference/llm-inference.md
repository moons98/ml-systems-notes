---
title: LLM Inference — Prefill & Decode
description: LLM 추론의 두 phase. Prefill은 입력 전체를 한 번에 처리하는 compute-bound, Decode는 토큰을 하나씩 생성하는 memory-bound. 두 phase의 병목이 달라서 최적화 전략도 달라야 함.
tags: [llm, inference, prefill, decode, kv-cache, memory-bound, compute-bound]
type: concept
related_papers:
  - sosp25-heteroinfer
---

# LLM Inference — Prefill & Decode

## TL;DR

- LLM 추론은 **Prefill**과 **Decode** 두 phase로 나뉨.
- **Prefill**: 
  - 입력 토큰 전체를 한 번에 처리 (GEMM → compute-bound). 
  - 첫 토큰까지의 시간(TTFT)을 결정.
- **Decode**: 
  - 토큰을 하나씩 auto-regressive하게 생성 (GEMV → memory-bound). 
  - 토큰당 생성 속도(TPOT)를 결정.
- 두 phase의 병목이 다르기 때문에 **같은 최적화가 둘 다에 효과적일 수 없음**.

---

## 1. Autoregressive 생성 전체 흐름

```
입력 토큰 [t1, t2, ..., tN]
        │
        ▼
  ┌─────────────┐
  │   Prefill   │  ← 입력 전체를 한 번에 처리, KV cache 생성
  └──────┬──────┘
         │ 첫 번째 출력 토큰 (TTFT)
         ▼
  ┌─────────────┐
  │   Decode    │  ← 새 토큰 1개 생성, KV cache에 추가
  │   (반복)    │  ← 종료 조건(EOS 또는 max_len)까지 반복
  └─────────────┘
         │ 출력 토큰 시퀀스
```

각 decode step은 이전 step의 출력 토큰을 입력으로 받음 — 이전 결과에 의존하므로 step 간 병렬화 불가.

---

## 2. Prefill

### 동작

입력 토큰 N개를 **한 번에** Transformer에 통과시켜 첫 출력 토큰을 생성.

각 레이어에서:
```
Q = X_input @ W_q   [N, d_model] × [d_model, d_head] → [N, d_head]   ← GEMM
K = X_input @ W_k
V = X_input @ W_v

Attention(Q, K, V) = softmax(QKᵀ / √d) @ V
```

- N개 토큰이 행렬의 행으로 묶여 한 번에 weight를 통과 → **같은 weight를 N개 토큰이 나눠 부담** (GEMM)
- 모든 N개 토큰의 K, V를 계산해 **KV cache에 저장**

### Output logit과 KV cache

각 위치마다 "다음 토큰 예측" logit이 나오지만, 실제로 쓰는 건 **마지막 위치(tN)의 logit뿐**. (t1~tN-1의 output logit은 버림)

중간 계산이 낭비가 아닌 이유: 
- **모든 위치의 K, V가 KV cache로 저장**되기 때문. 
- 각 레이어를 통과하면서 N개 토큰의 hidden state가 유지되어야 다음 레이어에서도 N개의 K, V를 뽑을 수 있음. 
- 이 KV cache가 decode에서 재사용됨.

### Causal Masking

N개 토큰을 동시에 처리하면 t3가 t4, t5를 미리 볼 수 있게 됨. Decode 때는 t4, t5가 아직 존재하지 않으므로, 이 정보가 섞이면 학습·추론 간 불일치가 생김.

이를 막기 위해 attention logit(`QKᵀ / √d`) 행렬에서 **미래 위치를 -∞로 마스킹** (softmax 후 0이 됨):

```
         t1    t2    t3    t4    t5
t1   [  att    -∞    -∞    -∞    -∞ ]
t2   [  att   att    -∞    -∞    -∞ ]
t3   [  att   att   att    -∞    -∞ ]  ← t3는 t1, t2, t3만 볼 수 있음
t4   [  att   att   att   att    -∞ ]
t5   [  att   att   att   att   att ]
```

각 토큰은 자신 이전 토큰들의 맥락만 반영 → decode와 동일한 조건.

### 성능 특성

- **Compute-bound**: N이 크면 FLOP 수가 지배적. 연산 유닛 활용률이 병목.
- Arithmetic intensity = `2 × N × d_model² / (weight load)` — N에 비례해 올라감
- N이 작으면 (짧은 입력) memory-bound로 전환될 수 있음

### 지표: TTFT (Time To First Token)

사용자가 요청을 보내고 첫 토큰을 받기까지의 시간. Prefill latency가 직접 결정.

---

## 3. Decode

### 동작

이전 출력 토큰이 다시 입력으로 들어오는 **auto-regressive** 구조. 매 step 새 토큰 1개를 생성.

각 레이어에서:
```
q = x_new @ W_q   [1, d_model] × [d_model, d_head] → [1, d_head]   ← GEMV (새 토큰만)
k = x_new @ W_k   ← 새 토큰의 k 계산 후 KV cache에 append
v = x_new @ W_v   ← 새 토큰의 v 계산 후 KV cache에 append

K = [KV_cache_K; k]   ← 과거 + 새 토큰
V = [KV_cache_V; v]

Attention(q, K, V) = softmax(q @ Kᵀ / √d) @ V
```

- 새로 구해야 하는 Q, K, V가 1 토큰분 → **weight 로드 비용을 혼자 부담** (GEMV)
- 새 토큰의 k, v를 KV cache에 **append** — 다음 step을 위해 저장
- 과거 토큰의 K, V는 cache에서 load해 재사용

### 성능 특성

- **Memory-bound**: weight를 매 step HBM에서 읽는 비용이 지배적. BW가 병목.
- Arithmetic intensity ≈ `2 × d_model / bytes_per_weight` — N=1이라 극도로 낮음
  - 예: Llama-8B FP16 → ~1~2 FLOP/Byte (A100 ridge point ~250 FLOP/Byte)
- Compute unit은 대부분 idle, HBM transfer를 기다림

### 지표: TPOT (Time Per Output Token)

토큰 1개를 생성하는 데 걸리는 시간. Decode throughput이 직접 결정.

---

## 4. KV Cache

### 왜 필요한가

Decode step마다 새 토큰 1개가 추가되지만, **과거 토큰들의 K, V는 변하지 않음**. 매번 다시 계산하면:
- 과거 토큰 N개 × 매 step → 중복 계산이 step마다 선형 증가
- Step i에서 총 FLOP = `i × (전체 레이어 연산량)` → O(N²) 누적

KV cache는 이미 계산한 K, V를 메모리에 저장해두고 재사용 → decode당 FLOP을 O(1)로 유지.

### 메모리 비용

```
KV cache 크기 = 2 × L × H × d_head × S × precision
  L: 레이어 수
  H: attention head 수
  d_head: head 차원
  S: 시퀀스 길이 (generate 길어질수록 증가)
  precision: FP16 = 2 byte
```

예: Llama-8B (L=32, H=32, d_head=128, S=4096, FP16)
```
= 2 × 32 × 32 × 128 × 4096 × 2 byte ≈ 2 GB
```

- 시퀀스가 길어질수록 선형 증가 → 동시 요청(batch) 늘리기 어려움
- 모바일 환경에서는 제한된 메모리가 KV cache 크기의 hard limit

### Prefill vs Decode 정리

| | Prefill | Decode |
|---|---|---|
| 입력 크기 | N 토큰 (행렬) | 1 토큰 (벡터) |
| 연산 형태 | GEMM (행렬 × 행렬) | GEMV (행렬 × 벡터) |
| Weight 로드 | N 토큰이 나눠 부담 | 1 토큰이 전부 부담 |
| KV cache | 생성 (N개 위치) | append (1개) + load (과거 전체) |
| Arithmetic intensity | 높음 (N에 비례) | 극도로 낮음 (~1~2 FLOP/Byte) |
| 병목 | Compute | Memory BW |

---

## 5. 두 Phase가 서로 다른 최적화를 요구하는 이유

- **Prefill 최적화** → compute unit 활용률 극대화
  - FlashAttention (attention 연산의 memory traffic 감소)
  - Chunked prefill (긴 입력을 chunk로 나눠 decode와 interleave)
  - NPU 활용 (compute-bound에서 W당 TOPS 높음)

- **Decode 최적화** → memory BW 극대화 / weight 로드 감소
  - Quantization (weight INT4 → BW 절반, W4A16)
  - Continuous batching (여러 요청의 decode를 묶어 arithmetic intensity ↑)
  - Speculative decoding (draft model로 여러 토큰 후보 생성 후 한 번에 검증)
  - GPU+NPU 병렬 (양쪽 메모리 채널 동시 활용 → BW ↑)
