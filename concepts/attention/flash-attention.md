---
title: FlashAttention
description: IO-aware exact attention 구현. HBM 접근을 최소화하기 위해 attention 연산을 타일 단위로 쪼개 SRAM 안에서 처리. 학습/추론 모두에 이점이 있으나 원 motivation은 학습 시 메모리 절약.
tags: [attention, kernel, io-aware, flash-attention, gpu, sram, hbm]
type: concept
related: [kv-cache, llm-inference]
---

# FlashAttention

## TL;DR

- Standard attention은 N×N attention matrix를 HBM에 저장 → O(N²) 메모리, HBM traffic이 병목.
- FlashAttention은 **결과는 동일**하되, N×N matrix를 HBM에 쓰지 않고 타일 단위로 SRAM 안에서 처리 → HBM traffic 대폭 감소.
- 핵심 알고리즘: **Tiling + Online Softmax** — 전체 row를 보지 않고 타일을 하나씩 보면서 softmax를 incremental하게 계산.
- 메모리: O(N²) → O(N). 속도: prefill 2~4× 향상 (학습도 동일).
- **FlashAttention-2**: seq_len 차원 병렬화 추가, ~2× 추가 향상.
- **FlashDecoding**: decode 단계 특화. KV cache를 chunk로 쪼개 병렬 처리 후 reduce.

---

## 1. Standard Attention의 문제

### 연산 흐름

```
입력: Q, K, V  각각 [N, d]  (N: seq_len, d: head_dim)

① S = Q @ K^T          → [N, N]   HBM에 write
② P = softmax(S)        → [N, N]   HBM에 write
③ O = P @ V             → [N, d]   HBM에 write

총 HBM read/write: QKV load + S write/read + P write/read + O write
```

### 병목

GPU 메모리 계층:

```
SRAM (shared memory): ~20 MB / A100 SM 전체   ← 빠름 (~19 TB/s)
HBM:                  40~80 GB                 ← 느림 (~2 TB/s)
```

N×N matrix (N=4096, FP16):
```
4096 × 4096 × 2 byte = 32 MB per head per layer
→ SRAM에 올라가지 않음 → 반드시 HBM에 내려야 함
```

① S를 HBM에 쓰고, ② softmax를 위해 다시 HBM에서 읽는 왕복이 발생. N이 커질수록 HBM traffic이 O(N²)으로 증가 → **attention이 memory-bound**가 되는 원인.

---

## 2. 핵심 아이디어: Tiling + Online Softmax

### Tiling 구조

tile은 token(N) 방향만 쪼개고, d(head_dim)는 보존. score 계산(QK^T)이 d 전체 내적이라 d를 쪼개면 score 하나를 계산할 수 없음.

```
Q:  [N, d]  →  [B_q, d] tile  (N/B_q 개)
K:  [N, d]  →  [B_k, d] tile  (N/B_k 개)
V:  [N, d]  →  [B_k, d] tile  (N/B_k 개)
```

### 문제: Softmax는 전체 row를 봐야 한다

```
softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

분모를 계산하려면 전체 row의 값을 알아야 함 → KV tile을 하나씩 볼 때 전체 score를 알 수 없음.

### 해법: Online Softmax (incremental 계산)

KV tile을 하나씩 볼 때마다 running max와 running sum을 업데이트하고, 이전 결과를 rescale:

```
Q tile 하나 기준, KV tile을 하나씩 처리하는 과정:
  S_1, S_2: [B_q, B_k]               [B_q, d] @ [d, B_k]  ← Q_i @ K_j^T
  V_1, V_2: [B_k, d]
  m: [B_q], l: [B_q], O: [B_q, d]

KV 타일 1을 본 뒤:
  m_1 = max(S_1)              [B_q]    ← query token별 최댓값
  l_1 = sum(exp(S_1 - m_1))  [B_q]    ← query token별 exp 합
  O_1 = exp(S_1 - m_1) @ V_1 [B_q, d] ← 분모는 아직 나누지 않음
  m_1, l_1, O_1 → HBM write           ← 다음 KV tile을 위해 중간값 저장

KV 타일 2를 본 뒤:
  m_1, l_1, O_1 ← HBM read            ← 이전 중간값 로드
  m_2   = max(m_1, max(S_2))  [B_q]   ← 새 global max
  corr  = exp(m_1 - m_2)      [B_q]   ← max 변화 보정 스칼라

  l_2   = corr * l_1          [B_q]   ← 이전 합 rescale
         + sum(exp(S_2 - m_2))

  O_2   = corr * O_1          [B_q, d] ← 이전 output rescale
         + exp(S_2 - m_2) @ V_2

  m_2, l_2, O_2 → HBM write           ← 다음 KV tile을 위해 저장

모든 KV tile 처리 후:
  O, l ← HBM read
  최종 O = O / l              [B_q, d] ← 분모 정규화
  O → HBM write
```

수학적으로 standard softmax와 **정확히 동일한 결과**. N×N matrix를 HBM에 쓰지 않아도 됨.

### 전체 루프

```
for each K_j, V_j [B_k, d]:               HBM → SRAM, KV는 SRAM에 유지

  for each Q tile Q_i [B_q, d]:           HBM → SRAM
    m_i, l_i, O_i  ← HBM에서 로드        ← 이전 KV tile 처리 후 저장한 중간값
                                             (첫 번째 KV tile이면 -inf, 0, 0으로 초기화)

    S_ij = Q_i @ K_j^T  [B_q, B_k]       ← SRAM에서만 존재, HBM에 쓰지 않음

    m_new = max(m_i, max(S_ij))  [B_q]
    corr  = exp(m_i - m_new)     [B_q]

    l_i = corr * l_i + sum(exp(S_ij - m_new))  [B_q]
    O_i = corr * O_i + exp(S_ij - m_new) @ V_j [B_q, d]

    m_i = m_new
    m_i, l_i, O_i → HBM write             ← 다음 KV tile을 위해 중간값 저장

모든 KV tile 처리 후:
  for each Q tile Q_i:
    O_i, l_i  ← HBM에서 로드
    O_i = O_i / l_i                        [B_q, d]  ← 분모 정규화
    O_i → HBM write
```

HBM access:
```
Standard:       S [N,N] write → read, P [N,N] write → read  →  O(N²)
FlashAttention: Q/K/V read (각 1회) + O write (1회)          →  O(Nd)
```

---

## 3. Backward Pass: Recomputation

Forward에서 N×N matrix를 저장하지 않았으므로, backward에서 gradient 계산 시 attention matrix가 없음.

FlashAttention의 해법: **backward 시 attention을 재계산(recompute)**:
- Forward의 output O와 softmax normalization 통계(m, l)만 저장
- Backward에서 필요할 때 타일 단위로 attention을 다시 계산

메모리와 연산의 trade-off: 연산량은 늘지만 O(N²) activation 저장이 불필요 → 긴 context 학습에서 메모리가 결정적 병목이므로 이득.

---

## 4. FlashAttention-2

FA-1의 한계: warp 간 작업 분배가 비효율적, non-matmul 연산(rescale 등)이 많음.

FA-2 개선 (2023):

| 개선 | 내용 |
|---|---|
| **Loop order 역전** | outer=Q, inner=KV → m, l, O를 HBM에 중간 저장 불필요 |
| **Non-matmul 감소** | rescale 연산 최소화, Tensor Core 점유율 ↑ |
| **Warp 분배 개선** | Q tile을 warp에 독립 배분 → warp 간 동기화 제거 |

### FA-2 루프 구조

```
for each Q tile Q_i [B_q, d]:           HBM → SRAM

  초기화 (SRAM):
    m: [B_q] = -inf,  l: [B_q] = 0,  O: [B_q, d] = 0

  for each K_j, V_j [B_k, d]:           HBM → SRAM

    S_ij = Q_i @ K_j^T  [B_q, B_k]    ← SRAM에서만 존재

    m_new = max(m, max(S_ij))  [B_q]
    corr  = exp(m - m_new)     [B_q]

    l = corr * l + sum(exp(S_ij - m_new))  [B_q]
    O = corr * O + exp(S_ij - m_new) @ V_j [B_q, d]

    m = m_new

  O = O / l            [B_q, d]        ← 분모 정규화
  O → HBM write        [B_q, d]        ← Q tile당 HBM write 1회
                                           m, l, O 중간 저장 없음
```

FA-1 대비 차이: m, l, O가 SRAM에만 상주 → Q tile당 HBM write가 최종 1회뿐.

결과: FA-1 대비 ~2× 추가 향상. A100 기준 theoretical max BW의 ~70% 달성.

---

## 5. FlashDecoding

### Decode에서의 문제

Decode는 Q가 새 토큰 1개:
```
Q: [1, d]     ← 새 토큰 하나
K: [S, d]     ← KV cache 전체 (S = 현재 seq_len)
V: [S, d]
```

FA-1/2의 병렬화 축: batch size × num_heads.
Decode에서는 batch=1, head 수도 제한적 → **SM 대부분이 idle**.

S(KV cache 길이)가 길수록 문제 심각 (long context serving).

### FlashDecoding의 해법

KV cache를 chunk로 쪼개 **seq_len 방향으로 병렬화**:

```
KV cache [S, d] → chunk 0 [S/C, d], chunk 1 [S/C, d], ..., chunk C [S/C, d]

각 chunk를 별도 SM에서 병렬로 attention 계산:
  partial_O_0, partial_lse_0
  partial_O_1, partial_lse_1
  ...

Reduce: online softmax로 partial 결과들을 합산 → 최종 O
```

lse(log-sum-exp)를 저장해두면 reduce 시 정확한 rescale 가능.

결과: long context decode에서 SM 활용률 대폭 향상.

---

## 6. 학습 vs 추론에서의 역할

| | Prefill | Decode |
|---|---|---|
| Q shape | [N, d] (N 크다) | [1, d] |
| 병목 | HBM traffic (N×N matrix) | KV cache load BW |
| FlashAttention 효과 | 크다 — N×N 저장 제거 | 있지만 제한적 |
| FlashDecoding 효과 | 해당 없음 | 크다 — seq 차원 병렬화 |

FA-1의 원 motivation은 **학습**: O(N²) 메모리 → O(N)으로 줄여 더 긴 context 학습 가능. 학습/prefill에서 효과가 가장 큼.

Decode에서는 Q=1이라 N×N matrix 자체가 없어 FA의 핵심 이득이 작음. 이를 보완하기 위해 FlashDecoding이 등장.
