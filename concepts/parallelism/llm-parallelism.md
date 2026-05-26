---
title: LLM Parallelism
description: 대규모 모델 학습·추론에서 GPU 여러 장을 활용하는 분산 병렬화 전략. Data, Tensor, Pipeline, Sequence, Expert Parallelism과 이들의 조합(3D parallelism, ZeRO)을 다룸.
tags: [parallelism, distributed, tensor-parallel, pipeline-parallel, data-parallel, megatron, deepspeed, moe]
type: concept
related: [llm-inference, kv-cache]
---

# LLM Parallelism

## TL;DR

- 모델이 GPU 한 장에 안 올라가거나, 올라가도 처리량이 부족할 때 분산 병렬화가 필요.
- **Data Parallelism (DP)**: 배치를 나눠 각 GPU에서 같은 모델을 실행 → gradient 동기화. 모델이 한 장에 들어올 때만 가능.
- **Tensor Parallelism (TP)**: 레이어 내부의 weight를 GPU 간에 쪼갬 → 한 레이어를 여러 GPU가 협력해 계산. 통신이 레이어마다 발생.
- **Pipeline Parallelism (PP)**: 레이어 묶음을 GPU 단계별로 배치 → micro-batch로 pipeline을 채워 bubble을 줄임. 통신은 스테이지 경계에서만.
- **Sequence Parallelism (SP)**: TP와 결합해 activation(LayerNorm, Dropout 등)도 시퀀스 차원으로 분산.
- **Expert Parallelism (EP)**: MoE에서 각 expert를 별도 GPU에 배치 → all-to-all 통신.
- **조합**: Megatron-LM의 3D parallelism(DP × TP × PP), DeepSpeed ZeRO가 실전에서 주로 쓰임.

---

## 1. Data Parallelism (DP)

### 기본 동작

동일한 모델 복사본을 모든 GPU에 두고, 배치를 나눠 각자 forward/backward 후 gradient를 평균내어 동기화:

```
GPU 0: model copy | batch [0:N/4]   → grad_0
GPU 1: model copy | batch [N/4:N/2] → grad_1
GPU 2: model copy | batch [N/2:3N/4]→ grad_2
GPU 3: model copy | batch [3N/4:N]  → grad_3
         ↓ All-Reduce (평균)
         모든 GPU가 동일한 grad로 weight 업데이트
```

- 구현이 가장 단순 (PyTorch DDP)
- 모델이 GPU 한 장에 올라와야 함 → 큰 모델에는 단독으로 쓸 수 없음
- All-Reduce 통신량: 모델 파라미터 크기 × 2 (reduce + broadcast)

### 한계

70B 모델 FP16: 파라미터만 140 GB → A100 80 GB 한 장에 불가.  
optimizer state(Adam: parameter × 3) 포함 시 560 GB → DP만으로는 불가능.

---

## 2. Tensor Parallelism (TP)

### 핵심 아이디어 (Megatron-LM)

단일 레이어의 weight 행렬을 GPU 간에 분할. Transformer의 두 핵심 연산에 적용:

#### Column Parallel (MLP의 첫 번째 Linear)

Weight를 **열(column)** 방향으로 쪼갬:

```
W: [hidden, 4×hidden]  →  GPU 0: W[:, :2h]  |  GPU 1: W[:, 2h:]

입력 X: [seq, hidden] (각 GPU에 동일하게 broadcast)

GPU 0: X @ W[:, :2h] = Y_0  [seq, 2h]
GPU 1: X @ W[:, 2h:] = Y_1  [seq, 2h]

Y = concat([Y_0, Y_1], dim=-1)  [seq, 4h]
```

입력은 공유, 출력은 분산됨 → All-Gather 없이 다음 Row Parallel로 이어짐.

#### Row Parallel (MLP의 두 번째 Linear)

Weight를 **행(row)** 방향으로 쪼갬:

```
W: [4×hidden, hidden]  →  GPU 0: W[:2h, :]  |  GPU 1: W[2h:, :]

GPU 0: Y_0 @ W[:2h, :] = Z_0  [seq, hidden]
GPU 1: Y_1 @ W[2h:, :] = Z_1  [seq, hidden]

Z = Z_0 + Z_1  ← All-Reduce (합산)
```

Column → Row를 이어 붙이면 레이어 하나를 거칠 때 **All-Reduce 1회**만 필요.

#### Attention에서의 TP

Multi-Head Attention은 head를 GPU 간에 나눔:

```
총 32 heads, GPU 4장 → 각 GPU가 8 heads 담당

GPU 0: Q_0, K_0, V_0 (head 0~7)
GPU 1: Q_1, K_1, V_1 (head 8~15)
...

각 GPU에서 self-attention 독립 계산 → 출력 합산 (All-Reduce)
```

GQA에서도 KV head 수 ≥ TP 차수이면 동일하게 분할 가능.

### 통신 패턴과 비용

| 위치 | 통신 종류 | 크기 |
|---|---|---|
| Column Parallel 입력 | (없음, 이미 분산) | — |
| Row Parallel 출력 | All-Reduce | activation 크기 |
| 레이어당 총 | All-Reduce × 2 (MLP + Attn) | 2 × B × S × H |

TP는 **같은 서버 내(NVLink)**에서 써야 효율적. 노드 간 PCIe나 InfiniBand로 All-Reduce를 매 레이어마다 하면 통신이 bottleneck.

---

## 3. Pipeline Parallelism (PP)

### 기본 구조

레이어 묶음을 GPU 단계(stage)에 배치:

```
GPU 0 (stage 0): Layer 0~7
GPU 1 (stage 1): Layer 8~15
GPU 2 (stage 2): Layer 16~23
GPU 3 (stage 3): Layer 24~31

데이터 흐름: GPU0 → GPU1 → GPU2 → GPU3 (forward)
             GPU3 → GPU2 → GPU1 → GPU0 (backward)
```

스테이지 간 통신: activation만 전달 → 크기가 weight보다 훨씬 작음 (`B × S × H`).

### Pipeline Bubble 문제

단순 구현에서는 한 번에 배치 하나만 넣으면 stage 간 idle time이 생김:

```
        t0   t1   t2   t3   t4   t5   t6
GPU 0: [F0] [   ] [   ] [B0] [   ] [   ]  ← 대부분 idle
GPU 1:      [F0] [   ] [   ] [B0] [   ]
GPU 2:           [F0] [   ] [   ] [B0]
GPU 3:                [F0] [   ] [B0]

Bubble ratio = (p-1)/(m+p-1)   p: 스테이지 수, m: micro-batch 수
```

### Micro-batch로 Bubble 줄이기

배치를 micro-batch로 쪼개 pipeline을 채움 (GPipe, Megatron의 1F1B):

```
Micro-batch 4개 (m=4), 스테이지 4개 (p=4):

        t0  t1  t2  t3  t4  t5  t6  t7  t8  t9
GPU 0: [F1][F2][F3][F4][   ][B4][B3][B2][B1]
GPU 1:     [F1][F2][F3][F4][B4][B3][B2][B1]
GPU 2:         [F1][F2][F3][F4][B3][B2][B1]
GPU 3:             [F1][F2][F3][F4][B2][B1]

Bubble ratio = (p-1)/(m+p-1) = 3/7 ≈ 43%  (m 늘릴수록 bubble 줄어듦)
```

**1F1B (One-Forward-One-Backward)**: forward 한 번 하면 곧바로 backward 시작 → activation 메모리를 p 배가 아닌 p 배로만 유지.

### PP의 trade-off

- 통신량 적음 (스테이지 경계에서만)
- 레이어를 균등하게 나눠야 load balance 맞음
- Bubble이 남으므로 m (micro-batch 수)이 p (stage 수)보다 충분히 커야 효율적
- TP처럼 매 레이어 통신이 없어서 노드 간(InfiniBand)에서도 사용 가능

---

## 4. Sequence Parallelism (SP)

### 문제: TP가 못 쪼개는 부분

TP로 Attention/MLP는 나눴지만, 그 사이의 LayerNorm과 Dropout은 여전히 한 GPU에서 전체 시퀀스를 처리:

```
LayerNorm([B, S, H])  → 시퀀스 전체 필요, GPU 한 장이 담당
Dropout([B, S, H])    → 동일
```

시퀀스가 길어질수록 이 activation이 bottleneck.

### SP의 해결 방식

시퀀스 차원(S)으로 activation을 나눔. TP와 SP를 교대로 적용:

```
입력: [B, S, H] → 시퀀스를 TP_degree로 분할

GPU 0: [B, S/2, H]   (token 0~S/2)
GPU 1: [B, S/2, H]   (token S/2~S)

LayerNorm, Dropout을 각 GPU에서 독립 수행

Attention 직전: All-Gather로 전체 시퀀스 복원 → TP Column Parallel
Attention/MLP 완료 후: Reduce-Scatter로 다시 분산 → 다음 LayerNorm
```

통신이 All-Reduce → All-Gather + Reduce-Scatter로 분리되지만 **총량은 동일**, 대신 activation 메모리가 1/TP_degree로 줄어듦.

---

## 5. Expert Parallelism (EP) — MoE

### MoE 구조 복습

MoE(Mixture of Experts)는 MLP를 여러 expert로 교체하고 router가 각 토큰을 top-k expert에게 배분:

```
token x → Router → expert_id {2, 7}
         → Expert 2의 FFN → 출력 y_2
         → Expert 7의 FFN → 출력 y_7
         → y = y_2 × gate_2 + y_7 × gate_7
```

expert 수가 많으면 (e.g., 64개) 한 GPU에 다 올리기 어려움.

### EP 배치

expert를 GPU 간에 나눠 배치:

```
총 8 experts, GPU 4장:
  GPU 0: Expert 0, 1
  GPU 1: Expert 2, 3
  GPU 2: Expert 4, 5
  GPU 3: Expert 6, 7
```

각 토큰이 배정된 expert가 있는 GPU로 이동해야 함 → **All-to-All 통신**.

```
All-to-All:
  GPU 0이 가진 token 중 Expert 2로 가야 하는 것 → GPU 1로 전송
  GPU 1이 가진 token 중 Expert 0으로 가야 하는 것 → GPU 0으로 전송
  ...
  동시에 모든 GPU 간 교환 (N² 쌍)
```

All-Reduce와 달리 All-to-All은 GPU마다 주고받는 양이 routing 결과에 따라 달라져 **load imbalance** 발생 가능. Router loss(auxiliary loss)로 균형 유도.

---

## 6. 조합 전략

### Megatron-LM 3D Parallelism

DP, TP, PP를 동시에 적용:

```
총 64 GPU = DP(4) × TP(4) × PP(4)

  TP group (4 GPU, 같은 노드 NVLink):   레이어 내부 weight 분산
  PP group (4 GPU, 노드 간 IB):          레이어 묶음을 스테이지로
  DP group (4 GPU):                       배치를 나눠 복제본 실행
```

배치 → 4개 DP 복제본 → 각 복제본 내에서 PP로 스테이지 분리 → 각 스테이지 내에서 TP로 레이어 분산.

통신 위치별 요구 BW:
- TP: NVLink (서버 내, ~600 GB/s)
- PP: InfiniBand (서버 간, ~400 Gb/s)
- DP: InfiniBand (All-Reduce, 주기 짧음)

### DeepSpeed ZeRO

DP의 메모리 낭비(모든 GPU가 모델 전체 보유)를 해결:

```
ZeRO-1: optimizer state를 GPU 간 분산 (메모리 ~4× 절약)
ZeRO-2: gradient도 분산           (메모리 ~8× 절약)
ZeRO-3: parameter도 분산          (메모리 모델 크기/GPU 수)

ZeRO-3:
  GPU 0: param[0:N/4], grad[0:N/4], optim[0:N/4]
  GPU 1: param[N/4:N/2], ...
  ...
  forward 시 필요한 param은 All-Gather로 임시 복원 후 사용, 이후 즉시 해제
```

TP처럼 레이어를 수술하지 않아도 되어 코드 변경이 적음. 단, All-Gather가 매 레이어마다 발생 → TP보다 통신 overhead 클 수 있음.

---

## 7. 비교 요약

| 전략 | 분산 단위 | 주 통신 | 통신 위치 | 주요 제약 |
|---|---|---|---|---|
| **DP** | 배치 | All-Reduce (grad) | 노드 간 | 모델이 GPU 한 장에 들어와야 |
| **TP** | 레이어 내 weight | All-Reduce (매 레이어) | 노드 내 (NVLink) | 서버 내 GPU 수 제한 |
| **PP** | 레이어 묶음 | P2P activation | 노드 간 (IB) | Bubble, load balance |
| **SP** | 시퀀스 차원 activation | All-Gather + Reduce-Scatter | TP와 동일 | TP와 결합 필수 |
| **EP** | Expert | All-to-All | 노드 간 | Load imbalance, MoE 전용 |
| **ZeRO** | 파라미터·상태 | All-Gather (매 레이어) | 노드 간 | TP보다 통신 많을 수 있음 |

### 메모리·처리량 trade-off 한 줄 요약

- DP: 처리량 ↑, 메모리 절약 없음
- TP: 메모리 ↓ (weight), 통신 overhead (NVLink 필요)
- PP: 메모리 ↓ (activation 분산), Bubble overhead
- ZeRO-3: 메모리 ↓↓, 통신 overhead (TP 없이도 작동)
- 3D: 세 가지 모두 조합, 하드웨어 토폴로지에 맞게 설계 필요
