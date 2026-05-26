---
title: Collective Communication
description: 분산 시스템에서 여러 프로세스·GPU가 데이터를 주고받는 통신 패턴. Broadcast, Reduce, All-Reduce, All-Gather, Reduce-Scatter, All-to-All, P2P의 동작 원리, Ring All-Reduce 알고리즘, 하드웨어 토폴로지, 연산-통신 overlap을 다룸.
tags: [distributed, collective-communication, all-reduce, nccl, nvlink, infiniband]
type: concept
---

# Collective Communication

## TL;DR

- Collective communication은 N개의 GPU가 특정 패턴으로 데이터를 교환하는 연산. 각 primitive는 입력·출력·통신량이 다름.
- **All-Reduce**: 모든 GPU의 값을 합산(또는 평균)해 모든 GPU에 결과를 돌려줌. Gradient 동기화에 사용.
- **All-Gather**: 각 GPU가 가진 조각을 모아 모든 GPU가 전체를 갖게 함.
- **Reduce-Scatter**: 합산 후 결과를 조각으로 나눠 각 GPU에 배분. All-Gather와 역방향.
- **All-to-All**: 각 GPU가 각 GPU에게 서로 다른 데이터를 보냄. Expert routing에 사용.
- **Ring All-Reduce** = Reduce-Scatter + All-Gather. bandwidth-optimal 구현.
- NVLink(서버 내)와 InfiniBand(서버 간)의 대역폭 차이가 알고리즘 선택을 좌우함.

---

## 1. 기본 Primitives

표기: N개의 GPU, 각 GPU가 크기 M의 데이터를 가짐.

### Broadcast

GPU 0의 데이터를 나머지 모든 GPU에 복사:

```
Before:  GPU 0: [A]   GPU 1: [?]   GPU 2: [?]   GPU 3: [?]
After:   GPU 0: [A]   GPU 1: [A]   GPU 2: [A]   GPU 3: [A]

통신량: (N-1) × M
```

### Reduce

모든 GPU의 데이터를 GPU 0 하나에 합산:

```
Before:  GPU 0: [A]   GPU 1: [B]   GPU 2: [C]   GPU 3: [D]
After:   GPU 0: [A+B+C+D]   GPU 1: [B]   GPU 2: [C]   GPU 3: [D]
                (결과는 GPU 0만 가짐)

통신량: (N-1) × M
```

### All-Reduce

모든 GPU의 데이터를 합산해 **모든 GPU**에 결과를 돌려줌:

```
Before:  GPU 0: [A]      GPU 1: [B]      GPU 2: [C]      GPU 3: [D]
After:   GPU 0: [A+B+C+D] GPU 1: [A+B+C+D] GPU 2: [A+B+C+D] GPU 3: [A+B+C+D]

통신량: 2 × (N-1)/N × M × N ≈ 2M  (Ring 구현 기준)
```

Reduce + Broadcast와 논리적으로 같지만, Ring 구현으로 훨씬 효율적으로 할 수 있음.

### All-Gather

각 GPU가 가진 **조각**을 모아 모든 GPU가 전체를 갖게 함:

```
Before:  GPU 0: [A]   GPU 1: [B]   GPU 2: [C]   GPU 3: [D]
After:   GPU 0: [A,B,C,D]   GPU 1: [A,B,C,D]   GPU 2: [A,B,C,D]   GPU 3: [A,B,C,D]

통신량: (N-1)/N × M × N = (N-1) × M/N × N = (N-1)M
```

### Reduce-Scatter

합산 후 결과를 **조각으로 나눠** 각 GPU에 배분. All-Gather의 역방향:

```
Before:  GPU 0: [A0,A1,A2,A3]   GPU 1: [B0,B1,B2,B3]   ...
After:   GPU 0: [A0+B0+C0+D0]   GPU 1: [A1+B1+C1+D1]   ...
         (각 GPU가 합산 결과의 1/N씩 가짐)

통신량: (N-1) × M/N × N = (N-1)M
```

**All-Reduce = Reduce-Scatter + All-Gather**: 두 단계의 통신량 합이 All-Reduce와 같음.

### All-to-All

각 GPU가 **각 GPU에게 서로 다른 데이터**를 보냄:

```
GPU 0이 보낼 데이터: [to_0, to_1, to_2, to_3]
GPU 1이 보낼 데이터: [to_0, to_1, to_2, to_3]
...

결과: GPU k는 모든 GPU로부터 자기에게 보낸 데이터를 수신

통신량: N × M  (각 GPU가 M/N씩 N개 GPU에게 송신)
```

각 GPU가 주고받는 데이터 양이 routing 결과에 따라 달라질 수 있어 **load imbalance**가 발생하기 쉬움.

### Point-to-Point (P2P)

두 GPU 사이의 단방향 전송. Send/Recv 쌍으로 구성:

```
GPU 0 → GPU 1: activation tensor 전달
GPU 1 → GPU 2: 이어서 전달
...

통신량: M (전송 데이터 크기)
```

파이프라인에서 스테이지 경계를 넘을 때 사용. N-to-N이 아닌 특정 쌍 간 전송이므로 collective는 아니지만 분산 시스템의 기본 빌딩 블록.

---

## 2. Ring All-Reduce

### 왜 Ring인가

Naive All-Reduce (star topology): GPU 0이 모든 데이터를 받아 합산 후 다시 전파 → GPU 0이 병목, 통신량 O(N×M).

Ring: GPU를 원형으로 연결, 각 GPU가 이웃에게만 데이터를 보냄 → 병목 없음, 통신량 O(M) (N에 무관).

### 알고리즘: Reduce-Scatter 단계

N=4, 각 GPU가 4조각 [s0, s1, s2, s3]을 가짐:

```
초기:
  GPU 0: [A0, A1, A2, A3]
  GPU 1: [B0, B1, B2, B3]
  GPU 2: [C0, C0, C2, C3]
  GPU 3: [D0, D1, D2, D3]

Step 1 (GPU i → GPU (i+1) % N):
  GPU 0 → GPU 1: A3   GPU 1 → GPU 2: B0   GPU 2 → GPU 3: C1   GPU 3 → GPU 0: D2
  수신 후 합산:
  GPU 0: [A0, A1, A2+D2, A3]   GPU 1: [A3+B0(?), ...]  ...

(3 step 반복 후)

결과:
  GPU 0: [_, _, A2+B2+C2+D2, _]  (chunk 2의 완전한 합)
  GPU 1: [A3+B3+C3+D3, _, _, _]  (chunk 3의 완전한 합)
  GPU 2: [_, A0+B0+C0+D0, _, _]  (chunk 0의 완전한 합)
  GPU 3: [_, _, _, A1+B1+C1+D1]  (chunk 1의 완전한 합)
```

각 GPU는 전체 합산 결과의 1/N 조각을 가짐.

### 알고리즘: All-Gather 단계

Reduce-Scatter 결과를 다시 Ring으로 돌려 모든 GPU가 전체를 갖게 함:

```
Step 1: GPU 0 → GPU 1: chunk 2 완성본 전달
        GPU 1 → GPU 2: chunk 3 완성본 전달
        ...

(N-1 step 반복 후)

결과: 모든 GPU가 [sum0, sum1, sum2, sum3] 보유
```

### 대역폭 최적성

각 단계에서 각 GPU는 M/N 크기의 데이터를 송신·수신.  
총 단계 수: 2 × (N-1).  
GPU당 총 송수신량: 2 × (N-1)/N × M ≈ 2M (N이 크면 M에 수렴).

GPU 수가 늘어도 GPU 한 장이 처리하는 통신량은 **2M로 고정** → bandwidth-optimal.

---

## 3. Reduce-Scatter와 All-Gather의 등가 관계

```
All-Reduce(M) = Reduce-Scatter(M) + All-Gather(M)

통신량:
  Reduce-Scatter: (N-1)/N × M
  All-Gather:     (N-1)/N × M
  합계:           2(N-1)/N × M ≈ 2M   ← Ring All-Reduce와 동일
```

왜 이렇게 쪼개느냐:

Reduce-Scatter만 수행하면 각 GPU가 전체 합산 결과의 **1/N 조각만** 가짐. All-Gather 없이 그 조각만 쓰는 경우, All-Gather 비용을 아낄 수 있음.

예: N개의 GPU가 파라미터를 N조각으로 나눠 각자 1/N씩 책임질 때, Reduce-Scatter로 gradient를 합산하고 각자 담당 조각만 optimizer에 적용 → All-Gather 없이 끝. forward 시에만 All-Gather 필요.

---

## 4. 하드웨어 토폴로지

### NVLink (서버 내 GPU 간)

```
A100/H100 NVLink 4세대:
  단방향 BW: 300 GB/s × 18 link = 최대 ~600 GB/s (bidirectional 합산)
  8-GPU 서버에서 GPU 간 full-mesh NVLink → 어느 GPU 쌍도 직접 연결
```

- All-Reduce 알고리즘을 ring이 아닌 tree나 direct로 구현 가능
- 레이어마다 All-Reduce가 발생하는 Tensor Parallelism에 적합
- NVSwitch가 GPU 간 라우팅을 처리

### InfiniBand (서버 간)

```
HDR InfiniBand: 200 Gb/s ≈ 25 GB/s (단방향, per port)
NDR InfiniBand: 400 Gb/s ≈ 50 GB/s
RDMA 지원 → CPU 개입 없이 GPU 메모리 직접 전송 (GPUDirect RDMA)
```

- NVLink 대비 10× 이상 느림
- 서버 간 통신이 잦은 전략 (TP 서버 간 사용 등)은 병목
- Pipeline Parallelism처럼 경계에서만 통신하는 패턴이 IB에 적합

### PCIe (같은 서버 내, NVLink 없을 때)

```
PCIe Gen 4 x16: ~32 GB/s 단방향
PCIe Gen 5 x16: ~64 GB/s 단방향
NVLink 대비 ~10× 느림
```

- 소비자용 GPU (NVLink 없음) 환경에서의 기본 경로
- GPU-CPU-GPU 경유 → latency 추가

### 비교

| 인터커넥트 | 단방향 BW | 주 사용처 |
|---|---|---|
| NVLink 4세대 | ~300 GB/s/link | 서버 내 GPU 간 |
| InfiniBand NDR | ~50 GB/s | 서버 간 |
| PCIe Gen 4 x16 | ~32 GB/s | CPU-GPU, IB 없을 때 |

### 토폴로지가 알고리즘에 미치는 영향

8-GPU 서버 2대 (총 16 GPU)에서 All-Reduce 시:

```
방법 1: 16 GPU를 하나의 ring으로
  - 서버 간 IB를 8번 사용
  - IB가 bottleneck

방법 2: 서버 내 ring + 서버 간 All-Reduce 분리
  Step 1: 각 서버 내 8 GPU끼리 Reduce-Scatter (NVLink)
  Step 2: 서버 대표 GPU끼리 All-Reduce (IB)  ← 크기 1/8로 줄어든 채
  Step 3: 각 서버 내 All-Gather (NVLink)
  → IB 통신량을 1/8로 줄임
```

NCCL은 이런 topology-aware 알고리즘을 자동으로 선택.

---

## 5. Computation-Communication Overlap

### 왜 중요한가

All-Reduce가 완료될 때까지 GPU가 idle하면 통신 시간이 고스란히 latency에 추가됨. Overlap: 한 레이어의 통신과 다음 레이어의 연산을 동시에 실행.

### Backward에서의 Overlap (DP Gradient All-Reduce)

Backward는 마지막 레이어부터 역순으로 gradient를 계산. 각 레이어의 gradient가 준비되는 즉시 All-Reduce 시작:

```
시간 →
  backward layer N:   [grad 계산]
  All-Reduce layer N:               [통신   통신   통신]
  backward layer N-1:               [grad 계산]
  All-Reduce layer N-1:                          [통신   통신]
  backward layer N-2:                            [grad 계산]
  ...

grad 계산과 이전 레이어 All-Reduce가 동시에 실행
```

PyTorch DDP의 `bucket`이 이 단위. 레이어 여러 개의 gradient를 하나의 bucket으로 모은 뒤 All-Reduce → 작은 통신을 여러 번 하는 overhead 줄임.

### TP에서의 Overlap

Row Parallel 후 All-Reduce와 다음 레이어 연산을 overlap:

```
레이어 i: Row Parallel 완료 → All-Reduce 시작
레이어 i+1: (All-Reduce 진행 중) LayerNorm, 다음 Column Parallel 시작
```

All-Reduce가 완료되기 전에 다음 연산에 그 결과가 필요하면 overlap 불가. 구조적으로 가능한 부분만 겹칠 수 있음.

### 구현 방법

CUDA stream을 분리해 통신(NCCL)과 연산(compute kernel)이 독립적으로 진행:

```
compute_stream: [kernel_A] ........... [kernel_B]
comm_stream:             [all_reduce]
```

`torch.cuda.Stream`으로 명시적으로 분리하거나, NCCL이 별도 stream에서 동작하도록 설정.

---

## 6. 통신 primitive 비교 요약

| Primitive | 입력 → 출력 | 통신량 | 특징 |
|---|---|---|---|
| Broadcast | 1 GPU → 모두 | (N-1)M | 한 GPU가 데이터 전파 |
| Reduce | 모두 → 1 GPU | (N-1)M | 한 GPU에 결과 집중 |
| All-Reduce | 모두 → 모두 (합산) | 2(N-1)M/N ≈ 2M | Ring 구현으로 bandwidth-optimal |
| All-Gather | 조각 → 전체 (모두) | (N-1)M | 각 GPU의 조각을 전체로 |
| Reduce-Scatter | 모두 → 조각 (합산) | (N-1)M | 합산 후 분배 |
| All-to-All | 각자 → 각자 (다름) | N×M | 전원이 전원에게 다른 데이터 |
| P2P | 1 GPU → 1 GPU | M | 스테이지 간 activation 전달 |
