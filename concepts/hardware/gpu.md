---
title: GPU (Graphics Processing Unit)
description: 범용 병렬 가속기. SIMT 실행 모델, SM/warp 계층 구조, warp 기반 latency hiding. 어떤 shape·연산 패턴도 처리 가능한 유연성이 핵심.
tags: [hardware, accelerator, gpu, simt, warp, tensor-core, ml-inference]
type: concept
related_papers:
  - sosp25-heteroinfer
---

# GPU

## TL;DR

- GPU = 대규모 병렬 연산에 특화된 범용 가속기.
- **SIMT(Single Instruction, Multiple Threads)** 실행 모델: 수천 개의 thread가 SM/warp 계층으로 조직되어 같은 명령어를 동시에 실행.
- 핵심 효율 메커니즘은 **warp 기반 latency hiding** — stall된 warp 대신 다른 warp로 전환해 compute unit을 쉬지 않게 함.
- NPU 대비 W당 TOPS는 낮지만, **어떤 shape·연산 패턴·동적 제어 흐름도 처리** 가능.
- ML 워크로드에서는 **Tensor Core**(행렬곱 전용 유닛)로 throughput을 추가 확보.

---

## 1. 구조 — SM / warp / thread 계층

### 1.1 전체 구조

```
GPU
├── SM 0                    ← Streaming Multiprocessor (수십~수백 개)
│   ├── warp 0  [32 threads]
│   ├── warp 1  [32 threads]
│   ├── ...
│   ├── Register file       ← thread당 수십~수백 개 레지스터
│   ├── Shared Memory (SMEM)← SM 내 thread 간 공유, ~48~228 KB
│   ├── L1 Cache            ← SMEM과 물리 합산 (configurable split)
│   └── Tensor Core (s)     ← 행렬곱 전용 유닛
├── SM 1
├── ...
├── L2 Cache                ← 전체 SM 공유, 수 MB~수십 MB
└── HBM / GDDR              ← 글로벌 메모리, 수십~수백 GB/s BW
```

### 1.2 Thread 계층

SW가 GPU를 프로그래밍할 때 쓰는 추상화:

| 단위 | 설명 | 하드웨어 대응 |
|---|---|---|
| **Thread** | 최소 실행 단위. 각자 register 보유 | 물리 실행 lane 1개 |
| **Warp** | 32 thread 묶음. HW 스케줄링 단위 | 1 warp = 1 스케줄링 슬롯 |
| **Block** | 프로그래머가 정의하는 thread 그룹. SMEM 공유 | 1 SM에 배치 (분할 없음) |
| **Grid** | 모든 block의 집합 = kernel 1회 launch | 전체 GPU에 분산 |

warp는 SW 추상화가 아닌 **HW 스케줄링의 실제 단위**다. block이 SM에 배치되면 HW가 자동으로 warp로 쪼갠다.

---

## 2. SIMT 실행 모델

**Single Instruction, Multiple Threads**: 한 warp의 32 thread가 매 cycle **같은 명령어**를 실행하되, 각자 다른 데이터(다른 레지스터 값)를 처리.

CPU의 SIMD(벡터 연산)와 유사하지만 차이가 있다:

| | SIMD | SIMT |
|---|---|---|
| 추상화 | 벡터 레지스터 | 독립된 thread (각자 PC, 레지스터 보유) |
| 분기 처리 | 불가 (동일 연산 강제) | 가능 (thread별 분기 — 단, 비효율 발생) |
| 프로그래밍 모델 | 벡터 intrinsic | 일반 코드 그대로 (HW가 병렬화) |

### Warp divergence

warp 내 thread들이 **서로 다른 분기**(`if/else`)를 타면 두 경로를 순차 실행하고 한쪽을 mask off. 32 lane 중 절반만 실제로 연산 → 효율 50% 손실.

```
if (thread_id % 2 == 0) { ... }   // 짝수 thread만 실행
else                   { ... }   // 홀수 thread만 실행
                                  // → 두 path를 직렬로 실행, 항상 둘 다 소비
```

회피: 같은 warp 내 thread들이 같은 경로를 타도록 데이터 / 알고리즘 재정렬.

---

## 3. Latency hiding — warp 기반

GPU의 핵심 효율 메커니즘. latency를 **줄이는** 게 아니라 **critical path에서 빼는** 방식.

### 동작

SM에 여러 warp를 동시에 in-flight로 올려두고, 한 warp가 메모리 접근 등으로 stall되면 **0-cycle overhead로 다른 warp로 전환**:

```
cycle:  0    1    2    3    4    5    6    7  ...
warp 0: MEM_REQ  ...stall...stall...stall  RESULT  compute
warp 1:      compute  compute  compute
warp 2:               MEM_REQ  ...stall   RESULT  compute
```

warp 0이 메모리 기다리는 동안 warp 1이 실행. compute unit이 쉬지 않음.

### Occupancy

SM이 동시에 수용 가능한 warp 수 대비 실제 active warp 수의 비율. 높을수록 latency hiding 효과 큼.

occupancy를 제한하는 세 자원:

| 자원 | 영향 | 예시 |
|---|---|---|
| **Register** | thread당 register 수 ↑ → SM당 수용 thread 수 ↓ | 레지스터 많이 쓰는 kernel → occupancy ↓ |
| **Shared Memory** | block당 SMEM 사용 ↑ → SM당 수용 block 수 ↓ | SMEM 많이 쓰면 block 적게 배치됨 |
| **Block 크기** | block당 thread 수가 warp 크기(32) 배수가 아니면 낭비 | block=33 → warp 2개 배치, 한 warp 31 lane idle |

---

## 4. 메모리 계층

접근 속도와 용량의 trade-off:

```
빠름 / 작음 ←─────────────────────────── 느림 / 큼
Register → SMEM → L1 → L2 → HBM/GDDR
  ~1 cycle   ~5     ~30  ~200   ~수백 cycle
```

| 메모리 | 위치 | 용량 | 공유 범위 | 관리 주체 |
|---|---|---|---|---|
| **Register** | SM 내 | thread당 수십~수백 B | thread 전용 | 컴파일러 |
| **Shared Memory (SMEM)** | SM 내 | ~48~228 KB | block 내 thread | 프로그래머 |
| **L1 Cache** | SM 내 | SMEM과 물리 공유 | SM 내 자동 | HW |
| **L2 Cache** | GPU 전역 | 수 MB~수십 MB | 전체 SM | HW |
| **HBM / GDDR** | 칩 외부 | 수십~수백 GB | 전체 GPU | 프로그래머 |

### Memory coalescing

global memory(HBM) 접근은 **warp 내 32 thread의 접근이 연속된 주소로 합쳐질 때** (coalesced) 효율적. 흩어지면 (strided, random) 트랜잭션 수가 폭증:

```
Coalesced:   thread 0~31이 addr[0]~addr[31] 접근 → 트랜잭션 1회
Strided:     thread 0~31이 addr[0], addr[32], addr[64]... → 트랜잭션 32회
```

---

## 5. Tensor Core — ML 전용 유닛

A100부터 표준화. 일반 CUDA core가 scalar FMA를 하는 것과 달리, Tensor Core는 **작은 행렬 곱(tile matmul)을 1~수 cycle에 처리**하는 전용 회로.

대표 연산 (`mma` — matrix multiply-accumulate):

```
D[M, N] = A[M, K] × B[K, N] + C[M, N]
```

Tensor Core가 지원하는 tile 크기 (Ampere 기준):

| 정밀도 | Tile (M×N×K) | 비고 |
|---|---|---|
| FP16 | 16×16×16 | 기본 |
| BF16 | 16×16×16 | 학습에서 선호 |
| TF32 | 16×16×8 | FP32 대체 (Ampere+) |
| INT8 | 16×16×32 | 추론 최적화 |
| FP8 | 16×16×32 | H100+ (학습 추론 모두) |

CUDA에서는 `wmma` API 또는 `cutlass` 라이브러리로 접근. cuBLAS·cuDNN이 내부적으로 사용.

### CUDA core vs Tensor Core

| | CUDA core | Tensor Core |
|---|---|---|
| 연산 단위 | scalar FMA (a×b+c) 1회 | tile matmul 1회 |
| 처리량 (A100 FP16) | ~기준 | ~8× |
| 사용 조건 | 임의 연산 | 행렬곱 tile 크기에 align |
| 주요 용도 | 일반 병렬 연산 | GEMM, Attention, Conv |

LLM 추론에서 matmul이 지배적인 이유로 Tensor Core 활용률이 GPU 효율의 핵심 지표가 됨.

---

## 6. ML 추론에서의 특성

### 6.1 GPU가 잘하는 상황

- **Dynamic shape**: 입력 길이가 매번 달라도 runtime kernel dispatch로 처리
- **Decode 단계 (seq=1, batch=1)**: NPU는 M=1에서 weight load 비용이 compute를 압도하지만 GPU는 다른 warp로 latency hiding
- **조건 분기·복잡한 제어 흐름**: speculative decoding, beam search 등
- **NPU에 약한 shape**: 차원 축소 matmul (FFN-down 등)

### 6.2 GPU의 한계

- **W당 TOPS**: 제어 로직 실리콘 비용으로 NPU 대비 낮음
- **Memory BW bottleneck**: Decode 단계에서 weight를 매 step HBM에서 읽어야 함 → arithmetic intensity 낮아 BW bound
- **전력·열**: 모바일·엣지에서 thermal budget 초과 위험

### 6.3 Compute bound vs Memory bound

GPU 워크로드의 병목 판별:

- **Compute bound**: FLOP 수가 병목. Tensor Core 활용률 ↑, BW는 여유.
  - 예: Prefill (긴 sequence, 큰 batch)
- **Memory bound**: HBM BW가 병목. compute unit이 데이터 기다리며 idle.
  - 예: Decode (batch=1, weight 전체를 매 step 읽음)

```
Arithmetic Intensity (FLOP/Byte) vs Hardware Ridge Point 비교:
- AI > Ridge: Compute bound
- AI < Ridge: Memory bound (BW bottleneck)
```

Decode에서 arithmetic intensity ≈ 1~2 FLOP/Byte (A100 ridge ~250) → 극단적 memory bound → KV cache, quantization, continuous batching 등의 최적화 동기.

---

## 7. 대표 구현

| | SM 수 (대략) | HBM BW | 정밀도 | 주 용도 |
|---|---|---|---|---|
| NVIDIA H100 SXM | 132 SM | ~3.35 TB/s | FP8/FP16/BF16/TF32 | 학습·추론 (datacenter) |
| NVIDIA A100 | 108 SM | ~2 TB/s | FP16/BF16/TF32/INT8 | 학습·추론 (datacenter) |
| NVIDIA RTX 4090 | 128 SM | ~1 TB/s | FP16/INT8 | 추론·연구 (consumer) |
| Qualcomm Adreno (모바일) | 수 SM | ~50~60 GB/s (UMA 공유) | FP16/INT8 | 모바일 GPU 추론 |
| Apple M3 GPU | 비공개 | ~100 GB/s (UMA 공유) | FP16 | 모바일·데스크톱 |

모바일 GPU는 UMA로 CPU·NPU와 BW를 공유. Discrete GPU는 전용 HBM으로 절대 BW가 한 자릿수 위.

---