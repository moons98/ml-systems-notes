---
title: NPU (Neural Processing Unit)
description: 모바일·엣지 AI 가속기. Systolic array + weight-stall 기반. 텐서 shape·order에 민감하고 정적 그래프만 받음.
tags: [hardware, accelerator, npu, systolic-array, edge-ai]
type: concept
related_papers:
  - sosp25-heteroinfer
---

# NPU

## TL;DR

- NPU = matrix 연산에 특화된 ML 가속기.
- **Systolic array에 weight를 미리 로드(weight-stall)** 해두고 activation을 흘려보내는 방식으로 매 cycle 모든 PE가 일하도록 설계. 
- GPU 대비 W당 TOPS가 한 자릿수 높지만, **텐서 모양·순서·크기에 매우 민감**하고 **정적 그래프만 받음**. 
- 모바일·엣지 추론 가속기의 사실상 표준 패러다임.

## 1. 구조와 dataflow

### 1.1 Systolic array

2D PE(Processing Element) 격자. 각 PE = MAC(곱+누적) 유닛 + 1-cycle register.

설계 철학:
- 캐시·분기 예측·동적 dispatch **없음**
- 그 실리콘 절약분을 전부 MAC에 투자
- 매 cycle 모든 PE가 동시에 일함 (GPU SIMT의 warp divergence 같은 idle 없음)
- 결과: W당 TOPS가 GPU의 한 자릿수 위, 클럭은 ~1 GHz

### 1.2 INT PE와 FP PE

PE는 INT용과 FP용이 구분된다. 이유는 MAC 회로 복잡도의 차이:

- **INT8 MAC**: 정수 곱셈기 + 누산기. 회로가 단순.
- **FP16 MAC**: 지수부 정렬 → 가수부 곱셈 → 정규화까지 여러 단계 필요. INT 대비 트랜지스터 수가 많음.

같은 실리콘 면적에 INT MAC을 FP MAC보다 더 많이 집적할 수 있으므로, NPU는 INT 연산 유닛을 더 많이 두어 **INT TOPS > FP TOPS** 구조를 취한다.

대표 수치 (Snapdragon 8 Gen 3):

| 정밀도 | TOPS |
|---|---|
| INT4 / INT8 | 34 |
| FP16 | 17 |

INT/FP 배열의 내부 배치(크기, 개수, 구조)는 칩마다 다르고 비공개 영역. 확실한 것은 INT MAC 회로가 더 단순해서 집적 밀도가 높다는 것.

### 1.3 Weight-stall + skew streaming

`C[M,N] = A[M,K] × B[K,N]`을 K×N 격자에서 처리할 때:

- **B를 PE 격자에 preload** — PE[k,n]에 B[k,n]이 stall (K-tile 동안 고정)
- **A를 좌측 edge로 skew streaming** — A[m,k]가 PE[k,0]에 cycle (m+k)에 입장
- **Partial sum이 column 따라 위→아래로 전달** — PE[k,n]은 위에서 받은 합에 자기 곱을 더해 아래로 패스

```
                   A 입력 (skew된 좌측)
                       ↓
            ┌──────────────────────┐
            │   K×N PE array       │
            │   B preloaded        │  ← partial sum 위→아래 ↓
            │                      │
            └──────────────────────┘
                ↓↓↓ N개 wire (bottom edge)
            ┌──────────────────────┐
            │  Accumulator SRAM    │  ← 외부, 별도 자원
            └──────────────────────┘
```

- **Bottom row의 PE 결과만 SRAM으로 토출.** 
- 다른 모든 PE의 부분합은 격자 안에서 chain되어 column 끝까지 hop down. (한 cycle씩 한 PE 아래로 전달)

**예시: 3×3 격자, A[3,3] × B[3,3] 스트리밍**

Input skew — A의 각 원소가 left edge에 들어오는 cycle (row k는 k cycle 늦게 시작):

```
       cycle:  0      1      2      3      4
row 0:        A[0,0] A[1,0] A[2,0]
row 1:               A[0,1] A[1,1] A[2,1]
row 2:                      A[0,2] A[1,2] A[2,2]
```

Cycle별 PE 활성 패턴 (○ = MAC 수행, · = idle):

```
cycle 0:     cycle 1:     cycle 2:     cycle 3:     cycle 4:
  ○ · ·        ○ ○ ·        ○ ○ ○        · ○ ○        · · ○
  · · ·        ○ · ·        ○ ○ ·        ○ ○ ○        · ○ ○
  · · ·        · · ·        ○ · ·        ○ ○ ·        ○ ○ ○

  warmup       warmup       steady       drain        drain
```

대각선 wave가 좌상단에서 시작해 채워졌다가 우하단으로 빠져나감.

C[0,0]이 만들어지는 chain 추적 (column 0):

- **cycle 0**: PE[0,0] = `A[0,0] × B[0,0]` → 부분합 P → 아래로 패스
- **cycle 1**: PE[1,0] = `P + A[0,1] × B[1,0]` → 갱신된 P → 아래로 패스
- **cycle 2**: PE[2,0] = `P + A[0,2] × B[2,0]` = **C[0,0]** → bottom edge → SRAM의 slot[0]

같은 cycle에 다른 column·row에서도 wave가 진행 중. cycle 2~6의 bottom edge 토출 패턴:

| cycle | column 0 | column 1 | column 2 |
|---|---|---|---|
| 2 | C[0,0] | — | — |
| 3 | C[1,0] | C[0,1] | — |
| 4 | C[2,0] | C[1,1] | C[0,2] |
| 5 | — | C[2,1] | C[1,2] |
| 6 | — | — | C[2,2] |

→ output도 입력처럼 **대각선 wave**로 emerge. m+n이 일정한 anti-diagonal이 같은 cycle에 토출됨.

### 1.4 Skew의 두 가지 역할

(겉보기엔 그냥 비스듬히 넣는 트릭이지만 두 가지 본질적 기능을 동시에 수행)

1. **공간 reduction을 시간으로 펼치기**. 같은 cycle에 column 전체 PE의 곱이 동시에 나오면 N-입력 가산기 트리(log N 단)가 필요해 클럭의 critical path 길어짐. Skew하면 매 cycle 한 PE의 합산만 일어나 **2-입력 가산기**로 끝남 → 클럭 ↑.

2. **Output index 정합성**. PE[k+1, n]이 새 곱을 만드는 cycle과 PE[k, n]에서 한 cycle 늦게 내려온 partial sum의 cycle이 정확히 일치 → 같은 output `C[m, n]`에 속하는 항들만 누적됨. Skew 없으면 다른 output의 항들이 섞임.

### 1.5 Tile 실행

격자보다 큰 matmul은 tile로 분할. 32×32 격자에서 64×64×64 matmul 처리 시:

B를 4개 tile로 분할:
```
B[0:32, 0:32]    B[0:32, 32:64]
B[32:64, 0:32]   B[32:64, 32:64]
```

C[:, 0:32] (좌측 32 column)을 구하려면:
```
C[:, 0:32] = A[:, 0:32]  × B[0:32,  0:32]   ← K-tile 0 (accumulator에 write)
           + A[:, 32:64] × B[32:64, 0:32]   ← K-tile 1 (accumulator에 RMW)
```

C[:, 32:64]도 같은 식으로 K-tile 0/1 처리 (accumulator 다른 slot 영역).

**A 슬라이스 재사용**: 같은 K 범위를 공유하는 두 B-tile은 같은 A 슬라이스를 stream → A 총 read = N-tile 수만큼만 (예에선 2회).

### 1.6 Accumulator SRAM

PE 격자 외부의 별도 SRAM. 격자 크기와 독립.

- 매 cycle bottom edge가 N개 값 토출 (한 row of partial C)
- M cycle 동안 M개 row가 차례로 slot[0..M−1]에 기록
- K-tile 사이는 **read-modify-write**로 같은 slot에 누적
- Accumulator 정밀도는 보통 MAC보다 넓음 (예: FP16 MAC + FP32 acc) — K가 큰 누적의 정밀도 손실 방지

(실제로는 출력도 입력처럼 skew된 대각선 wave로 emerge. Deskew shift register나 cycle 기반 slot indexing으로 정렬됨.)

## 2. 자원과 축의 매핑

NPU 성능 분석의 핵심: 매트릭스의 세 축 M·N·K가 **각각 다른 종류의 자원에 매핑**됨.

| 차원 | 의미 | 매핑되는 자원 | 비용 척도 |
|---|---|---|---|
| **N** (output column) | 격자 column 수에 묶임 | 공간 | N-tile 수만큼 격자 setup |
| **M** (output row, streaming dim) | 시간 (cycle 수) | 시간 | warmup 분담률 ∝ M |
| **K** (inner dim) | accumulator RMW 누적 | 누적 패스 | K-tile 수만큼 weight load |

**산술 강도 (per B tile load)**:
- Useful work: M × 32 × 32 MACs
- Load cost: 32 × 32 weight 원소 한 번
- 비율 = M

→ **M이 클수록 weight load가 시간으로 amortize되어 효율 증가**. NPU에서 "큰 batch / 긴 sequence가 좋다"의 정량적 근거.

## 3. 정적 그래프 제약

### 3.1 AOT compile이 결정하는 것

NPU는 graph compilation이 **사전(AOT, Ahead-Of-Time)** 에 일어나고 모든 결정이 baked-in:
- Tile 분할 (M/K/N 각 축)
- Accumulator slot 할당 (얼마나, 어디에)
- 각 cycle에 어느 slot에 write 또는 RMW
- DMA를 언제 띄워 weight tile을 가져올지
- 첫 K-tile은 write 모드, 이후는 RMW 모드

이 모든 결정이 **shape의 함수**. shape이 1픽셀이라도 바뀌면 통째로 무효 → 다시 컴파일.

### 3.2 "Stall"의 두 가지 의미 — NPU 비효율의 정체

같은 단어지만 분리해서 봐야 함:
1. **Compute paradigm**: weight가 PE에 머물면서 activation이 흐름 (의도된 효율)
2. **Load barrier**: PE 격자가 weight tile을 다 채우기 전까지 idle (비효율의 원천)

| | Weight (B) | Activation (A) |
|---|---|---|
| 도달 단위 | **batch** (한 tile = K×N개 한 번에) | **drip** (매 cycle 32값) |
| compute 시작 시점 | preload barrier 후 | 첫 column 도착 후 |
| latency 노출 | critical path **노출** | compute 뒤에 **숨음** |
| DRAM access pattern | strided tile loads | sequential streaming |

핵심 통찰: **Latency hiding은 latency 자체를 줄이는 게 아니라 critical path에서 빼는 것.** Weight stall load는 항상 노출됨 → tile 수가 많으면 그 비용이 곱해져서 폭증.

### 3.3 Dynamic shape 비용

Shape별 그래프 컴파일 시간이 텐서 크기에 비례. 모바일 NPU에서 단일 matmul 그래프 생성에 **수백 ms~초 단위** 소요 (Snapdragon 8 Gen 3 기준 seq=1024 matmul ~920 ms). → Runtime 그래프 생성 사실상 불가능.

대안: 미리 컴파일된 표준 shape의 그래프 풀을 두고 입력을 거기 맞추거나(padding), 표준 shape의 시퀀스로 분해(pipe), 잔여만 다른 backend로 위탁(GPU partition).

### 3.4 GPU SIMT와의 본질 차이

| | GPU (SIMT) | NPU (Systolic) |
|---|---|---|
| 데이터 도달 | warp가 그때그때 load (drip) | weight batch preload |
| Shape flexibility | 어떤 shape이든 처리 (덜 효율) | 정해진 모양만 효율적 |
| Peak 효율 | 낮음 (warp divergence, scheduling overhead) | 높음 (모든 PE 매 cycle 일함) |
| 정적 vs 동적 | runtime kernel dispatch | AOT compile 후 정해진 시퀀스 실행 |

trade-off의 정수: **batch load → peak 효율 ↑ + flexibility ↓**.

## 4. 성능 특성 (NPU-1, 2, 3)

이 셋이 NPU 성능 분석의 거의 전부. HeteroInfer (SOSP'25) 분류를 빌림.

### 4.1 Stage performance (NPU-1)

HW tile 크기(예: 32×32)에 align되지 않는 shape은 padding 필요. 32 배수 미만은 사실상 같은 latency를 내는 **계단식 함수**.

```
matmul latency
   |          ___
   |      ___|        ← 다음 stage로 점프
   |     |
   |  ___|
   |_|
   +--------------------- shape (K나 N)
       32   64   96
```

**원인**: PE 격자가 32×32 단위로만 tiling 가능 → 33은 64로 padding되거나 두 tile로 쪼개짐.

**회피**:
- 활성을 표준 사이즈 청크로 사전 분할 (chunked / pipe)
- 표준 shape에서 벗어난 잔여만 GPU로 offload (activation-centric partition)

### 4.2 Order-sensitive performance (NPU-2)

같은 연산량인데 텐서 순서를 바꾸면 **6× 차이**도 가능. 원인: weight tile 수 폭증 + tile당 amortization 붕괴.

**예**: `[M=8, K=4096] × [K=4096, N=14336]`
- B = [4096, 14336] → tile 수 = 128 × 448 = **57,344**
- per tile compute = M × 32 × 32 = 8192 MACs (M=8 cycle만 stream)
- per tile load barrier가 8 cycle compute를 압도

**Transpose invariant** `(A @ B)ᵀ = Bᵀ @ Aᵀ` 로 swap → `[14336, 4096] × [4096, 8]`:
- B = [4096, 8] → tile 수 = 128 × 1 = **128** (448× 적음)
- per tile compute = 14336 × 32 × 32 = 14.7M MACs
- tile load 비용이 compute에 완전히 묻힘

같은 데이터, 같은 연산량, 다른 시간. Output transpose(Cᵀ)는 결과 read 시 자동 처리.

**회피**: matmul 호출 시 weight·activation의 역할을 항상 swap해서 streaming dim(M)을 최대화.

### 4.3 Shape-sensitive performance (NPU-3)

Input의 row > column이어야 효율적. 두 메커니즘이 동시에 작용:

1. **Streaming amortization**: row(M)이 클수록 weight tile 한 번 load당 더 많은 cycle stream → load 비용 분담률 ↑
2. **Weight tensor 자체 크기**: column(K)이 weight의 row와 같음 → K가 크면 weight tensor 자체가 커짐 → tile 수 폭증

차원 축소 matmul (예: FFN-down `[hidden, intermediate] × [intermediate, hidden]`)은 transpose 후에도 stall 텐서가 [hidden, hidden]으로 여전히 큼 → NPU 이점 0.5~1.5×에 그침.

**회피**: NPU에 약한 shape의 op는 GPU에 weight-centric partition으로 분할 위탁.

## 5. 실무 우회 패턴

LLM 추론처럼 dynamic shape이 불가피한 경우 일반적으로 쓰이는 전략들:

| 패턴 | 방법 | trade-off |
|---|---|---|
| **Padding** | 표준 사이즈로 padding, 한 그래프로 처리 | 단순. 입력이 표준 사이즈를 살짝 넘으면 평균 ~1.91× 오버헤드 |
| **Online graph prep** | runtime에 그래프 생성 | 그래프 생성 비용 자체가 추론보다 클 수 있음 (~수 초) |
| **Chunked-prefill** | 고정 크기 청크로 분할 후 순차 처리 | 청크 크기 튜닝 필요. 짧은 시퀀스에서 효율 50% 저하 가능 |
| **NPU-pipe** | 동적 그래프를 표준 그래프 시퀀스로 분해 | activation-centric보다 13~30% 느림 |
| **Activation-centric partition** | 표준 청크는 NPU, 동적 잔여만 GPU | 가장 효율적, GPU 협업 필요 (HeteroInfer) |
| **Weight-centric partition** | weight를 row 차원으로 GPU/NPU 분할 | NPU 약한 shape에서 GPU와 병렬 (HeteroInfer) |
| **Hybrid partition** | 잔여를 padding으로 표준화 + weight-centric | 잔여가 너무 작아 GPU도 NPU도 underutilize일 때 |

## 6. 대표 구현

| | 격자 크기 (대략) | 정밀도 | 메모리 |
|---|---|---|---|
| Google TPU v3/v4 | 128×128 | bf16, int8 | HBM (datacenter) |
| Qualcomm Hexagon (Snapdragon 8 Gen 3) | 32×32 (다중) | int8, fp16 | UMA (LPDDR) |
| Apple Neural Engine | 비공개 | fp16, int8 | UMA |
| MediaTek APU 790 | 비공개 | int8, fp16 | UMA |
| NVDLA (오픈소스) | 16×16~64×64 (configurable) | int8, fp16 | shared SoC |

(클라우드 TPU는 데이터센터 가속기지만 동일 패러다임. 모바일 NPU는 UMA + thermal/power 제약이 추가)
