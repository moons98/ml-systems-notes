---
title: PIM (Processing-in-Memory)
description: 메모리 내부 또는 가까이에 연산기를 배치해 데이터 이동을 줄이는 가속기 구조. DRAM-PIM의 구성, 명령 실행, 데이터 매핑, LLM 추론 적용과 병목을 정리한다.
tags: [hardware, accelerator, pim, dram, memory-bandwidth, llm-inference, attention]
type: concept
related_papers:
  - hpca26-pimphony
---

# PIM (Processing-in-Memory)

## TL;DR

- **PIM은 데이터를 프로세서로 가져오는 대신, 연산을 메모리 가까이 가져간다.**
- DRAM-PIM은 각 channel 또는 bank 근처에 작은 MAC 연산기를 배치해, DRAM 내부의 높은 병렬성과 대역폭을 직접 사용한다.
- 낮은 arithmetic intensity를 가진 GEMV, recommendation, graph processing, LLM decode attention처럼 **데이터를 많이 읽지만 계산은 적은 작업**에 유리하다.
- PIM 연산기는 GPU보다 단순하고 느리지만, 데이터를 외부 bus로 이동하지 않아 전체 시스템에서 더 높은 throughput과 energy efficiency를 얻을 수 있다.
- 높은 내부 대역폭이 자동으로 높은 성능을 뜻하지는 않는다. **데이터 배치, channel utilization, I/O와 MAC scheduling, 동적 메모리 관리**가 함께 맞아야 한다.

---

## 1. 왜 PIM이 필요한가

일반적인 시스템에서는 연산을 위해 데이터를 메모리에서 프로세서로 이동한다.

```text
DRAM -> memory bus -> cache/register -> ALU
```

이 방식은 연산량보다 읽어야 할 데이터가 많은 workload에서 비효율적이다.

- 프로세서의 계산 성능은 빠르게 증가했다.
- 외부 memory bandwidth는 상대적으로 느리게 증가했다.
- DRAM 내부에는 많은 bank가 병렬로 동작하지만, 외부 bus 대역폭 때문에 그 잠재력을 모두 노출하지 못한다.
- 데이터 이동은 연산 자체보다 많은 시간과 에너지를 소비할 수 있다.

이를 흔히 **memory wall** 또는 **von Neumann bottleneck**이라고 부른다.

PIM은 DRAM 내부 또는 매우 가까운 곳에 연산기를 추가한다.

```text
Conventional:
  DRAM banks -> 좁은 외부 bus -> GPU/CPU 연산기

PIM:
  DRAM banks -> 근처의 작은 연산기
             -> 작은 결과만 외부로 전송
```

핵심은 PIM 연산기가 GPU보다 강력하다는 것이 아니다. **외부로 이동해야 하는 데이터 양을 줄이고 DRAM 내부 병렬성을 활용한다는 것**이다.

---

## 2. DRAM 구조 기초

PIM을 이해하려면 DRAM의 계층 구조부터 알아야 한다. 아래에서 **channel은 chip이 아니라 memory controller가 제공하는 독립적인 command/data interface**다.

```text
CPU / xPU
└── Memory Controller
    ├── Channel 0: 독립 command/data bus
    │   └── DIMM
    │       ├── Rank 0: 여러 DRAM chip이 동시에 동작하는 묶음
    │       │   ├── DRAM Chip 0
    │       │   │   ├── Bank 0 -> Rows + Row Buffer
    │       │   │   ├── Bank 1 -> Rows + Row Buffer
    │       │   │   └── ...
    │       │   ├── DRAM Chip 1 -> Banks
    │       │   └── ...
    │       └── Rank 1 (optional)
    └── Channel 1
```

### Channel

- memory controller에서 나오는 독립적인 command/data bus 또는 interface다.
- **channel 하나가 DRAM chip 하나를 의미하지 않는다.**
- 일반적인 DDR 시스템에서는 channel 하나에 하나 이상의 DIMM이 연결될 수 있고, DIMM 하나에는 여러 DRAM chip이 있다.
- 여러 channel은 동시에 동작할 수 있다.
- channel 수가 많을수록 외부 memory bandwidth도 일반적으로 증가한다.

### DIMM, Rank, DRAM Chip

- **DIMM**: PCB 위에 여러 DRAM chip을 장착한 물리적 memory module이다.
- **Rank**: memory controller의 한 command에 함께 반응해 하나의 넓은 data word를 만드는 DRAM chip 묶음이다.
  - 예를 들어 x8 DRAM chip 8개가 함께 동작하면 64-bit rank를 구성할 수 있다.
- **DRAM chip**: 실제 DRAM die/package이며, 각 chip 내부가 여러 bank로 나뉜다.
- controller가 어떤 rank의 bank와 row를 접근할지 명령하면, 해당 rank에 속한 chip들이 병렬로 자신의 데이터 조각을 제공한다.

### Bank

- 물리적으로는 **각 DRAM chip 내부**에 존재하는 독립적인 DRAM 배열이다.
- controller 관점에서는 같은 rank의 여러 chip에 있는 대응 bank들이 함께 동작해 channel 폭에 맞는 데이터를 제공한다.
- 서로 다른 bank의 작업은 겹쳐 실행할 수 있다.
- PIM은 이 bank-level parallelism을 활용하도록 bank마다 연산기를 두거나, 여러 bank가 하나의 연산기를 공유하도록 설계할 수 있다.

### Row와 Row Buffer

DRAM은 임의의 byte를 바로 읽지 않는다.

1. **ACTIVATE (ACT)**: 원하는 row 전체를 row buffer로 연다.
2. **READ/WRITE**: 열린 row의 column을 읽거나 쓴다.
3. **PRECHARGE (PRE)**: row를 닫아 다른 row를 열 준비를 한다.

같은 열린 row를 반복 접근하면 빠르지만, row를 자주 바꾸면 ACT/PRE 비용이 반복된다. 따라서 PIM에서도 **데이터를 어느 row와 bank에 배치하는가**가 성능에 중요하다.

---

### DDR DIMM과 HBM/PIM module의 차이

위 계층은 일반적인 DDR DIMM을 설명한 것이다. HBM이나 논문 속 PIM module은 packaging과 용어가 다를 수 있다.

- HBM stack은 하나의 package 안에 여러 channel 또는 pseudo-channel을 제공한다.
- PIM 전용 module은 내부에 여러 PIM channel, bank, controller, shared buffer를 통합할 수 있다.
- 따라서 논문의 `PIM module -> channels -> banks` 그림은 일반 DIMM의 물리 계층을 그대로 그린 것이 아니라, **PIM 장치가 노출하는 논리적 실행 자원**을 나타낸다.

---

## 3. PIM의 종류

PIM이라는 이름은 연산기가 메모리와 얼마나 가까운지에 따라 넓게 사용된다.

| 분류 | 연산기 위치 | 특징 |
|---|---|---|
| Processing-Near-Memory (PNM) | memory controller, logic die, module 내부 | xPU보다 메모리에 가깝지만 DRAM array 밖에 있을 수 있음 |
| Processing-Using-Memory (PUM) | memory cell/array 자체 | 셀의 전기적 특성을 이용, 높은 병렬성이나 지원 연산이 제한적 |
| DRAM-PIM | DRAM chip/module 내부의 bank 또는 bank group 근처 | 기존 DRAM 구조와 MAC 연산기를 결합해 내부 대역폭 사용 |

따라서 PIM은 넓게 보면 "xPU보다 메모리에 가까운 연산"을 포함하지만, 모든 PIM이 단순한 외장 xPU인 것은 아니다. **DRAM-PIM은 연산기가 실제 DRAM chip/module 쪽에 통합되어, 데이터를 외부 memory channel로 내보내지 않고 bank 내부 대역폭으로 처리한다.**

LLM inference용 PIM 논문에서 흔히 다루는 것은 DRAM-PIM 또는 PNM에 가까운 구조다. PIMphony는 DRAM bank 근처의 vector MAC과 module-level controller를 전제로 한다.

---

## 4. 전형적인 DRAM-PIM 구조

```text
PIM Module
├── PIM HUB
│   ├── Instruction Sequencer
│   ├── General-Purpose Registers (GPR)
│   ├── Extra Processing Unit (EPU)
│   └── Multicast Interconnect
│
├── Channel 0
│   ├── Global Buffer (GBuf)
│   ├── Bank 0 + Vector MAC + Output Register
│   ├── Bank 1 + Vector MAC + Output Register
│   └── ...
├── Channel 1
└── ...
```

### Bank-side MAC

- 각 bank 가까이에 작은 vector MAC을 두거나, 하나의 MAC이 bank group의 여러 bank를 담당한다. 정확한 배치는 PIM architecture마다 다르다.
- weight나 KV cache는 DRAM row에 저장한다.
- input vector를 작은 buffer에 넣고 여러 bank가 동시에 dot product를 수행한다.
- 결과는 Output Register에 누적한 뒤 module의 GPR 또는 EPU로 모은다.

### Global Buffer (GBuf)

- channel 내 bank들이 공유하는 작은 input buffer다.
- 외부에서 들어온 activation, query, attention score 등을 저장한다.
- 용량이 작아 input을 자주 교체하면 I/O가 병목이 된다.

### PIM HUB

- compiler가 생성한 PIM instruction을 받는다.
- instruction을 channel별 command로 펼쳐 multicast한다.
- channel 결과를 모으고 reduction, softmax 같은 보조 연산을 수행할 수 있다.

---

## 5. PIM에서 GEMV가 실행되는 방식

다음 matrix-vector multiplication을 생각하자.

```text
y = W x

W: [M, K] matrix, DRAM에 저장
x: [K] input vector
y: [M] output vector
```

전형적인 실행 흐름은 다음과 같다.

```text
1. WR-INP: x의 작은 tile을 GBuf에 기록
2. MAC:    각 bank가 자신의 W row와 input tile을 dot product
3. MAC:    K 방향의 다음 tile을 누적
4. RD-OUT: 완성된 결과를 Output Register에서 읽음
```

여러 bank가 서로 다른 row의 dot product를 동시에 계산하므로 큰 GEMV를 병렬화할 수 있다.

### 왜 GEMV에 적합한가

GPU에서 GEMV는 matrix의 각 원소를 읽어 거의 한 번만 곱하므로 arithmetic intensity가 낮다. 계산기가 빠르더라도 HBM에서 weight 또는 KV cache가 도착하기를 기다린다.

PIM은 matrix가 저장된 bank에서 바로 MAC을 수행하므로 외부 bus로 matrix 전체를 보낼 필요가 없다.

### 왜 GEMM에는 항상 유리하지 않은가

GEMM은 matrix tile을 여러 번 재사용하여 arithmetic intensity가 높다. GPU/NPU의 강력한 matrix unit과 cache가 유리할 수 있다.

따라서 heterogeneous system에서는 흔히 다음처럼 분담한다.

```text
GPU/NPU: compute-bound GEMM, FFN, prefill
PIM:     bandwidth-bound GEMV, attention, decode
```

---

## 6. 데이터 매핑이 중요한 이유

PIM은 데이터를 저장한 위치가 곧 연산 위치가 된다. GPU처럼 데이터를 쉽게 다른 core로 가져와 처리한다고 가정하기 어렵다.

### Bank와 Channel 병렬성

모든 channel과 bank에 작업이 균등하게 있어야 높은 내부 대역폭을 활용할 수 있다.

```text
좋은 매핑:
Channel 0 [work]  Channel 1 [work]  Channel 2 [work]  Channel 3 [work]

나쁜 매핑:
Channel 0 [work]  Channel 1 [idle]  Channel 2 [idle]  Channel 3 [idle]
```

한 channel에만 데이터와 작업이 몰리면 다른 channel의 MAC은 놀게 된다. 이때 PIM의 명목상 높은 내부 bandwidth는 의미가 없다.

### Row Locality

같은 DRAM row를 열어 둔 채 여러 연산을 수행하면 ACT/PRE 비용을 줄일 수 있다. 반대로 데이터 배치가 나쁘면 row switching이 잦아진다.

### Reduction

하나의 결과를 여러 bank/channel이 나눠 계산하면 partial result를 합쳐야 한다.

- module 내부 reduction은 비교적 저렴할 수 있다.
- module 간 reduction은 interconnect와 synchronization 비용이 크다.
- 좋은 partition은 channel 활용률을 높이면서 reduction 범위를 제한한다.

---

## 7. PIM Instruction과 Command

PIM 시스템에서는 CPU/GPU instruction 대신, PIM controller가 이해하는 비교적 큰 단위의 instruction을 사용한다.

| Instruction | 역할 |
|---|---|
| `WR-INP` | GPR 또는 host에서 GBuf로 input 이동 |
| `MAC` | DRAM row와 GBuf input으로 dot product 수행 |
| `RD-OUT` | Output Register의 결과를 GPR로 이동 |
| `ACT` / `PRE` | DRAM row 열기 / 닫기 |

하나의 PIM instruction에는 다음 정보가 포함될 수 있다.

- 대상 channel mask
- 반복 횟수
- GBuf/Output Register index
- DRAM row/column address
- source/destination GPR address

Instruction Sequencer는 이를 여러 channel-specific command로 펼쳐 실행한다.

### 정적 실행 모델의 장단점

기존 PIM은 compiler가 command 순서와 주소를 미리 정한다.

장점:

- controller가 단순하다.
- 작은 면적과 전력으로 구현할 수 있다.
- 반복적이고 규칙적인 workload에서 효율적이다.

단점:

- runtime에 길이가 변하는 workload에 대응하기 어렵다.
- 실제 data dependency가 없어도 고정 순서를 따를 수 있다.
- 동적 메모리 할당과 비연속 주소 접근이 어렵다.

---

## 8. LLM Decode와 PIM

LLM decode에서는 새 token 하나를 생성할 때 각 layer에서 다음 작업을 수행한다.

```text
1. projection / FFN: weight GEMV
2. QK^T: query와 전체 K cache의 dot product
3. softmax
4. SV: attention score와 전체 V cache의 weighted sum
5. 새 K/V를 KV cache에 append
```

batch가 작을 때 대부분 GEMV이며, 매 step마다 큰 weight와 KV cache를 읽는다. 이 때문에 decode는 memory bandwidth bound가 되기 쉽다.

### 장문맥에서 PIM의 잠재력

- context가 길수록 읽어야 하는 KV cache가 커진다.
- attention의 arithmetic intensity는 낮아진다.
- PIM은 KV cache가 저장된 위치에서 `QK^T`와 `SV`를 수행할 수 있다.
- 외부로는 query, score, 최종 output처럼 상대적으로 작은 데이터만 이동한다.

### 장문맥에서 새로 생기는 문제

장문맥이 PIM에 유리하기만 한 것은 아니다.

- 요청 하나의 KV cache가 커져 batch size가 감소한다.
- 요청마다 context length가 달라 channel 간 작업량이 불균형해진다.
- attention의 작은 head dimension 때문에 input/output buffer를 자주 교체한다.
- 최대 context 기준 정적 할당은 막대한 메모리를 낭비한다.

---

## 9. PIM의 주요 병목

### 9.1 Channel Underutilization

작업이 channel 수보다 적거나 불균형하게 배치되면 일부 MAC만 일한다.

원인:

- 작은 batch size
- 요청별 context length 차이
- head 수와 channel 수의 불일치
- pipeline stage bubble

해결 방향:

- head/batch 대신 충분히 긴 token 축으로 partition
- runtime workload를 고려한 mapping
- module 내부 reduction을 활용해 모든 channel 활성화

### 9.2 I/O와 MAC Pipeline Stall

PIM 내부에서도 data movement는 공짜가 아니다.

```text
WR-INP -> MAC -> RD-OUT
```

작은 buffer와 정적 command 순서 때문에 I/O가 MAC을 자주 멈출 수 있다.

해결 방향:

- input/output buffering
- I/O와 compute overlap
- 실제 buffer dependency를 추적하는 dynamic scheduling
- 열린 DRAM row의 재사용

### 9.3 정적 주소와 메모리 낭비

기존 PIM command에 physical address와 반복 횟수가 고정되면, 길이가 변하는 KV cache를 다루기 어렵다.

해결 방향:

- runtime loop bound
- virtual-to-physical address translation
- chunk/page 단위 lazy allocation
- module 내부 dispatcher 또는 lightweight MMU

### 9.4 제한된 Programmability

PIM 연산기는 면적과 전력 제약 때문에 단순하다.

- 복잡한 control flow에 불리하다.
- 지원하는 operation과 precision이 제한적이다.
- compiler와 data layout에 강하게 의존한다.
- PIM-friendly하지 않은 연산은 CPU/GPU/NPU로 보내야 한다.

---

## 10. PIMphony로 연결하기

PIMphony의 세 기법은 위 병목에 정확히 대응한다.

| 기존 PIM 문제 | PIMphony 기법 | 핵심 아이디어 |
|---|---|---|
| head/batch 병렬성 부족과 channel idle | Token-Centric PIM Partitioning (TCP) | 긴 token 축을 모듈 내 모든 channel에 분산 |
| 정적 `WR-INP -> MAC -> RD-OUT`의 stall | Dynamic PIM Command Scheduling (DCS) | buffer entry별 실제 dependency를 추적해 I/O와 MAC overlap |
| 최대 context 기준 KV cache 예약 | Dynamic PIM Access (DPA) | runtime loop/address 생성, VA-to-PA 변환, lazy allocation |

### TCP를 이해하는 관점

기존 방식은 head 또는 request를 channel에 하나씩 배치한다. 장문맥에서는 batch가 작아 channel을 채우기 어렵다. TCP는 한 request의 긴 token sequence 자체를 여러 channel에 나눈다.

### DCS를 이해하는 관점

기존 controller는 모든 command 사이에 보수적인 간격을 둔다. DCS는 GBuf/OBuf entry 단위로 hazard를 추적해, 서로 독립적인 input write, MAC, output read를 겹친다.

### DPA를 이해하는 관점

GPU의 PagedAttention과 목적은 비슷하지만 구현 위치가 다르다. PIM에서는 연산 command가 DRAM 주소를 직접 포함하므로, module 내부에서 주소를 변환하고 동적 command를 생성해야 한다.

---

## 11. GPU, NPU, PIM 비교

| 특성 | GPU | NPU | PIM |
|---|---|---|---|
| 핵심 목표 | 범용 병렬 처리 | dense tensor 연산 효율 | 데이터 이동 최소화 |
| 강점 | 유연한 programming, 높은 compute | 높은 TOPS/W | 높은 내부 memory bandwidth |
| 대표 연산 | 다양한 kernel, GEMM | 정적 shape GEMM | memory-bound GEMV |
| 데이터 위치 | HBM에서 core로 이동 | memory에서 array로 이동 | DRAM bank 근처에서 연산 |
| 동적 workload | 강함 | 약함 | 전통적으로 약함 |
| 주요 병목 | HBM bandwidth, occupancy | shape, tiling, graph compile | channel utilization, mapping, I/O scheduling |

PIM은 GPU나 NPU를 완전히 대체하기보다, memory-bound 연산을 맡는 **heterogeneous accelerator**로 사용하는 경우가 많다.

---

## 12. 논문을 읽을 때 볼 질문

PIM 기반 시스템 논문을 읽을 때 다음을 확인하면 설계를 빠르게 파악할 수 있다.

1. **어떤 연산을 PIM에 배치하는가?**
2. **데이터를 channel/bank/row에 어떻게 매핑하는가?**
3. **input과 output을 어디에 buffer하며 I/O와 MAC을 겹치는가?**
4. **partial result를 어디서 reduction하는가?**
5. **runtime 길이, batch 변화, 비연속 메모리를 어떻게 처리하는가?**
6. **실제 hardware인지 simulator인지, area/power overhead를 포함하는가?**
7. **baseline과 memory capacity 및 bandwidth 조건이 공정한가?**

---

## Related Notes

- [GPU](./gpu.md)
- [NPU](./npu.md)
- [KV Cache](../inference/kv-cache.md)
- [LLM Parallelism](../parallelism/llm-parallelism.md)
- [PIMphony](../../notes/hpca26-pimphony.md)
