---
title: PIM (Processing-in-Memory)
description: 메모리 내부 또는 가까이에 연산기를 배치해 데이터 이동을 줄이는 가속기 구조. DRAM 계층, PIM의 종류와 구성요소, 연산 매핑, 데이터 배치, 본질적 한계를 처음 배우는 관점에서 정리한다.
tags: [hardware, accelerator, pim, dram, memory-bandwidth]
type: concept
---

# PIM (Processing-in-Memory)

## TL;DR

- **PIM은 데이터를 프로세서로 가져오는 대신, 연산을 메모리 가까이 가져간다.**
- DRAM-PIM은 각 channel 또는 bank 근처에 작은 MAC 연산기를 배치해, DRAM 내부의 높은 병렬성과 대역폭을 직접 사용한다.
- 낮은 arithmetic intensity를 가진 GEMV, recommendation, graph processing처럼 **데이터를 많이 읽지만 계산은 적은 작업**에 유리하다.
- PIM 연산기는 GPU보다 단순하고 느리지만, 데이터를 외부 bus로 이동하지 않아 전체 시스템에서 더 높은 throughput과 energy efficiency를 얻을 수 있다.
- 높은 내부 대역폭이 자동으로 높은 성능을 뜻하지는 않는다. **데이터 배치, channel utilization, I/O와 MAC scheduling**이 함께 맞아야 한다.

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

> **bit-slice 저장과 PIM**: 이때 하나의 data word는 여러 chip에 **비트 단위로 쪼개져(bit-slice)** 저장된다. 예를 들어 64-bit word의 bit[7:0]은 chip0, bit[15:8]은 chip1에 있는 식이다. 따라서 **어떤 chip도 혼자서는 완전한 값을 갖지 못하고**, rank의 chip들이 함께 움직여야 의미 있는 데이터가 된다. 이 배치는 일반 DRAM에는 자연스럽지만, **각 연산기가 자기 자리의 데이터만으로 곱셈을 끝내야 하는 PIM에는 그대로 쓸 수 없다**. 그래서 PIM은 보통 한 chip/bank에 완결된 operand가 떨어지도록 다른 layout을 사용한다(§6).

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

PIM이 빠른 핵심 이유도 여기에 있다. ACT로 열린 row 전체(보통 수 KB)는 **매우 넓은 내부 배선**을 통해 row buffer까지 한 번에 올라온다. 하지만 일반 시스템은 이 넓은 row buffer를 좁은 외부 핀으로 **burst 직렬화**해 조금씩 내보낸다(한 번의 READ가 여러 beat에 걸쳐 캐시라인 하나를 전송). 이 직렬화가 곧 외부 대역폭 병목이다. **PIM은 연산기를 row buffer 옆에 두어, 직렬화 없이 넓은 내부 대역폭을 그대로 먹는다.** 이것이 §1에서 말한 "DRAM 내부 병렬성/대역폭 활용"의 구체적 메커니즘이다.

### DDR DIMM과 HBM/PIM module의 차이

위 계층은 일반적인 DDR DIMM을 설명한 것이다. HBM이나 PIM 전용 module은 packaging과 용어가 다를 수 있다.

- HBM stack은 하나의 package 안에 여러 channel 또는 pseudo-channel을 제공한다.
- PIM 전용 module은 내부에 여러 PIM channel, bank, controller, shared buffer를 통합할 수 있다.
- 따라서 PIM 장치의 `module -> channels -> banks` 그림은 일반 DIMM의 물리 계층을 그대로 그린 것이 아니라, **PIM 장치가 노출하는 논리적 실행 자원**을 나타내는 경우가 많다.

---

## 3. PIM의 종류

PIM이라는 이름은 연산기가 메모리와 얼마나 가까운지에 따라 넓게 사용된다.

| 분류 | 연산기 위치 | 특징 |
|---|---|---|
| Processing-Near-Memory (PNM) | memory controller, logic die, module 내부 | xPU보다 메모리에 가깝지만 DRAM array 밖에 있을 수 있음 |
| Processing-Using-Memory (PUM) | memory cell/array 자체 | 셀의 전기적 특성을 이용, 높은 병렬성이나 지원 연산이 제한적 |
| DRAM-PIM | DRAM chip/module 내부의 bank 또는 bank group 근처 | 기존 DRAM 구조와 MAC 연산기를 결합해 내부 대역폭 사용 |

따라서 PIM은 넓게 보면 "xPU보다 메모리에 가까운 연산"을 포함하지만, 모든 PIM이 단순한 외장 xPU인 것은 아니다. **DRAM-PIM은 연산기가 실제 DRAM chip/module 쪽에 통합되어, 데이터를 외부 memory channel로 내보내지 않고 bank 내부 대역폭으로 처리한다.**

이 노트는 주로 DRAM-PIM(과 PNM에 가까운 구조)을 기준으로 설명한다. 이런 구조는 보통 DRAM bank 근처에 작은 vector MAC을 두고, module 수준의 controller가 연산을 조율한다.

---

## 4. PIM의 기본 구성요소

세부 이름과 배치는 PIM architecture마다 다르지만, DRAM-PIM은 대체로 다음 요소를 공통으로 갖는다.

```text
PIM Module
├── Module Controller
│   ├── 명령을 받아 channel별 command로 펼치는 sequencer
│   ├── 작은 register file / scratchpad
│   └── reduction, activation 같은 보조 연산 유닛
│
├── Channel 0
│   ├── 공유 Input Buffer
│   ├── Bank 0 + Vector MAC + Output Register
│   ├── Bank 1 + Vector MAC + Output Register
│   └── ...
├── Channel 1
└── ...
```

### Bank-side 연산기 (MAC)

- 각 bank 가까이에 작은 vector MAC을 두거나, 하나의 MAC이 bank group의 여러 bank를 담당한다.
- weight나 큰 행렬 데이터는 DRAM row에 저장한다.
- input vector를 작은 buffer에 넣고 여러 bank가 동시에 dot product를 수행한다.
- 결과는 Output Register에 누적한 뒤 module의 register나 보조 연산 유닛으로 모은다.

### 공유 Input Buffer

- channel 내 bank들이 공유하는 작은 input buffer다.
- 외부에서 들어온 activation, vector 등을 저장한다.
- 용량이 작아 input을 자주 교체하면 I/O가 병목이 된다.

### Module Controller

- compiler가 생성한 PIM 명령을 받는다.
- 명령을 channel별 command로 펼쳐 여러 channel에 전달(multicast)한다.
- channel 결과를 모으고 reduction, softmax 같은 보조 연산을 수행할 수 있다.

> 실제 논문이나 제품은 이 요소들에 고유한 이름(HUB, global buffer 등)을 붙이지만, 기능은 대체로 위 세 가지로 묶인다.

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
1. 입력 적재: x의 작은 tile을 input buffer에 기록
2. 연산:     각 bank가 자신의 W row와 input tile을 dot product
3. 연산:     K 방향의 다음 tile을 누적
4. 결과 회수: 완성된 결과를 Output Register에서 읽음
```

여러 bank가 서로 다른 row의 dot product를 동시에 계산하므로 큰 GEMV를 병렬화할 수 있다.

여기서 두 데이터의 역할이 **비대칭**이라는 점이 중요하다.

- **x (작음)**: input buffer에 올린 뒤 channel 내 모든 bank에 **broadcast**된다. 작아서 저장 layout이 자유롭다(어떻게 저장돼 있든 읽어 buffer에 모으면 된다).
- **W (큼)**: bank들에 **행 단위로 나눠 저장**되며, 이 layout이 성능을 좌우한다. 외부로 절대 내보내지 않고 bank 안에 머문다.

각 bank는 자기가 맡은 W 행과 broadcast된 x만으로 자신의 출력을 끝까지 계산하므로, 출력을 bank에 나누면 bank 간 의존이 생기지 않는다(§6 참고).

### 왜 GEMV에 적합한가

GPU에서 GEMV는 matrix의 각 원소를 읽어 거의 한 번만 곱하므로 arithmetic intensity가 낮다. 계산기가 빠르더라도 HBM에서 weight가 도착하기를 기다린다.

PIM은 matrix가 저장된 bank에서 바로 MAC을 수행하므로 외부 bus로 matrix 전체를 보낼 필요가 없다.

### 왜 GEMM에는 항상 유리하지 않은가

GEMM은 matrix tile을 여러 번 재사용하여 arithmetic intensity가 높다. GPU/NPU의 강력한 matrix unit과 cache가 유리할 수 있다.

따라서 heterogeneous system에서는 흔히 다음처럼 분담한다.

```text
GPU/NPU: compute-bound GEMM, FFN
PIM:     bandwidth-bound GEMV, attention
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

reduction이 필요한지는 **무엇을 기준으로 나누는가**에 달려 있다.

- **출력(M)축 분할**: 서로 다른 출력을 다른 bank가 맡는다. 각 bank가 완결된 dot product를 수행하므로 **합칠 partial result가 없다**.
- **축약(K)축 분할**: 하나의 출력을 위한 dot product를 여러 bank가 나눠 맡는다. 각자 부분합만 내므로 **마지막에 partial result를 합쳐야 한다**.

partial result를 합쳐야 할 때, 그 비용은 합산 위치에 달려 있다.

- module 내부 reduction은 비교적 저렴할 수 있다.
- module 간 reduction은 interconnect와 synchronization 비용이 크다.
- 그래서 좋은 partition은 보통 **M축으로 먼저 나눠 reduction을 피하고**, 불가피할 때만 K축을 나눠 module 내부에서 합치도록 범위를 제한한다.

---

## 7. PIM의 실행 모델 (정적 명령)

PIM 프로그래밍 모델은 CPU/GPU instruction보다 큰 단위의 연산을 다룬다. 다만 "GEMV를 해라" 같은 큰 명령은 **컴파일러 단계의 추상화**이지, hardware가 그 단위를 통째로 받는 것이 아니다. 컴파일러가 이를 PIM 명령 시퀀스로 낮추고, 전선 위에서는 결국 **일반 DRAM command(ACT/RD/WR, mode 전환을 위한 MRS 등)로 분해**되어 발행된다. (CPU에서 matrix 연산이 컴파일러를 거쳐 load/FMA/store로 쪼개지는 것과 같은 구도다.)

개념적으로 PIM 명령은 보통 다음과 같은 큰 동작을 나타낸다.

| 동작 | 역할 |
|---|---|
| 입력 적재 | host나 register에서 input buffer로 데이터 이동 |
| 연산 (MAC) | DRAM row와 input buffer로 dot product 수행 |
| 결과 회수 | Output Register의 결과를 register/host로 이동 |
| ACT / PRE | DRAM row 열기 / 닫기 |

하나의 PIM 명령에는 대상 channel, 반복 횟수, buffer index, DRAM row/column address 같은 정보가 함께 들어가고, controller가 이를 여러 channel-specific command로 펼쳐 실행한다.

### 명령은 누가 분해하는가

- **컴파일러/드라이버**: 연산을 어느 bank/channel에 어떤 순서·주소로 펼칠지 **미리 결정**한다(똑똑한 부분, 오프라인).
- **host memory controller**: 그 결과로 나온 DRAM command를 channel/rank로 발행한다. PIM이든 아니든 동작 단위는 항상 DRAM command다.
- **PIM 측 로직(또는 module controller)**: mode가 PIM일 때 해당 command를 all-bank로 펼쳐 MAC을 실행한다. host memory controller와는 별개의 주체다.

여기서 두 가지를 구분하면 좋다. **데이터 layout**은 PIM이 결정하지 않는다 — 어느 주소에 쓰느냐와 memory controller의 주소매핑이 각 operand를 어느 bank에 떨어뜨릴지 정한다(§6). **연산 트리거**는 host memory controller가 mode 전환(MRS) 후 발행하고, PIM 로직이 그에 반응해 MAC을 실행한다.

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

## 8. PIM의 본질적 한계

PIM의 장점(데이터 이동 최소화)은 동시에 구조적 제약을 동반한다. 어떤 PIM 설계를 보든 아래 네 가지가 반복해서 등장한다.

### 8.1 Channel Underutilization

작업이 channel 수보다 적거나 불균형하게 배치되면 일부 MAC만 일한다.

원인:

- 작은 batch size
- 작업량이 channel 수에 맞게 나뉘지 않는 경우
- pipeline stage bubble

이를 피하려면 작업을 충분히 많은 단위로 쪼개 모든 channel에 고루 분산하고, reduction 범위를 제한해 channel을 활성 상태로 유지해야 한다.

### 8.2 I/O와 MAC Pipeline Stall

PIM 내부에서도 data movement는 공짜가 아니다.

```text
입력 적재 -> 연산 -> 결과 회수
```

작은 buffer와 정적 command 순서 때문에 I/O가 MAC을 자주 멈출 수 있다. input/output을 buffering하고, 독립적인 I/O와 compute를 overlap하며, 열린 DRAM row를 재사용하는 것이 일반적인 완화 방향이다.

### 8.3 정적 주소와 메모리 낭비

기존 PIM command에 physical address와 반복 횟수가 고정되면, 길이가 변하는 데이터를 다루기 어렵다. 최악의 경우 최대 크기 기준으로 메모리를 미리 잡아 낭비가 커진다. runtime loop bound, 주소 변환, chunk/page 단위 lazy allocation 같은 기법이 이를 완화한다.

### 8.4 제한된 Programmability

PIM 연산기는 면적과 전력 제약 때문에 단순하다.

- 복잡한 control flow에 불리하다.
- 지원하는 operation과 precision이 제한적이다.
- compiler와 data layout에 강하게 의존한다.
- PIM-friendly하지 않은 연산은 CPU/GPU/NPU로 보내야 한다.

---

## 9. GPU, NPU, PIM 비교

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

## 10. 응용 예시: LLM decode와 PIM

PIM이 memory-bound 연산에 유리하다는 점을 보여주는 대표적인 예가 LLM decode다.

LLM decode에서는 새 token 하나를 생성할 때 각 layer에서 weight GEMV, `QK^T`(query와 전체 K cache의 dot product), softmax, `SV`(score와 전체 V cache의 weighted sum)를 수행한다. batch가 작을 때 대부분 GEMV이며, 매 step마다 큰 weight와 KV cache를 읽는다. 이 때문에 decode는 **memory bandwidth bound**가 되기 쉽다.

특히 context가 길수록 읽어야 하는 KV cache가 커지고 attention의 arithmetic intensity는 낮아진다. PIM은 KV cache가 저장된 위치에서 `QK^T`와 `SV`를 수행하고, 외부로는 query·score·최종 output처럼 상대적으로 작은 데이터만 이동시킬 수 있다.

다만 장문맥이 PIM에 유리하기만 한 것은 아니다. KV cache가 커져 batch size가 줄고, 요청마다 context length가 달라 channel 간 작업량이 불균형해지며(§8.1), 최대 context 기준 정적 할당은 메모리를 크게 낭비한다(§8.3). 즉 이 예시는 PIM의 장점과 본질적 한계가 동시에 드러나는 경우다.

---

## 11. PIM 설계를 볼 때 확인할 점

PIM 기반 시스템(논문·제품)을 살펴볼 때 다음을 확인하면 설계를 빠르게 파악할 수 있다.

1. **어떤 연산을 PIM에 배치하는가?**
2. **데이터를 channel/bank/row에 어떻게 매핑하는가?**
3. **input과 output을 어디에 buffer하며 I/O와 MAC을 겹치는가?**
4. **partial result를 어디서 reduction하는가?**
5. **runtime 길이, batch 변화, 비연속 메모리를 어떻게 처리하는가?**
6. **실제 hardware인지 simulator인지, area/power overhead를 포함하는가?**
7. **baseline과 memory capacity 및 bandwidth 조건이 공정한가?**
