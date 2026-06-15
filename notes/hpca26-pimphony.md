---
title: "PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System"
venue: HPCA 2026
authors: Hyucksung Kwon, Kyungmo Koo, Janghyeon Kim, Woongkyu Lee, Minjae Lee, Gyeonggeun Jung, Hyungdeok Lee, Yousub Jung, Jaehan Park, Yosub Song, Byeongsu Yang, Haerang Choi, Guhyun Kim, Jongsoon Won, Woojae Shin, Changhyun Kim, Gyeongcheol Shin, Yongkee Kwon, Ilkon Kim, Euicheol Lim, John Kim, Jungwook Choi
affiliations: Hanyang University, KAIST, SK hynix
tags: [llm-inference, long-context, pim, kv-cache, attention, scheduling, memory-management, mlir]
read_date: 2026-06-10
status: read
url: https://arxiv.org/abs/2412.20166
pdf: ../pdfs/hpca26-pimphony.pdf
---

## TL;DR

- **문제**: PIM은 높은 내부 메모리 대역폭 덕분에 장문맥 LLM decode의 attention을 가속하기 좋지만, 기존 시스템은 문맥이 길어질수록 채널 활용률 저하, 고정된 I/O-연산 순서에 따른 MAC stall, 최대 문맥 길이 기준의 정적 KV cache 할당으로 성능과 용량을 낭비한다.
- **접근**: PIMphony는 토큰 축으로 작업을 모든 채널에 분산하는 **TCP**, 실제 데이터 의존성만 지키며 I/O와 MAC을 겹치는 **DCS**, PIM 내부에서 동적 주소 변환과 lazy allocation을 지원하는 **DPA**를 compiler/runtime/hardware에 걸쳐 공동 설계한다.
- **결과**: cycle-accurate simulation에서 기존 PIM-only 대비 최대 **11.3x**, xPU+PIM 대비 최대 **8.4x** throughput 향상. DPA는 평균 메모리 활용률을 **36.2%에서 75.6%**로 높이며, 1M-token 조건에서는 PIM-only baseline 대비 최대 **46.6x** 향상을 보인다.

---

## Key Insights

1. **장문맥은 batch/head 병렬성을 줄이지만 token 병렬성은 늘린다.**
   - 기존 Head-First Partitioning(HFP)은 head-batch pair를 채널에 배치한다.
   - 문맥이 길어지면 요청 하나의 KV cache가 커져 batch size가 줄고, 서로 다른 요청 길이 때문에 채널 간 load imbalance도 커진다.
   - 반대로 token 축은 문맥과 함께 길어진다. TCP는 한 head의 token들을 모듈 내 모든 채널에 나눠 장문맥일수록 풍부해지는 병렬성을 사용한다.

2. **PIM의 높은 내부 대역폭만으로는 충분하지 않다.**
   - Attention의 작은 head dimension은 input/output reuse가 낮아 `WR-INP -> MAC -> RD-OUT`을 자주 반복하게 한다.
   - 정적 스케줄러는 실제 의존성이 없어도 고정 순서를 지켜 MAC을 멈춘다. 작은 attention 차원에서 MAC utilization은 **14.7%**까지 떨어진다.
   - DCS는 buffer entry 단위 의존성을 추적해 독립적인 I/O와 MAC을 out-of-order로 겹친다.

3. **동적 KV cache 관리는 PIM에도 필요하지만, 주소 변환 위치가 다르다.**
   - GPU의 paged attention과 달리 기존 PIM 명령은 loop count와 물리 주소가 compile time에 고정된다.
   - DPA는 PIM module 내부 dispatcher가 runtime token length와 VA-to-PA table을 이용해 명령과 주소를 생성한다.
   - host는 새 요청, 추가 chunk 할당, 요청 해제 때만 개입하고 매 decode step에는 개입하지 않는다.

4. **세 기법은 독립 최적화가 아니라 병목을 차례로 이동시키는 공동 설계다.**
   - TCP가 채널을 채우면 I/O가 다음 병목이 되고, DCS가 이를 완화한다.
   - Attention이 빨라지면 xPU의 FC 단계가 상대 병목이 되는데, DPA가 더 큰 batch를 허용해 xPU utilization도 높인다.

---

## Background

> PIM 구조와 기본 동작 방식은 [PIM concept note](../concepts/hardware/pim.md) 참고.

### 왜 장문맥 decode가 PIM에 적합한가

- autoregressive decode에서 attention은 매 token마다 전체 KV cache를 읽는다.
- context length와 batch size가 증가할수록 KV cache 용량과 읽기 트래픽이 선형 증가한다.
- 연산 집약도가 낮은 GEMV가 지배적이므로 GPU compute보다 메모리 bandwidth/capacity가 병목이 된다.
- DRAM-based PIM은 각 bank 근처의 vector MAC으로 데이터를 외부로 이동하지 않고 GEMV를 수행한다.

### 대상 시스템

- **PIM-only / CENT 계열**: 여러 PIM node를 CXL 등으로 연결해 전체 모델을 실행한다.
- **xPU+PIM / NeuPIMs 계열**: compute-intensive FC/GEMM은 NPU가, bandwidth-bound attention은 PIM이 실행한다.
- 여러 모듈 간에는 Tensor Parallelism(TP) 또는 Pipeline Parallelism(PP)을 사용한다.

---

## Motivation: 기존 PIM의 세 병목

### 1. Channel Underutilization

기존 HFP는 head와 batch 축을 채널에 나눈다.

- **TP에서** 서로 다른 request 길이 때문에 짧은 요청을 맡은 채널이 먼저 끝나고 기다린다.
- **PP에서** 한 pipeline stage의 request와 연결된 일부 채널만 활성화되어 나머지가 빈다.
- 장문맥에서는 KV cache 용량 때문에 batch size가 작아져 빈 채널에 다른 요청을 채우기도 어렵다.

32K context에서 기존 PIM의 MAC utilization은 4K 대비 **48% 감소**한다.

### 2. I/O Bottleneck

PIM GEMV는 다음 primitive command를 반복한다.

```text
WR-INP -> MAC -> RD-OUT
```

- 작은 attention head dimension은 input/output reuse를 제한한다.
- 기존 controller는 실제 hazard를 추적하지 않고 고정 latency와 명령 순서를 적용한다.
- 독립적인 I/O와 MAC도 직렬화되어 pipeline stall이 발생한다.

### 3. Static Memory Management

- 기존 PIM instruction은 loop count와 operand physical address가 compile time에 결정된다.
- 실제 요청 길이가 달라도 최대 context 기준으로 KV cache를 예약해야 한다.
- workload별 capacity utilization은 **31.0-40.5%**, 평균 **36.2%**에 불과하다.

---

## Design

### 1. Token-Centric PIM Partitioning (TCP)

TCP는 모듈 간 병렬화 방식(TP/PP)은 유지하면서, **모듈 내부에서 한 attention head의 token 축을 모든 채널에 분산**한다.

- `QK^T`: 각 채널이 서로 다른 token 구간의 score를 계산한다. 이후 softmax 과정에서 concatenate하므로 별도 reduction 비용이 거의 없다.
- `SV`: 각 채널이 token 구간별 partial output을 계산하고 PIM HUB/EPU에서 한 번 reduction한다.
- 16-channel, channel당 16-bank 구성에서는 `QK^T`는 256 tokens, `SV`는 32 tokens 이상이면 모든 채널을 활성화할 수 있다.
- SV의 inter-channel reduction은 7B/16K 조건에서 attention latency의 **0.2% 미만**이다.
- 분할이 모듈 내부에만 존재하므로 inter-module synchronization을 추가하지 않는다.

### 2. Dynamic PIM Command Scheduling (DCS)

#### I/O-aware buffering

- 기존 multi-entry GBuf를 input buffer로 활용한다.
- OutReg를 더 큰 Output Buffer(OBuf)로 확장한다.
- 두 buffer를 dual-port로 만들어 MAC 실행 중 다음 input prefetch 또는 이전 output drain을 허용한다.

#### Entry-level dependency tracking

- **D-Table**: 각 GBuf/OBuf entry를 마지막으로 접근한 command ID를 기록하고 새 command의 dependency ID(DID)를 만든다.
- **S-Table**: 마지막 접근 command ID, 완료 예상 cycle, 연속 MAC 여부를 기록한다.
- command를 I/O queue와 compute queue로 분리한다.
- 각 queue 내부 순서는 유지하지만, 두 queue 사이에서는 실제 의존성이 해결된 command를 먼저 발행한다.

예시 command stack은 정적 scheduling의 34 cycles에서 DCS 적용 후 **22 cycles**로 감소한다.

#### GQA와의 관계

GQA에서는 같은 KV row를 여러 query가 공유하므로 열린 DRAM row에서 관련 계산을 모두 수행하면 ACT/PRE를 줄일 수 있다. 하지만 이 row-reuse mapping은 GBuf input 교체를 늘린다. DCS는 이 추가 I/O를 MAC과 겹쳐 GQA의 KV reuse 이득을 실제 성능으로 연결한다.

### 3. Dynamic PIM Access (DPA)

#### Dynamic instructions

- **Dyn-Loop**: 최대 context가 아니라 현재 요청의 실제 token length로 반복 횟수를 결정한다.
- **Dyn-Modi**: loop 안에서 row/column operand를 stride만큼 수정해 virtual address를 생성한다.

기존 명령열 크기는 token length에 선형 증가하지만, DPA instruction은 문맥 길이와 무관하게 거의 일정한 크기를 유지한다.

#### On-module dispatcher

PIM HUB에 lightweight dispatcher를 추가한다.

- instruction buffer: compact DPA instruction 저장
- configuration buffer: request ID, 현재 token index 등 저장
- VA2PA table: 논리 KV 주소를 실제 비연속 physical chunk에 매핑
- decoder: runtime context에 맞춰 최종 PIM instruction 생성

dispatcher는 decode step마다 token length를 자체 증가시키며, instruction decode와 실행을 pipeline한다.

#### Lazy allocation

- host가 요청의 KV cache가 커질 때만 **1MB chunk**를 추가 할당한다.
- chunk는 물리적으로 연속일 필요가 없으며 VA2PA table로 연결한다.
- fragmentation은 요청별 마지막 chunk로 제한된다.

---

## Implementation

- MLIR dialect와 custom pass로 transformer decoder pattern 및 PIM 가능 kernel을 찾고 TCP/DPA metadata가 포함된 명령을 생성한다.
- IREE runtime/HAL을 확장해 상용 PIM SDK와 연결한다.
- NeuPIMs와 CENT의 Ramulator 기반 cycle-accurate simulator에 통합한다.

### Hardware overhead

| Component | Overhead |
|---|---:|
| I/O-aware output buffering | bank MAC area의 0.47% |
| DCS controller | HUB control area 0.5%, power 1.3% |
| DCS metadata | controller당 576B |
| DPA on-module dispatcher | area 4%, 내부 buffer 200KB 미만 |

---

## Evaluation

### Setup

- Models: 7B/72B, GQA 및 non-GQA, 최대 128K context의 실제 benchmark
- Workloads: LongBench, LV-Eval
- Capacity: 7B에 128GB, 72B에 512GB
- Baselines: PIM-only CENT, heterogeneous NeuPIMs
- 추가 scalability 실험: 최대 1M context, 1024GB

### Main results

| Result | Improvement |
|---|---:|
| Non-GQA 32K models | 2.1-4.5x |
| GQA 128K models | up to 11.3x |
| PIM-only overall | up to 11.3x |
| xPU+PIM overall | up to 8.4x |
| Attention acceleration | up to 19x |
| Attention energy reduction | up to 3.46x |

### Capacity and scalability

- DPA는 평균 capacity utilization을 **36.2% -> 75.6%**로 높인다.
- 1M-token, 512GB 조건에서 CENT baseline utilization은 pipeline bubble 때문에 2%까지 하락한다.
- 같은 조건에서 PIMphony는 CENT 대비 **46.6x**, NeuPIMs 대비 **5.0x** 향상된다.
- 짧은 256-token 조건에서도 baseline 대비 **2.1x** 향상을 보여 장문맥 전용 최적화에만 머물지 않는다.

### DCS vs. ping-pong buffering

- ping-pong은 두 buffer region이 모두 idle일 때 ownership을 전환해야 해 hand-off stall이 생긴다.
- DCS는 하나의 buffer에서 entry 단위 hazard를 추적해 같은 buffer 내부에서도 I/O와 MAC을 겹친다.
- 동일 buffer 크기에서 DCS는 compute-unit utilization을 최대 **1.4x** 높인다.

---

## Open Questions / Critique

1. **평가가 simulator 중심이다.**
   - AiMX timing을 반영한 cycle-accurate simulator이지만, DCS와 DPA를 실제 PIM silicon에 구현했을 때의 timing closure, contention, firmware 복잡도는 검증되지 않았다.

2. **DPA의 scheduler/allocator 정책이 단순하다.**
   - 1MB lazy allocation으로 capacity utilization은 크게 개선하지만, 실제 serving의 eviction, prefix sharing, preemption, request migration과 결합한 정책은 다루지 않는다.

3. **TCP가 attention 외 workload에도 일반화되는가?**
   - token 축이 충분히 긴 attention에서는 자연스럽지만, 다른 memory-bound operator에서 reduction 및 data-layout 비용이 어떻게 변하는지는 추가 검증이 필요하다.

4. **PIMphony가 병목을 FC/xPU 쪽으로 이동시킨다.**
   - 논문도 이 현상을 확인하고 DPA로 batch를 늘려 완화한다. 더 긴 문맥이나 더 빠른 PIM에서는 xPU compute 또는 interconnect가 새로운 한계가 될 수 있다.

5. **GPU 비교의 시스템 비용과 programmability 차이**
   - memory-matched A100 대비 이점을 보이지만, PIM 전용 하드웨어 변경과 compiler/runtime 유지 비용까지 포함한 총비용 비교는 없다.

---

## Takeaway

PIMphony의 중요한 메시지는 "PIM은 대역폭이 높다"에서 끝나지 않는다. 장문맥 inference에서 그 대역폭을 실제로 쓰려면 **작업 분할, 명령 scheduling, 메모리 가상화**를 함께 바꿔야 한다. 특히 PIM 내부에 작고 제한적인 MMU-like 기능을 넣어 paged KV cache에 가까운 동적 관리를 가능하게 한 DPA는, 향후 PIM 기반 serving system에서 재사용할 수 있는 설계 방향이다.
