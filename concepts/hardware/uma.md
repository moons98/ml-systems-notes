---
title: UMA (Unified Memory Architecture)
description: 모든 프로세서(CPU/GPU/NPU/...)가 같은 물리 메모리·주소공간 공유. 데이터 복사 없지만 BW 공유 + 일관성 관리 비용이 새 병목.
tags: [hardware, memory, soc, heterogeneous-compute]
type: concept
related_papers:
  - sosp25-heteroinfer
---

# UMA (Unified Memory Architecture)

## TL;DR

- **UMA = 모든 프로세서가 같은 물리 메모리 + 같은 주소공간 공유**. 모바일·엣지 SoC의 표준.
- 장점: **데이터 복사 비용 0** (CPU↔GPU↔NPU 사이 전송 불필요).
- 새 병목: **BW가 공유 자원**, **캐시 일관성 관리 비용**, **API 파편화**.
- 핵심 trade-off: 전송 비용 ↓ vs BW 경쟁 ↑.

## 1. UMA란

**정의**: 시스템 내 모든 프로세서가
1. **같은 물리 메모리**에 접근하고,
2. **같은 가상 주소**가 같은 물리 위치를 가리킴.

대비되는 두 모델:

| | UMA | Discrete | NUMA |
|---|---|---|---|
| 메모리 위치 | 단일 풀 | 가속기별 분리 (예: GPU VRAM) | 분산되지만 공유 가능 |
| 접근 비용 | 모든 프로세서가 비슷 | 가속기 자기 메모리 빠름 | 노드별 다름 (가까운 ↔ 먼) |
| 전송 필요 | 없음 | 필수 (PCIe, NVLink) | 가능 (자동 또는 명시) |
| 대표 예 | Apple M-series, 모바일 SoC | NVIDIA RTX/H100, AMD Radeon | 멀티 소켓 서버 |

UMA가 모바일·엣지에서 우세한 이유:
- **전력**: 별도 메모리 칩 + 전송 인터페이스 = 추가 전력 소비
- **면적·비용**: 단일 LPDDR 패키지가 더 경제적
- **지연**: PCIe 같은 인터커넥트 셋업 비용 회피

## 2. 하드웨어 구조와 메모리 모델

```
        CPU         GPU         NPU
       ┌───┐       ┌───┐       ┌───┐
       │L1 │       │L1 │       │SRAM│   ← 사적 (프로세서별)
       │L2 │       │L2 │       └───┘
       └─┬─┘       └─┬─┘         │
         └───┬───────┴───┬───────┘
             ↓           ↓
        ┌──────────────────────┐
        │  Memory Controller   │       ← 공유
        │  + 채널 (LPDDR 등)    │
        └──────────┬───────────┘
                   ↓
        ┌──────────────────────┐
        │   DRAM (단일 풀)      │
        └──────────────────────┘
```

| | 공유 | 사적 |
|---|---|---|
| 무엇 | 물리 메모리, 주소공간, 메모리 컨트롤러, BW 채널 | 각 프로세서의 캐시, MMU/TLB, 레지스터 |
| 함의 | 한 프로세서의 read/write가 다른 프로세서에 즉시 보이지 않을 수 있음 (캐시 때문) | 같은 데이터의 여러 사본이 동시 존재 가능 → 일관성 문제 |

## 3. 캐시 일관성 스펙트럼 (★)

UMA의 가장 중요한 디자인 결정. **HW가 일관성을 어디까지 책임지냐**에 따라 세 단계:

### 3.1 Fully coherent
- HW가 자동으로 캐시 간 동기화 (snoop, invalidation)
- SW는 일관성을 신경 쓸 필요 없음 — 그냥 read/write
- 비용: snoop traffic, 추가 회로
- 예: **Apple Silicon CPU↔GPU** (전용 coherency fabric), x86 CPU 간

### 3.2 Partial / scoped coherent
- 특정 buffer 타입이나 메모리 영역만 coherent
- 명시적으로 "coherent buffer" 할당
- non-coherent 접근이 가능하면 더 빠름 (snoop 없음)
- 예: **OpenCL SVM (fine-grained vs coarse-grained)**, Vulkan `HOST_COHERENT` 플래그

### 3.3 Non-coherent
- HW가 일관성 보장 안 함
- SW가 명시적으로 **flush** (캐시 → 메모리) 또는 **invalidate** (메모리 → 캐시) 호출
- 빠르지만 실수하면 stale data 읽음
- 예: **모바일 NPU 다수**, 임베디드 DMA buffer

```
빠름 ←─────────────────────────────────→ 안전
non-coherent  partial coherent  fully coherent
(SW 책임)     (영역별)          (HW 자동)
```

## 4. 프로그래밍 모델

UMA HW 위에 SW가 메모리를 어떻게 다루는지의 추상화:

| 모델 | 의미 | 대표 API |
|---|---|---|
| **SVM (Shared Virtual Memory)** | 같은 포인터를 host·device 양쪽에서 사용 | OpenCL 2.0+ |
| **UVA (Unified Virtual Addressing)** | host·device 주소공간 통합 (변환 자동) | CUDA 4+ |
| **Managed Memory** | 페이지 단위 자동 마이그레이션 | CUDA Managed, HMM |
| **Buffer mapping** | 기존 buffer를 다른 프로세서에 명시 매핑 | OpenCL `clEnqueueMapBuffer`, Vulkan `vkMapMemory` |
| **Pinned / page-locked** | 페이징 막아 DMA 가능하게 | CUDA `cudaHostAlloc`, Linux `mlock` |

OpenCL SVM의 세 단계 (UMA 환경에서 의미 차이 큼):
- **Coarse-grained buffer SVM**: 명시적 map/unmap, 그 사이에만 동기화
- **Fine-grained buffer SVM**: sync point에서 자동 동기화 (atomic 사용 가능)
- **Fine-grained system SVM**: 어떤 host 포인터든 공유 — full coherency 필요

→ Discrete GPU에서는 SVM이 시뮬레이션 (실제로는 page migration). UMA에서는 진짜 zero-copy.

## 5. 이기종 프로세서 간 통신

### 5.1 데이터 공유 (zero-copy)

이상적 사용:
1. CPU가 buffer 할당 (host 가시 영역)
2. GPU/NPU에 같은 buffer 매핑 (포인터 그대로 사용)
3. 처리 후 결과를 CPU가 그대로 읽음
4. 복사 없음

**"No copy ≠ no overhead"** — 잠재 비용:
- **캐시 flush/invalidate**: non-coherent에서 필요
- **메모리 layout 변환**: GPU는 swizzled/tiled, CPU는 linear → 변환 시 실제로는 copy 발생
- **Alignment**: 가속기는 종종 64B/128B alignment 요구
- **Ownership transfer**: API가 "지금 누가 쓰고 있는지" 추적 필요

### 5.2 동기화 메커니즘

비용 스펙트럼:

| 메커니즘 | 비용 | 용도 |
|---|---|---|
| **HW atomics** (compare-and-swap, fetch-add) | ~ns | 락-프리 자료구조, 작은 카운터 |
| **Memory barriers / fences** | sub-μs | reordering 방지 |
| **API events / semaphores** | μs ~ 수십 μs | 커널 간 의존성 |
| **Blocking sync** (clFinish, vkQueueWaitIdle) | 수십 ~ 수백 μs | 모든 작업 완료 대기 |
| **Polling** (busy wait on flag) | μs (커스텀) | predictable timing 시 |

모바일 NPU 컨텍스트에서 LLM 디코딩 커널이 수백 μs 단위라 blocking sync 비용이 커널 자체보다 비싸지는 일이 흔함 (HeteroInfer는 Snapdragon에서 `clFinish` ~400 μs 측정).

### 5.3 메모리 ordering

같은 주소공간 공유해도 **이벤트 순서**는 별도 문제.

producer-consumer 패턴:
```
Producer (GPU):                    Consumer (NPU):
  1. data 쓰기                       1. ready_flag 읽기 (true 대기)
  2. memory fence (write → flag)     2. acquire fence (flag → data)
  3. ready_flag = true              3. data 읽기
```

핵심:
- **release** (producer): 모든 이전 write가 fence 이후에 보이도록 보장
- **acquire** (consumer): 이후 read가 fence 이전 write를 보도록 보장
- HW atomic의 acquire/release semantics 또는 명시 fence 사용

UMA여도 producer가 아직 캐시에 데이터를 들고 있으면 consumer는 못 봄. fence가 캐시 flush를 트리거.

## 6. 성능 특성

### 6.1 BW가 공유 자원
- 단일 프로세서로는 메모리 BW의 일부만 사용 (지연·동시성 한계)
- 여러 프로세서 동시 실행 → 합산 BW 더 높음
- 이론 peak는 닿기 어려움 (메모리 컨트롤러 효율, refresh, 채널 충돌)

### 6.2 캐시 일관성 오버헤드
- snoop traffic: 한 프로세서가 write할 때 다른 프로세서 캐시 확인
- 워크로드가 서로 다른 데이터 영역을 쓰면 무시 가능
- false sharing: 같은 cache line의 다른 워드를 다른 프로세서가 쓰면 ping-pong

### 6.3 Memory contention
UMA의 "공유"는 **다른 시스템 활동**과도 공유:
- GPU rendering (게임, UI)
- 카메라 ISP (이미지 처리 파이프)
- 비디오 코덱 (HW decoder)
- 다른 앱

→ ML 가속기가 단독으로 BW 다 못 쓰는 게 사실 좋음 (시스템 반응성 유지).

## 7. UMA vs Discrete trade-off

| | UMA | Discrete |
|---|---|---|
| 데이터 전송 | 없음 | PCIe 4.0 ~32 GB/s, NVLink ~600 GB/s |
| 전송 셋업 비용 | 없음 | DMA descriptor + 인터럽트 |
| 메모리 BW (가속기 측) | 시스템 LPDDR ~50~100 GB/s | HBM ~1~3 TB/s |
| 메모리 용량 | 시스템 전체 (~수~수십 GB) | 가속기별 (~10~80 GB) |
| 동기화 비용 | μs ~ 수백 μs (API 의존) | DMA 완료 인터럽트 (~수 μs) |
| 적합한 시나리오 | 모바일·엣지, 작은 모델 빈번 통신 | 대형 모델 학습, GPU 집중 워크로드 |

**판단 기준**:
- 데이터 전송 빈도 / 전송량이 크면 → UMA 유리
- 가속기 단일 워크로드의 BW 요구가 크면 → Discrete + 전용 HBM 유리
- 다중 가속기 협업이 필요하면 → UMA 유리

## 8. 대표 구현

| 플랫폼 | Coherency | 특징 |
|---|---|---|
| **Apple Silicon (M, A)** | CPU↔GPU fully coherent | "Unified Memory" 마케팅. ANE는 더 격리됨 |
| **Qualcomm Snapdragon** | partial (CPU↔GPU OpenCL SVM), NPU 별도 | OpenCL + QNN API 분리 |
| **MediaTek Dimensity** | partial | APU(NPU)는 자체 메모리 매핑 |
| **NVIDIA Tegra/Orin** | UVA + 일부 coherent | 자동차·로봇용 |
| **AMD APU (Ryzen+Radeon)** | partial coherent | HSA(Heterogeneous System Arch) 표준 |
| **Intel iGPU (Xe)** | partial coherent | OneAPI Level Zero |

모바일 ↔ 데스크톱·서버: 동일 패러다임이지만 BW 절대치 차이 큼 (모바일 ~60 GB/s vs 워크스테이션 APU ~100 GB/s vs HBM 가속기 ~1 TB/s+).

## 9. 실무 패턴 / 함정

- **"No copy ≠ no overhead"** — 캐시 flush, alignment, layout 변환 비용 잔존. 항상 측정.
- **API 파편화** — UMA여도 OpenCL/QNN/Metal/Vulkan이 분리 → 같은 buffer를 여러 API에서 쓰려면 추가 매핑 필요. 모바일 SoC에서 가장 큰 friction.
- **메모리 layout 차이** — texture(swizzled, NHWC) vs linear(NCHW). 변환 시 사실상 copy 발생.
- **BW 경쟁** — ML 워크로드가 게임·카메라·비디오와 같은 채널 사용. SLO 충돌 시 throttling 또는 priority 필요.
- **HW vs SW coherency 가정 실수** — non-coherent 영역에서 flush 빠뜨리면 stale data 읽음. 디버깅 가장 어려운 종류.
- **Fast sync 패턴** — blocking API sync(수백 μs)가 비싸면, predictable timing + sleep + flag polling으로 수 μs까지 줄임 (HeteroInfer 등에서 사용).
- **Memory pool 재사용** — buffer 할당/해제·매핑 자체가 비쌈. host↔device 매핑된 슬롯을 미리 잡아 레이어 간 재사용.
- **Pinned region 한도** — OS는 pin 가능한 메모리를 제한. 너무 많이 잡으면 시스템 불안정.
