---
title: GPU vs NPU — 구조와 특성 비교
description: GPU(SIMT, warp 기반 latency hiding)와 NPU(systolic array, weight-stall) 설계 철학 비교. 실리콘 투자처 차이, dynamic shape 제약, 워크로드별 적합성.
tags: [hardware, accelerator, gpu, npu, simt, systolic-array, heterogeneous-compute]
type: concept
related_papers:
  - sosp25-heteroinfer
---

# GPU vs NPU — 구조와 특성 비교

## TL;DR

- GPU는 **범용 병렬 프로세서**: SIMT 실행 모델로 어떤 shape·연산 패턴도 처리. 제어 로직에 실리콘을 써서 유연성을 얻음.
- NPU는 **행렬곱 특화 고정 회로**: systolic array로 전 PE를 매 cycle 가동. 제어 로직 대신 MAC에 실리콘을 투자해 W당 TOPS가 한 자릿수 높음.
- 핵심 trade-off: **실리콘 투자처의 차이** → 효율과 유연성의 교환.
- LLM처럼 dynamic shape이 불가피한 워크로드에서는 단독보다 **GPU-NPU 병렬 파티셔닝**이 실질적 대안.

---

## 1. GPU — 범용 병렬 프로세서

GPU는 다양한 병렬 연산을 유연하게 처리하기 위해 설계된 가속기다.

내부는 여러 개의 SM(Streaming Multiprocessor)으로 구성되며, 각 SM은 수백 개의 thread를 **warp**(보통 32개) 단위로 묶어 스케줄링한다. SIMT(Single Instruction, Multiple Threads) 실행 모델로, 한 warp의 thread들이 같은 명령어를 동시에 실행한다.

### 핵심 효율 메커니즘: warp 기반 latency hiding

여러 warp를 동시에 in-flight 상태로 올려두고, 특정 warp가 메모리 접근 등으로 stall되면 스케줄러가 즉시 다른 준비된 warp로 전환해 compute unit을 쉬지 않게 한다. latency를 줄이는 게 아니라 **critical path에서 빼는** 방식.

이 유연성을 위해 GPU는 캐시, 분기 예측기, 동적 스케줄러 등 **제어 로직에 상당한 실리콘을 할애**한다. 덕분에 어떤 shape, 어떤 연산 패턴도 처리할 수 있지만, 그만큼 W당 연산 효율은 낮아진다.

---

## 2. NPU — 행렬곱 특화 고정 회로

NPU는 행렬곱만큼은 매 cycle 모든 연산 유닛을 100% 가동하도록 설계된 가속기다.

내부 핵심은 2D PE(Processing Element) 격자로 구성된 **systolic array**다. 각 PE는 MAC(곱+누적) 유닛과 1-cycle register만 갖는다. GPU가 제어 로직에 쓰는 실리콘을 전부 MAC에 투자한 구조다.

### 동작 방식

`C = A × B` 연산 시:

- **weight(B)를 PE 격자에 preload해 고정(stall)**
- **activation(A)을 skew해서 왼쪽 edge에서 흘려 보냄** — 매 cycle 한 원소씩 각 PE를 통과
- **부분합(partial sum)은 위에서 아래로 전달** — 격자 아래에 도달한 값이 최종 누적 결과

warp divergence도 스케줄링 overhead도 없이 **모든 PE가 매 cycle 동시에 연산**한다.

### Dynamic shape 미지원

NPU의 효율은 연산 계획이 컴파일 타임에 완전히 확정되는 것에서 온다. tile 분할 방식, accumulator slot 할당, DMA 타이밍 — 이 모두가 shape의 함수다. shape이 하나라도 바뀌면 전체 계획이 무효화되어 재컴파일이 필요하다.

또한 PE 격자는 고정 크기(예: 32×32) 단위로만 tile을 처리하므로, 이 크기에 align되지 않는 shape은 padding이 필요하다. 결과적으로 latency가 계단식으로 증가하고, 텐서의 순서·크기에 따라 효율이 크게 달라진다.

---

## 3. 핵심 trade-off: 실리콘 투자처의 차이

| | GPU | NPU |
|---|---|---|
| 실리콘 투자처 | 제어 로직 (캐시, 분기 예측, 스케줄러) | MAC 유닛 |
| 연산 효율 | 낮음 (warp divergence, 제어 overhead) | 높음 (전 PE 매 cycle 가동) |
| W당 TOPS | 기준 | 한 자릿수 높음 |
| Latency hiding | warp 전환으로 숨김 | weight load는 critical path에 노출 |
| Shape 유연성 | 어떤 shape이든 처리 | 정해진 모양만 효율적 |
| 실행 모델 | runtime kernel dispatch | AOT compile 후 고정 시퀀스 실행 |

두 HW의 차이를 유발하는 근본 원인은 하나: **batch preload → peak 효율 ↑, flexibility ↓**.

---

## 4. 워크로드별 적합성

| 상황 | 적합한 HW | 이유 |
|---|---|---|
| 큰 batch / 긴 sequence의 행렬곱 | NPU | weight load 비용이 긴 streaming으로 amortize됨 |
| Decode 단계 (seq=1, 작은 M) | GPU 유리 | M=1이면 NPU tile load 비용이 compute를 압도 |
| Dynamic shape / 조건 분기 | GPU | NPU는 AOT 컴파일 범위 밖의 shape 처리 불가 |
| NPU에 약한 shape의 잔여분 | GPU로 offload | 두 가속기가 각자 잘하는 영역만 담당 |

LLM 추론처럼 shape이 동적으로 변하는 워크로드에서는 NPU 단독보다 **GPU-NPU 병렬 파티셔닝**이 실질적인 대안이 된다.

---