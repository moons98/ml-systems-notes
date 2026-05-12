---
title: "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference (HeteroInfer)"
venue: SOSP 2025
authors: Le Chen, Dahu Feng, Erhu Feng, Yingrui Wang, Rong Zhao, Yubin Xia, Pinjie Xu, Haibo Chen
affiliations: SJTU IPADS, Tsinghua, SenseTime Research
tags: [llm-inference, mobile, npu, gpu, heterogeneous-compute, edge-ai, quantization]
read_date: 2026-05-11
status: read
url: https://doi.org/10.1145/3731569.3764808
pdf: ../pdfs/sosp25-heteroinfer.pdf
---

## TL;DR

모바일 SoC의 GPU+NPU를 **동시에** 쓰는 첫 산업급 LLM 추론 엔진. 기존 엔진은 GPU-only 또는 NPU-only만 썼는데, NPU는 systolic array의 weight-stall 특성 때문에 텐서의 **순서·모양·크기**에 민감해서 raw TFLOPS만 보면 안 됨. GPU를 NPU의 약점(짧은 시퀀스, shape-misalign, 동적 길이)을 메우는 보조 컴퓨팅 + 추가 메모리 채널로 활용하고, UMA 기반 마이크로초급 동기화로 GPU-NPU 병렬을 실제로 작동시킴. W4A16(weight-only)만 써서 정확도 손실 없이 SOTA 대비 **1.34~6.02× 종단 속도**, Llama-8B prefill 247 tok/s, 디코딩 14 tok/s 달성.

## Key Insights

1. **NPU의 raw TFLOPS는 거짓말이다 — tensor-sensitive performance.** Systolic array + weight-stall 패러다임 때문에 (1) 타일 크기에 안 맞으면 stage 단위로 latency가 계단식, (2) `[M,K]×[K,N]` vs `[N,K]×[K,M]ᵀ`는 같은 연산량이지만 **최대 6× 차이**, (3) input row > column이어야 weight-stall이 유리. 따라서 동일 op도 phase·shape마다 GPU에 떨어뜨려야 더 빠를 수 있음.
2. **단일 가속기는 SoC 메모리 대역폭을 못 채운다.** UMA여도 GPU 단독은 40~45 GB/s만 쓰고(이론 68 GB/s, 실측 max 61.9), GPU+NPU 동시 실행하면 59.5 GB/s(=96%)까지 올라감. → 디코딩(memory-bound)에서 더 빠른 한쪽만 쓰는 게 아니라 **양쪽을 같이 돌리는 게 이득**이라는 비직관적 결과.
3. **NPU의 정적 그래프 제약을 회피하는 두 가지 partition.** Activation-centric: 시퀀스 길이를 표준 청크(예: 256)들로 쪼개 NPU로, 나머지 dynamic 잔여만 GPU로 — padding(평균 1.91× 오버헤드)이나 online graph 생성(시퀀스 1000에서 2초+) 회피. Weight-centric: weight를 row 차원으로 partition해서 NPU에 안 맞는 shape도 둘이 나눠 처리.
4. **400 μs `clFinish` 대신 sleep-then-poll 동기화.** LLM의 레이어가 동형이라 kernel 대기 시간이 예측 가능 → 예상치만큼 자고, `usleep` 최소 단위(80~100 μs)는 너무 거칠어 깬 뒤 출력 텐서 끝의 flag 비트를 폴링. 디코딩에서 동기화 켜면 **4.01× 차이**(커널 자체가 수백 μs라).
5. **W4A16의 위치 — 정확도 vs 속도 trade의 sweet spot.** Activation까지 INT 양자화하는 llm.npu/Qualcomm-AI는 소형 모델에서 20%+ 정확도 손실, mixed-precision은 입력 데이터셋 의존. W4A16은 weight만 INT4 저장하고 compute는 FP16 → NPU FLOAT 성능을 활용하면 같은 정확도로 INT 기반 솔루션과 동등 이상 속도.

## Background

논문 이해에 필요한 일반 개념. 재사용 가능한 건 `concepts/`로 빼고 link.

- **NPU 전반 (systolic array, weight-stall, 정적 그래프 제약, NPU-1/2/3 특성)** → see [concepts/npu.md](../concepts/npu.md). 이 논문이 §3에서 분류한 세 특성과 transpose trick이 그 노트의 출처.
- **UMA (Unified Memory Architecture)** → see [concepts/uma.md](../concepts/uma.md). 이 논문의 Memory-1 (BW 공유 정량) + §4.3 fast sync (memory pool, sleep+poll 우회)가 그 노트의 모바일 사례 출처.
- **W4A16 (weight-only quantization)** — weight는 INT4 저장, 컴퓨트 직전에 FP16으로 dequant. AWQ/GPTQ 계열이 대표적. activation은 FP16 유지 → 정확도 손실 거의 없음. _(TODO: 두 번째 양자화 논문 만나면 추출)_
- **Prefill vs decoding (TTFT, TPOT)** — prefill: 입력 시퀀스 batch matmul → compute-bound, TTFT 좌우. decoding: auto-regressive matrix-vector → memory-bound, TPOT 좌우. 두 phase의 병목이 달라서 최적화 전략도 달라야 함. _(TODO: 두 번째 LLM 추론 논문 만나면 추출)_

## Walkthrough

### §2.3 Limitations of existing mobile inference engines

논문이 풀려는 두 gap:

- **GPU-NPU 병렬 부재**: MLC/MNN은 GPU-only, PowerInfer/llm.npu는 NPU-only. 동기화·partition 어려움이 원인.
- **정확도 유지하며 NPU 풀가동 부재**: llm.npu/Qualcomm-AI는 INT4/8 activation 양자화로 ~20% 정확도 하락. PowerInfer는 FLOAT 시도했지만 NPU FLOAT 성능 활용이 비효율.

### §3 Performance Characterization (논문의 알짜)

| Char | 내용 | 시사점 |
|---|---|---|
| **GPU-1: Linear** | 소형 텐서는 memory-bound, 임계 넘으면 compute-bound로 평탄화 — 단조 증가 | GPU는 어떤 shape든 "예측 가능"하게 동작 → flexible 파트너로 적합 |
| **GPU-2: High-cost sync** | `clEnqueueWriteBuffer` 데이터 복사 + `clFinish` 고정 ~400 μs (드라이버 상태 동기화) | 디코딩 커널 자체보다 동기화가 더 비싸짐 → §4.3 fast sync 동기 |
| **NPU-1: Stage** | 32×32 systolic array tile에 안 맞는 차원은 padding → tile 경계까지 latency 동일(계단) | sequence length가 작거나 비표준이면 NPU가 GPU보다도 느려질 수 있음 |
| **NPU-2: Order-sensitive** | `[M,K]×[K,N]` vs `[N,K]×[K,M]ᵀ`: 연산량 같지만 6× 차이. 큰 쪽이 stall된 weight면 메모리 로드 폭증 | matmul 호출 시 invariant `[A,B]×[B,C] ≡ ([C,B]×[B,A])ᵀ`로 항상 weight를 stationary 쪽에 둠 |
| **NPU-3: Shape-sensitive** | input row > column이어야 weight-stall이 효율적. column 크면 weight tensor도 커져 stall 깨짐 | FFN-down처럼 차원 축소 matmul은 NPU 이점 0.5~1.5×로 제한적 → GPU 분담 필요 |
| **Memory-1: Underutilized BW** | 단일 프로세서 40~45 GB/s, GPU+NPU 동시 60 GB/s (이론 68, 실측 max 61.9). 75/25 split 최적 | 디코딩(memory-bound)에서 "더 느린 쪽도 같이 돌리기"가 정당화 |

### §4 Design

**원칙**: prefill=throughput 극대화, decode=memory bandwidth 극대화. CPU는 control plane 전용(에너지 효율 낮고 일반 작업과 충돌).

#### §4.1 Layer-level GPU-NPU execution

오퍼레이터 affinity 기반 정적 배치:
- **Prefill**: Matmul → NPU (FP 큰 시퀀스). RMSNorm/SwiGLU → GPU. NPU-2 보정 위해 weight·input 순서 swap.
- **Decoding**: Matmul이 matrix-vector(작은 seq) → NPU-1 stage performance 때문에 GPU가 primary.

#### §4.2 Tensor-level GPU-NPU parallelism (핵심 기여)

Layer-level이 못 푸는 3가지 — NPU shape degradation, BW 미활용, 정적 그래프 생성 비용 — 를 텐서 단위로 쪼개 해결.

1. **Weight-centric (static shape)**: weight를 row 차원으로 `gpu_size : npu_size`로 split → 두 sub-tensor가 각자 backend에서 동시 실행, 결과 merge. 시퀀스 길이 고정인 경우.
2. **Activation-centric (dynamic shape)**: 시퀀스를 표준 사이즈 청크(예: 256)들과 잔여 청크로 분할. 표준 청크는 NPU에서 순차, 잔여는 GPU 병렬. → padding(평균 1.91× 오버헤드)도 online graph 생성(seq 1000에서 ~2 s)도 회피.
3. **Hybrid**: activation-centric의 잔여가 너무 작아 GPU도 NPU도 모두 underutilize되면, 잔여를 padding으로 표준 사이즈로 키운 뒤 weight-centric으로 GPU/NPU에 나눠줌.

#### §4.3 Fast synchronization

- **UMA memory pool**: 입출력 텐서를 host↔device 공유 매핑된 슬롯에 고정 할당, 레이어 간 재사용 → 복사·재매핑 제거.
- **Sleep + poll**: 레이어 동형이므로 GPU/NPU 커널 대기 시간 예측 가능 → 그만큼 자고, `usleep` 80~100 μs 단위 한계 보완 위해 깬 뒤 출력 텐서 말단의 flag bit를 small core가 폴링. `clFinish` 400 μs를 수 μs로.
- Prefill은 NPU-dominant라 GPU를 NPU 안에 숨김; decoding은 GPU-dominant라 NPU를 GPU 안에 숨김 — 두 phase에서 동기화 방향이 반대.

#### §4.4 Profiler + Solver + Engine

- **Profiler** (offline, <20 min): LLM에 등장하는 weight shape × 표준 시퀀스 길이 조합만 측정 → 탐색 공간 축소.
- **Solver** (offline): operator별 모든 partition 전략 enumerate, GPU/NPU 실행시간 + sync + copy 합산 minimize. 비표준 시퀀스 길이는 GPU-1 linearity + NPU-1 stage 계단 함수로 보간.
- **Engine** (online): 시퀀스 길이 보고 GPU-only / NPU-only / 3가지 partition 중 선택.

### §5 Evaluation (놀라웠던 숫자)

**End-to-end (Snapdragon 8 Gen 3)**: SOTA 대비 **1.34~6.02×**. 디코딩 위주(BELLE multiturn) Llama-7B에서 PowerInfer-2(sparse model) 대비도 1.34× — sparse가 random small access로 BW 깎아먹어서.

**Prefill**: Llama-8B 247 tok/s, **InternLM-1.8B 1092 tok/s** (llm.npu 564 tok/s의 1.93×). Hetero-tensor가 Hetero-layer 대비 평균 30.2% 추가 향상.

**Dynamic seq length**: Online-prepare 대비 2.24×, Padding 대비 2.21×, NPU-pipe 대비 1.35× (시퀀스 525 기준). Online-prepare는 시퀀스 1000에서 그래프 생성에만 2050 ms.

**Decoding**: Llama-8B 14 tok/s, **InternLM-1.8B 51.1 tok/s**. GPU+NPU 동시 → BW가 43.3 → **59.5 GB/s (이론 max의 96%)**, 사실상 천장 도달.

**Fast sync 효과**: prefill +15.8~24.3%, **디코딩 최대 4.01×**.

**Ablation (Llama-8B prefill seq 320)**: naïve NPU는 GPU보다 62.5% 느림 → activation-centric 2.05× → order swap 1.79× → weight-centric 1.20× → fast sync 1.19×. **각 최적화가 독립적으로 곱해짐**.

**Game interference (Wild Rift 동시 실행)**: GPU-only는 FPS 0으로 추락(OpenCL이 GPU 제출큐 포화). Hetero-tensor는 prefill -2.2%, decode -17.7%, **FPS 무영향**.

**Energy**: Hetero-tensor가 GPU-only 대비 55%, Hetero-layer 대비 12.8% 적게 씀 (빠른 종료 덕).

### §6 Discussion (저자가 던지는 가속기 설계 메시지)

저자가 미래 SoC에 제안: (1) GPU·NPU 통합 스케줄러 (현재는 한 쪽 task가 다른 쪽 블로킹 가능), (2) coherent memory + 통합 API (지금은 OpenCL·QNN 분리), (3) lightweight HW 동기화 primitive.

## Open Questions / Critique

- **Vendor lock-in이 심함.** Qualcomm Hexagon NPU + QNN API + OpenCL에 깊게 묶임. Apple Neural Engine / MediaTek APU는 dataflow가 다르고 API 공개도 제한적 — "minimal retuning으로 포팅 가능"이라는 주장 검증 안 됨. NPU-1,2,3 특성이 다른 systolic 변형(예: NVDLA, Apple ANE의 vector engine)에서도 유효한가?
- **W4A16 의존.** Activation 양자화를 의도적으로 배제했는데, 모델이 더 커지거나 vision-language·MoE로 가면 activation memory가 새 병목이 됨. 활용 못하는 NPU INT TOPS가 큰데(34 TOPS vs FP16 17), 이걸 정확도 손실 없이 쓰는 길은?
- **17.7% decode 페널티는 어디서 문제가 되나?** 게임 같은 GPU-heavy 동시작업 시나리오에 대한 정량은 있지만, AR/카메라 pipeline 같이 GPU rendering 지연이 더 치명적인 경우의 SLO는?
- **Solver가 fully offline.** 실기기는 thermal throttling, battery 모드별 DVFS, 다른 앱의 GPU/NPU 점유율에 따라 동적으로 partition ratio가 바뀌어야 유리할 텐데, 모든 결정이 사전 고정. Online adaptation의 cost vs 이득은?
- **Llama 계열 dense 모델로만 평가.** MoE(routing이 dynamic) / SSM(상태 운반) 같이 텐서 모양과 데이터 흐름이 다른 아키텍처에서 동일 partition 가정이 깨지지 않는지?
- **"Snapdragon 8 Elite에서 디코딩 향상 거의 없음"** — 같은 메모리 BW(68 GB/s)에 묶여 있어서. 디코딩은 이미 천장에 닿아서 다음 세대의 BW 증가가 없으면 더 빨라질 여지가 없다는 뜻 → 이 논문은 **mobile decoding의 사실상 천장을 짚었다**고 볼 수도 있음.
