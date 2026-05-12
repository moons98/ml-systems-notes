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

- 모바일 SoC의 GPU+NPU를 **동시에** 쓰는 첫 산업급 LLM 추론 엔진.
- 기존 엔진은 GPU-only 또는 NPU-only. NPU는 systolic array의 weight-stall 특성 때문에 텐서의 **순서·모양·크기**에 민감해서 raw TFLOPS만 보면 안 됨.
- GPU를 NPU의 약점(짧은 시퀀스, shape-misalign, 동적 길이)을 메우는 보조 컴퓨팅 + 추가 메모리 채널로 활용. UMA 기반 마이크로초급 동기화로 GPU-NPU 병렬을 실제로 작동시킴.
- W4A16(weight-only)만 써서 정확도 손실 없이 SOTA 대비 **1.34~6.02× 종단 속도**. Llama-8B prefill 247 tok/s, 디코딩 14 tok/s.

---

## Background

- 당시 SOTA 모바일 추론 엔진(MLC-LLM, llm.npu, PowerInfer-2)은 모두 단일 가속기 사용
- GPU-NPU 동시 실행 시도가 없었던 이유: 동기화 비용·partition 설계의 어려움
- 기존 접근의 두 가지 실패:
  - GPU/NPU-only: 한쪽 가속기만 써서 BW·compute 모두 낭비
  - 정확도 타협: llm.npu·Qualcomm-AI는 INT4/8 activation 양자화로 ~20% 정확도 하락. PowerInfer는 FLOAT 시도했지만 NPU FLOAT 성능 활용이 비효율
- 이 논문의 신규 기여:
  - NPU 성능 함정을 체계적으로 분류 (NPU-1/2/3) — 이 수준의 characterization은 선행 연구 없음
  - Tensor-level heterogeneous parallelism 설계 (activation/weight-centric + hybrid)
  - 모바일 UMA에서 μs급 동기화 실현

---

## Key Insights

1. **NPU의 raw TFLOPS는 거짓말이다 — tensor-sensitive performance.**
   - Systolic array + weight-stall 패러다임 때문에 세 가지 성능 함정이 있음
   - (NPU-1) tile 크기에 안 맞으면 latency가 계단식으로 증가
   - (NPU-2) `[M,K]×[K,N]` vs `[N,K]×[K,M]ᵀ`는 같은 연산량이지만 **최대 6× 차이** (예: M=14336, N=4096)
     - systolic array는 오른쪽 operand를 stationary(weight)로 처리 → 순서가 바뀌면 실제 weight가 streaming 대상이 되어 tile 로드 반복 발생
   - (NPU-3) input row > column이어야 weight-stall이 유리 → 차원 축소 matmul은 NPU 이점 제한적
     - activation row가 많을수록 한 번 로드한 weight tile 재사용 횟수 ↑. FFN-down처럼 K가 크고 seq가 짧으면 tile 로드 비용 대비 재사용이 거의 없음
   - 동일 op도 phase·shape마다 GPU에 떨어뜨려야 더 빠를 수 있음

2. **단일 가속기는 SoC 메모리 대역폭을 못 채운다.**
   - GPU 단독: 40~45 GB/s (이론 68, 실측 max 61.9 GB/s)
   - GPU+NPU 동시: **59.5 GB/s (이론 max의 96%)** — 사실상 천장 도달

3. **NPU의 정적 그래프 제약을 activation/weight 두 축으로 우회한다.**
   - Activation-centric: 시퀀스를 표준 청크(예: 256)로 쪼개 NPU, 잔여만 GPU
     - padding(평균 1.91× 오버헤드) · online graph 생성(seq 1000에서 ~2 s) 모두 회피
   - Weight-centric: weight를 row 차원으로 분할해 NPU에 안 맞는 shape도 둘이 나눠 처리

4. **400 μs `clFinish` 대신 sleep-then-poll 동기화.**
   - 기존: `clFinish`/`QNN_Wait` 등 API 호출 → 커널 실행과 무관한 드라이버 고정 비용 ~400 μs 발생
   - 그러나 LLM 레이어는 동형(isomorphic)이라 커널 대기 시간 예측 가능
   - 예상치만큼 sleep → `usleep` 최소 단위(80~100 μs) 한계 보완 위해 깬 뒤 flag bit 폴링
   - 디코딩에서 동기화 비용 제거 효과: **4.01×** (커널 자체가 수백 μs라 동기화가 압도적으로 비쌌음)

5. **W4A16의 위치 — 정확도 vs 속도의 sweet spot.**
   - Weight: INT4 저장 → 로드 BW 절반 (decode의 병목을 직접 완화)
   - Dequant: compute 직전에 FP16으로 변환 (GPU/NPU 각 device에서 in-register 처리)
   - Compute: FP16 → NPU FLOAT 성능 풀 활용, 정확도 손실 거의 없음
   - 비교: activation도 INT로 양자화하는 llm.npu/Qualcomm-AI는 소형 모델에서 20%+ 정확도 손실

---

### §3 Performance Characterization

| Char | 내용 | 시사점 |
|---|---|---|
| **GPU-1: Linear** | 소형 텐서는 memory-bound, 임계 넘으면 compute-bound로 평탄화 — 단조 증가 | 어떤 shape이든 예측 가능하게 동작 |
| **GPU-2: High-cost sync** | `clFinish` 고정 ~400 μs (드라이버 상태 동기화) | 디코딩 커널 자체보다 동기화가 더 비쌈 → §4.3 fast sync 동기 |
| **NPU-1: Stage** | 32×32 tile에 안 맞는 차원은 padding → tile 경계까지 latency 동일(계단) | 비표준 시퀀스 길이면 NPU가 GPU보다도 느려질 수 있음 |
| **NPU-2: Order-sensitive** | `[M,K]×[K,N]` vs `[N,K]×[K,M]ᵀ`: 연산량 같지만 6× 차이 | invariant `(A@B)ᵀ = Bᵀ@Aᵀ`로 항상 weight를 stationary 쪽에 배치 |
| **NPU-3: Shape-sensitive** | row > column이어야 효율. | FFN-down 같은 차원 축소 matmul → NPU 이점 0.5~1.5×, GPU 분담 필요 |
| **Memory-1: Underutilized BW** | GPU 단독 40~45 GB/s, GPU+NPU 동시 60 GB/s (max 61.9). 75/25 split 최적 | 디코딩에서 "더 느린 쪽도 같이 돌리기"가 정당화됨 |

---

### §4 Design

**원칙**:
- Prefill = throughput 극대화 (compute-bound)
- Decode = memory BW 극대화 (memory-bound)
- CPU는 control plane 전용 (에너지 효율 낮고 일반 작업과 충돌)

#### §4.1 Layer-level GPU-NPU execution

operator affinity 기반 정적 배치:

- **Prefill**
  - Matmul → NPU (FP, 큰 시퀀스, NPU-2 보정 위해 weight·input 순서 swap)
  - RMSNorm/SwiGLU → GPU
- **Decoding**
  - Matmul이 matrix-vector (작은 seq) → NPU-1 stage performance 때문에 GPU가 primary

#### §4.2 Tensor-level GPU-NPU parallelism (핵심 기여)

layer-level이 못 푸는 세 가지 문제를 텐서 단위로 해결:
- NPU shape degradation
- BW 미활용
- 정적 그래프 생성 비용

**세 가지 partition 전략:**

1. **Weight-centric (static shape)**
   - weight를 row 차원으로 `gpu_size : npu_size` 비율로 split
   - 두 sub-tensor가 각자 backend에서 동시 실행 → 결과 merge
   - 시퀀스 길이 고정인 경우 (정적 shape 보장)

2. **Activation-centric (dynamic shape)**
   - 시퀀스를 표준 사이즈 청크(예: 256)들과 잔여 청크로 분할
   - 표준 청크 → NPU 순차 실행, 잔여 → GPU 병렬
   - padding(평균 1.91×)·online graph 생성(seq 1000에서 ~2 s) 모두 회피

3. **Hybrid**
   - activation-centric의 잔여가 너무 작아 GPU·NPU 모두 underutilize될 때
   - 잔여를 padding으로 표준 사이즈로 키운 뒤 weight-centric으로 분할

**Partition ratio 결정 (Solver):**
- 모든 partition 전략을 enumerate
- 각 전략의 비용 = `max(GPU_time, NPU_time) + sync_overhead + copy_overhead`
- 이를 minimize하는 전략·비율 선택
- 비표준 시퀀스 길이는 GPU-1 linearity + NPU-1 계단 함수로 보간

#### §4.3 Fast synchronization

- **UMA memory pool**
  - 입출력 텐서를 host↔device 공유 매핑 슬롯에 고정 할당, 레이어 간 재사용
  - 복사·재매핑 비용 제거
- **Sleep + poll**
  - 레이어 동형 → GPU/NPU 커널 대기 시간 예측 가능 → 그만큼 sleep
  - `usleep` 80~100 μs 단위 한계 보완을 위해 깬 뒤 출력 텐서 말단의 flag bit를 small core가 폴링
  - `clFinish` 400 μs → 수 μs로 단축
- **Phase별 동기화 방향이 반대**
  - Prefill: NPU-dominant → GPU를 NPU 안에 숨김
  - Decode: GPU-dominant → NPU를 GPU 안에 숨김

#### §4.4 Profiler + Solver + Engine

- **Profiler** (offline, \<20 min)
  - LLM weight shape 종류만 고려 (모델 확정 시 유한)
  - NPU-1 stage 특성이 sub-tensor 최소 크기 하한을 강제 → split ratio 후보 제한
  - activation은 표준 시퀀스 길이 집합으로 제한, 나머지는 보간
- **Solver** (offline)
  - operator별 모든 partition 전략 enumerate, 비용 minimize
  - 비표준 시퀀스는 GPU-1 linearity + NPU-1 stage 계단 함수로 보간
- **Engine** (online)
  - 입력 시퀀스 길이 보고 GPU-only / NPU-only / 세 가지 partition 중 선택

---

### §5 Evaluation

**End-to-end (Snapdragon 8 Gen 3)**
- SOTA 대비 **1.34~6.02×**
- 디코딩 위주(BELLE multiturn) Llama-7B에서 PowerInfer-2(sparse model) 대비 1.34× — sparse가 random small access로 BW 깎아먹어서

**Prefill**
- Llama-8B: 247 tok/s
- InternLM-1.8B: 1092 tok/s (llm.npu 564 tok/s의 1.93×)
- Hetero-tensor가 Hetero-layer 대비 평균 **30.2% 추가 향상**

**Dynamic seq length (seq 525 기준)**
- Online-prepare 대비 2.24×
- Padding 대비 2.21×
- NPU-pipe 대비 1.35×
- (Online-prepare는 seq 1000에서 그래프 생성에만 2050 ms)

**Decoding**
- Llama-8B: 14 tok/s / InternLM-1.8B: 51.1 tok/s
- GPU+NPU 동시 → BW **43.3 → 59.5 GB/s (이론 max의 96%)**

**Fast sync 효과**
- Prefill: +15.8~24.3%
- Decode: 최대 **4.01×**

**Ablation (Llama-8B prefill seq 320)**
- naïve NPU: GPU보다 62.5% 느림
- + activation-centric: 2.05×
- + order swap: 1.79×
- + weight-centric: 1.20×
- + fast sync: 1.19×
- → **각 최적화가 독립적으로 곱해짐**

**Game interference (Wild Rift 동시 실행)**
- GPU-only: FPS 0 (OpenCL이 GPU 제출큐 포화)
- Hetero-tensor: prefill -2.2%, decode -17.7%, **FPS 무영향**

**Energy**
- GPU-only 대비 55%, Hetero-layer 대비 12.8% 절감 (빠른 종료 덕)

---

### §6 Discussion

저자가 미래 SoC에 제안:
- GPU·NPU 통합 스케줄러 (현재는 한 쪽 task가 다른 쪽 블로킹 가능)
- Coherent memory + 통합 API (현재 OpenCL·QNN 분리)
- Lightweight HW 동기화 primitive
