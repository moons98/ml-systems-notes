---
title: "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference (HeteroInfer)"
venue: SOSP 2025
authors: Le Chen, Dahu Feng, Erhu Feng, Yingrui Wang, Rong Zhao, Yubin Xia, Pinjie Xu, Haibo Chen
affiliations: SJTU IPADS, Tsinghua, SenseTime Research
tags: [llm-inference, mobile, npu, gpu, heterogeneous-compute, edge-ai, quantization]
read_date: 2026-05-11
status: reading
url: https://doi.org/10.1145/3731569.3764808
pdf: ../pdfs/sosp25-heteroinfer.pdf
---

## TL;DR

> _(한 문단으로: 어떤 문제, 핵심 통찰, 결과)_

## Key Insights

> _(3–5개. 다른 논문/시스템에 transferable한 것 위주)_

1.
2.
3.

## Background

논문 이해에 필요했던 일반 개념들. 재사용 가능한 건 `concepts/`로 빼고 link.

- Systolic arrays & weight-stall computing → _(TODO: extract to concepts/)_
- Unified Memory Architecture (UMA) on mobile SoCs → _(TODO)_
- W4A16 quantization → _(TODO)_
- Prefill vs. decoding (TTFT, TPOT) → _(TODO)_

## Walkthrough

### §2 Background & Related Work

**§2.3 Limitations of existing mobile inference engines** — 논문이 풀려는 gap

- 기성 추론 엔진들은 이기종 가속기를 동시에 안 씀 (대부분 GPU-only 또는 NPU-only)
  - 이유: GPU-NPU 간 sync 맞추기 어려움, workload partition도 까다로움
- NPU에서 INT 기반 추론 시도들의 한계
  - Qualcomm AI: 작은 모델에서 최대 20% 정확도 저하
  - mixed precision / outlier 기반 quantization: 입력 데이터셋에 dependent → 사용자 입력에 따라 정확도가 흔들림

### §3 Performance Characterization

> _(여기가 진짜 알짜. NPU의 3가지 특성 + memory bandwidth 관찰)_

- **GPU-1**: Linear performance
- **GPU-2**: High-cost synchronization (~400μs from clFinish)
- **NPU-1**: Stage performance —
- **NPU-2**: Order-sensitive performance —
- **NPU-3**: Shape-sensitive performance —
- **Memory-1**: Underutilized bandwidth with single processor —

### §4 Design

#### §4.1 Layer-level GPU-NPU execution

#### §4.2 Tensor-level GPU-NPU parallelism

- Weight-centric partition (static shape)
- Activation-centric partition (dynamic shape)
- Hybrid partition

#### §4.3 Fast synchronization

#### §4.4 Profiler + Solver + Inference Engine

### §5 Evaluation

> _(놀라웠던 숫자 중심으로)_

### §6 Discussion

## Open Questions / Critique

-
