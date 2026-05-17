---
title: AWQ (Activation-aware Weight Quantization)
description: Weight quantization 시 activation 분포를 고려해 중요한 channel을 보호하는 기법. 4-bit weight quantization에서 GPTQ 대비 정확도 유지.
tags: [quantization, int4, weight-quantization, awq, inference]
type: concept
related: [smoothquant, llm-inference]
---

# AWQ (Activation-aware Weight Quantization)

## TL;DR

- INT4 weight quantization을 균일하게 적용하면 특정 channel에서 정확도 손실이 큼.
- 관찰: activation magnitude가 큰 channel에 대응하는 weight가 "중요" — 전체의 ~1%지만 quantization error의 대부분을 차지.
- AWQ: 중요 channel의 weight에 per-channel scale을 적용해 quantization error 감소. scale은 이전 레이어에 흡수 → 런타임 오버헤드 없음.
- 결과: W4A16, GPTQ 대비 정확도 유리하고 calibration 빠름.

---

## 1. 문제: INT4의 정확도 손실

```
quantization error:
  INT8: step size 작음 → error 작음
  INT4: step size 큼 → error 큼 (특히 outlier weight channel)

단순 per-channel INT4로도 일부 channel의 error가 크고,
그 channel이 다음 activation에 그대로 전파 → 정확도 하락
```

---

## 2. 핵심 관찰: Salient Weight

activation magnitude로 weight channel의 중요도를 판단:

```
calibration data로 X의 channel별 평균 magnitude 측정:
  importance_j = mean(|X[:, j]|)    ← j: input channel

importance가 높은 channel의 weight = salient weight
  → 이 channel의 quantization error가 출력에 크게 영향
  → 전체 channel의 ~1%가 error의 대부분을 차지
```

이 1%를 FP16으로 유지하면 hardware 효율이 깨짐 → 다른 방법 필요.

---

## 3. 해법: Per-channel Scale

```
Y = X @ W = (X / s) @ (W * s)      ← s: input channel별 scale vector

quant 적용:
  W_q = quant(W * s)    ← scale 후 양자화 → salient channel error ↓

추론 시:
  Y ≈ (X / s) @ dequant(W_q)

X / s 부분: 이전 레이어의 출력에 흡수 → 별도 연산 없음
```

scale s 선택:
```
s_j = mean(|X_j|)^α       ← calibration data 기반
                              α ≈ 0.5 (논문 기본값)

importance 높을수록 s 크게 → W * s 값 커짐
→ 같은 step size 내에서 상대적 quantization error 비율 감소
```

---

## 4. GPTQ와 비교

| | AWQ | GPTQ |
|---|---|---|
| 방법 | Activation 기반 per-channel scale | 2차 정보(Hessian)로 보정값 계산 |
| Calibration 비용 | 낮음 (scale만 찾으면 됨) | 높음 (레이어별 역행렬 계산) |
| 정확도 | 소형 모델에서 유리 | 대형 모델에서 경쟁력 |
| 적용 대상 | **W4A16** | W4A16, W3A16 |
| Runtime 오버헤드 | 없음 (이전 레이어 흡수) | 없음 |

공통점: 둘 다 PTQ(Post-Training Quantization), weight-only, 재학습 불필요.

---

## 5. SmoothQuant과 비교

| | SmoothQuant | AWQ |
|---|---|---|
| 목표 | W8A8 (throughput 향상) | W4A16 (메모리 절약) |
| Activation 역할 | outlier를 weight로 이전 | salient channel 식별에만 사용 (A는 FP16 유지) |
| Scale 흡수 위치 | LayerNorm | 이전 레이어 출력 |
