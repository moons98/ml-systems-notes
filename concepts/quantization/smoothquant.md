---
title: SmoothQuant
description: Activation의 outlier를 weight로 이전(migration)해 INT8 quantization을 가능하게 하는 기법. "Smooth" the activation, keep the quantization simple.
tags: [quantization, int8, activation, smoothquant, inference]
type: concept
related: [awq, llm-inference]
---

# SmoothQuant

## TL;DR

- LLM activation은 특정 channel에 outlier가 집중 → per-tensor INT8 quantization 시 작은 값들의 precision이 날아감.
- Weight는 상대적으로 smooth → quantize하기 쉬움.
- SmoothQuant: activation의 quantization 난이도를 weight 쪽으로 **이전(migration)** 해서 둘 다 INT8로 quantize 가능하게 만듦.
- Migration scale s를 LayerNorm에 흡수 → 런타임 오버헤드 없음.
- 결과: W8A8 INT8 quantization, FP16 대비 정확도 손실 최소.

---

## 1. 문제: Activation Outlier

```
Linear layer: Y = X @ W

X (activation): 특정 channel j에 outlier 집중
  channel 0:  [-0.1,  0.2, -0.3, ...]  ← 정상 범위
  channel 7:  [-92.3, 78.1, 85.4, ...]  ← 비정상적으로 큼

per-tensor scale = max(|X|) / 127 ≈ 92.3 / 127 ≈ 0.73
→ channel 0의 값들은 0, 0, 0 으로 반올림 → precision 소실
```

Weight는 distribution이 균일 → per-channel INT8이면 충분히 정확.
문제는 activation.

---

## 2. 핵심 아이디어: 난이도 이전

```
Y = X @ W = (X / s) @ (s * W)
```

channel별 scale vector s로 activation을 나누고 weight에 곱해주면:
- `X / s`: outlier channel이 줄어들어 quantize하기 쉬워짐
- `s * W`: weight가 약간 커지지만 원래 smooth해서 여전히 quantize 가능

α로 이전 비율 조절:
```
s_j = max(|X_j|)^α / max(|W_j|)^(1-α)

α = 0: 이전 없음 (activation 그대로, weight만 양자화)
α = 1: 전부 weight로 이전 (기존 문제 그대로)
α = 0.5: 균형 (논문 기본값)
```

---

## 3. 런타임 오버헤드 없음: LayerNorm 흡수

Linear layer 앞에는 보통 LayerNorm이 있고, LayerNorm은 이미 scale/bias를 가짐:

```
LayerNorm → Linear 구조:
  Y_ln = γ * (X - μ) / σ + β      ← γ, β가 학습된 파라미터

s를 γ에 흡수:
  γ' = γ / s    ← LayerNorm scale 수정
  W' = W * s    ← Weight에 s 곱함

런타임: Y_ln' 가 자동으로 smoothed activation → 별도 나눗셈 연산 없음
```

calibration 시 s 결정 → γ, W에 한 번만 흡수 → 추가 연산 zero.

---

## 4. 비교

| | Per-tensor INT8 | Per-token/Per-channel | SmoothQuant |
|---|---|---|---|
| Activation 처리 | 단순, 정확도 손실 큼 | scale 연산 overhead | migration으로 분포 자체를 개선 |
| Weight 처리 | per-tensor | per-channel | per-channel (s 흡수 후) |
| 런타임 오버헤드 | 없음 | scale 연산 필요 | 없음 (LayerNorm 흡수) |
| 적용 대상 | W8A8 | W8A8 | **W8A8** |
