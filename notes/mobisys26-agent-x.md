---
title: "Agent-X: Full Pipeline Acceleration of On-device AI Agents"
venue: MobiSys 2026
authors: Jinha Chung, Byeongjun Shin, Jiin Kim, Minsoo Rhu
affiliations: KAIST
tags: [llm-inference, on-device, speculative-decoding, prefix-caching, agentic, prefill, decode, edge]
read_date: 2026-05-26
status: read
url: https://arxiv.org/abs/2605.10380
pdf: ../pdfs/mobisys26-agent-x.pdf
---

## TL;DR

- **문제**: On-device LLM 에이전트는 서버와 달리 prefill(21.7%)과 decode(68.7%) 모두 bottleneck. H200 대비 메모리 대역폭 11%, 컴퓨팅 2%에 불과한 하드웨어 제약이 원인.
- **접근**: **PromptWeaver** — 에이전트 입력 패턴에 맞게 prompt를 재구성해 prefix caching 극대화. **ExSpec** — 출력의 96%가 입력과 겹친다는 관찰에 착안, n-gram LUT 기반 LLM-free speculative decoding으로 multi-token tax 회피.
- **결과**: Apple M4 Pro + TinyAgent-7B 기준 prefill 1.97×, decode 1.73×, end-to-end 1.61× speedup. 정확도 손실 없음.

---

## Background

기존 두 가지 가속 기법이 on-device 에이전트에 그대로 적용되지 않는 이유:

**Prefix caching** — 동일한 prefix가 반복될 때 KV cache를 재사용하는 기법. 에이전트 입력은 system prompt 뒤에 쿼리에 따라 선택된 tool descriptions가 일찍 삽입되어, static prefix가 전체의 1.6%에 불과 → cache hit 거의 없음.

**Speculative decoding** — 작은 draft model이 N개 토큰을 미리 생성하면 큰 target model이 한 번에 검증해 throughput을 높이는 기법. On-device에서는 두 가지 문제:
- **Multi-token tax**: 서버와 달리 on-device 프레임워크는 single-batch에 최적화 → verification 시 토큰 수에 비례해 latency 증가 (2토큰에 1.86×)
- **Draft model 트레이드오프**: draft가 작으면 acceptance rate 낮아 speedup 없고, 크면 draft 생성 비용이 bottleneck → SOTA draft model로도 최대 1.20×

Agent-X는 두 기법을 에이전트 워크플로의 고유한 특성(입력의 구조적 반복성, 출력의 예측 가능성)을 이용해 on-device에 맞게 재설계한다.

---

## Characterization (Section 3)

### LLMCompiler 워크플로

TinyAgent가 사용하는 LLMCompiler는 세 단계로 구성:

```
Planner  →  Execution Unit  →  Arbiter
  ↑                               |
  └─────────── replan ────────────┘ (필요 시)
```

- **Planner**: system prompt + tool descriptions/guidelines + few-shot examples + user query → 실행 계획(tool call 목록) 생성
- **Execution Unit**: tool들을 병렬 실행, call-observation 쌍 기록
- **Arbiter**: 실행 결과를 보고 태스크 완료 여부 판단. 미완료 시 Planner에 새 instruction 전달

### Latency 분포

- 태스크당 평균 end-to-end latency: **35.4초**
- Planner: **43.5%**, Arbiter: **46.9%**
- 각 컴포넌트 내 decode 비중: **68.7%**, prefill: **21.7%**

서버 LLM은 decode가 95% 이상인 것과 대조적으로 prefill도 무시 못 할 비중. 이유:
1. 에이전트 입력이 tool descriptions, guidelines, examples를 포함해 긴 구조
2. On-device 하드웨어의 compute 부족으로 prefill 비용이 상대적으로 큼

### Planner 입력 특성 (평균 1,739 토큰)

**① Early dynamicity despite large static portion**

```
[System prompt 32.7%] [Tool desc+guidelines (동적)] [Examples (동적)] [Query]
                       ↑ prompt의 1.6% 위치에서 dynamic 시작
```

전체의 32.7%가 static한 system prompt임에도, tool descriptions가 일찍 삽입되어 prefix caching이 거의 불가. 그런데 각 tool description은 고정 텍스트 — "동적"인 것은 어떤 툴이 선택되느냐뿐.

**② Tool co-activation locality**

특정 툴들이 함께 호출되는 경향이 있다:
- `get_zoom_meeting_link` ↔ `get_email_address`: 91%
- `get_zoom_meeting_link` ↔ `get_phone_number`: 6%

같은 테마(email, contacts, maps 등) 내 툴끼리 co-activation이 높고, 테마 간에도 거리 차이가 있음. 이 패턴은 소수의 클러스터로 수렴함을 시사.

**③ Single-tool example의 중요성**

실제 쿼리의 82%는 복합 tool task이지만, retrieved examples의 57%는 단일 툴 예시. 복합 task도 단일 툴 예시들의 조합으로 충분히 커버됨 → example DB를 단순하게 유지할 수 있음.

### Arbiter 입력 특성

LLMCompiler 내부 상태에 따라 **딱 2가지 static prefix variant**만 존재. 각각 전체 입력의 **88%, 90%**가 static prefix. Tool 실행 결과(call-observation 쌍)는 뒤에 누적되므로 prefix position은 항상 고정 → prefix caching 효율 높음.

### Decode 출력 특성

- **Planner 출력 토큰의 96%**가 입력 prompt 내 토큰과 겹침
- **Arbiter 출력 토큰의 87%**가 입력 내 토큰과 겹침

출력이 few-shot example의 구조를 따르며 argument만 바뀌는 템플릿 형태. 이 예측 가능성이 ExSpec의 핵심 근거.

### Multi-token tax (on-device 특유 문제)

On-device 프레임워크(MLX-LM)는 single-batch 추론에 최적화:

```
1토큰 decode:       131ms
2토큰 verification: 244ms  (×1.86)
```

이상적으로는 2토큰이 1토큰 대비 거의 동일해야 speculative decoding이 이득인데, on-device에서는 토큰 수에 비례해 latency 증가. 기존 speculative decoding 방식으로는 최대 **1.20×** speedup에 그침 (draft model 크기 트레이드오프 + tax 복합).

---

## PromptWeaver (Prefill 가속)

**목표**: prompt를 재구성해 최대한 많은 prefix를 static하게 만들고, 해당 KV cache를 오프라인 precompute해 SSD에 저장.

```
재구성된 Planner 입력:
  [① 전체 tool desc — 완전 static]
  [② 클러스터 example 세트 — semi-cacheable, SSD에서 로드]
  [③ K=1 dynamic example — 런타임 계산]
  [Query]
```

### Step 1. Early dynamic token 재구성

선택된 툴만 넣는 대신 **모든 툴의 description/guidelines를 고정 순서로 포함**. 어떤 쿼리가 와도 이 구간이 동일 → 완전한 static prefix.

- Uncacheable 토큰: 1,711 → 519 (**70% 감소**)
- SSD 저장 비용: 100개 툴 기준 1.4 GB

### Step 2. Co-activation locality-based clustering

Training data의 ground truth를 기반으로 co-activation matrix(adjacency matrix) 생성 후 **NMF(Non-negative Matrix Factorization)** 적용:

- 결과: **8개 클러스터**, 각 2~6개 툴
- 같은 테마 내 툴끼리 자연스럽게 묶임 (email, contacts, maps 등)

### Step 3. Theme-based cluster ordering

클러스터에 테마를 부여하고, **같은 테마의 클러스터끼리 인접 배치**. 이유: prefix matching 시 앞부분 클러스터가 일치할수록 KV cache reuse가 많아지기 때문. 예시 쿼리가 클러스터 1-2를 활성화하면 "1" prefix가 캐시되어 있을 때 부분 hit 가능.

### Step 4. Cluster combination selection (greedy)

가능한 조합 수는 2^C − 1개 (C = 클러스터 수). 전부 저장 불가능하므로 budget N 안에서 coverage를 최대화하는 조합을 greedy 선택:

1. Training data의 모든 샘플에서 activated cluster sequence의 prefix들을 수집 ("A-B-C" → "A", "A-B", "A-B-C")
2. 각 후보 prefix를 추가했을 때 coverage 증가량 계산
3. 증가량이 가장 큰 prefix를 greedy하게 선택, budget 소진까지 반복

결과: **15개 조합, 6.26 GB SSD**, training data의 **74.4% 쿼리 커버**.

### 정확도 유지: K=1 dynamic example

All-inclusive tool descriptions로 인해 관련 없는 툴이 포함되므로, 정확도 보완을 위해 ToolRAG로 retrieve한 **K=1개의 dynamic example을 prompt 끝에 추가**. K=1이 정확도와 caching 효율의 최적 균형점.

### Arbiter

Static prefix 비율이 이미 88~90%라 별도 재구성 없이도 prefix caching 효과적. PromptWeaver 적용 시 **4.35×** prefill speedup (Planner의 1.57× 대비 높음).

---

## ExSpec (Decode 가속)

**근거**: Planner/Arbiter 출력의 96%/87%가 입력 prompt 내 토큰과 겹침 → 입력의 few-shot example이 출력 패턴을 거의 결정함.

### n-gram LUT 구성 (on-the-fly)

쿼리가 들어올 때마다 **현재 쿼리의 few-shot examples + user query**로 trigram(n=3) LUT를 빌드:

```
토큰 스트림에 sliding window 적용:
  [t1, t2] → t3  (가장 빈번한 t3 기록)
  [t2, t3] → t4
  ...
```

메모리: **수 KB** (LLM draft 모델의 수백 MB~GB 대비 무시 가능).

### Draft 생성 및 Selective Fallback

```
decode 중 매 스텝:
  현재 앞 2토큰으로 LUT 조회
    ├─ 첫 토큰 hit  → draft 생성 계속 (이후 miss 나도 진행) → verification
    └─ 첫 토큰 miss → 즉시 autoregressive (verification 없음 → tax 없음)
```

**Selective**의 핵심: draft가 맞을 가능성이 없으면 verification 비용 자체를 안 씀. LUT는 hit 여부를 O(1)로 판단 가능하므로 이 결정에 overhead 없음.

- Planner draft token accuracy: **0.25**, Arbiter: **0.26**
- Fallback 빈도: Planner **17회/쿼리**, Arbiter **37회/쿼리**

---

## Evaluation

**환경**: Apple Mac mini M4 Pro (16 GPU cores, 546 GB/s, 64 GB RAM, 512 GB SSD), TinyAgent-7B, 1,022 test examples

| 컴포넌트 | Speedup |
|---|---|
| PromptWeaver — Planner prefill | 1.57× |
| PromptWeaver — Arbiter prefill | 4.35× |
| PromptWeaver — 전체 prefill | 1.97× |
| ExSpec — decode | 1.73× |
| Agent-X end-to-end | 1.61× |

- PromptWeaver 단독: 1.16×, ExSpec 단독: 1.43× → 결합 시 1.61×
- 정확도 손실 없음 (K=1 설정)
- Uncacheable 토큰 70% 감소 (1,711 → 519)
