---
title: KV Cache
description: Decode 단계에서 과거 토큰의 K, V를 재사용하기 위한 메모리 구조. 물리 저장 방식, 단편화 문제, 최적화 기법(PagedAttention, prefix caching, GQA, quantization, offloading)을 다룸.
tags: [inference, kv-cache, memory, paged-attention, prefix-caching, gqa, quantization]
type: concept
related: [llm-inference, paged-attention, flash-attention]
---

# KV Cache

## TL;DR

- KV cache는 decode 단계에서 과거 토큰의 K, V를 재계산하지 않기 위해 HBM에 저장해두는 구조. (개념·메모리 수식은 → [llm-inference](./llm-inference.md))
- 실제 서빙에서의 문제는 **단편화(fragmentation)** 
    - 요청마다 max_len을 미리 잡으면 짧게 끝나는 요청이 메모리를 낭비하고, batch를 키우지 못해 GPU 활용률이 떨어짐. (batch=1이면 낭비는 있어도 무해)

- **PagedAttention**: KV cache를 page 단위로 쪼개 필요할 때만 할당 → 단편화 제거.
- **Prefix caching**: 동일한 prefix의 KV page를 재사용 → 시스템 프롬프트 등 반복 prefill 제거.
- **GQA / MQA**: K, V head 수를 줄여 KV cache 크기 자체를 줄임 → 아키텍처 레벨.
- **KV quantization**: FP16 → FP8/INT8로 압축 → 2× 메모리 절약.
- **CPU offloading**: VRAM 부족 시 KV를 CPU DRAM에 두고 필요 시 prefetch.

---

## 1. 물리 구조

### 저장 위치와 shape

GPU HBM에 레이어별·헤드별로 K, V 텐서를 유지:

```
KV cache tensor shape (레이어 하나):

  K: [batch_size, num_heads, seq_len, head_dim]
  V: [batch_size, num_heads, seq_len, head_dim]

  seq_len은 decode가 진행될수록 1씩 증가
```

전체 요청 하나의 KV cache 크기:
```
2 × L × H × d_head × S × bytes_per_element

  L: 레이어 수
  H: KV head 수 (GQA면 Q head 수보다 작음)
  S: 현재 시퀀스 길이 (prefill 길이 + 생성된 토큰 수)
```

예: Llama-3-8B (L=32, H=8 GQA, d_head=128, FP16)
```
토큰당: 2 × 32 × 8 × 128 × 2 byte = 131,072 byte ≈ 128 KB/token
S=4096: ≈ 512 MB per request
```

요청 수가 많아질수록 KV cache가 VRAM을 지배 → batch size의 hard ceiling.

### 접근 패턴

각 decode step에서:
1. 새 토큰의 K, V를 계산해 cache 끝에 **append**
2. cache 전체를 **load**해서 attention 계산 (FlashAttention이 이 부분을 IO-efficient하게 처리)

append는 cheap, load는 S가 커질수록 expensive.

---

## 2. 단편화 문제와 PagedAttention

### 문제: 정적 할당의 비효율

전통적 방식: 요청이 들어오면 `max_seq_len`만큼 KV cache를 미리 예약:

```
Request A  [████████████░░░░░░░░]  max=2048, 실제=800  → 60% 낭비
Request B  [████░░░░░░░░░░░░░░░░]  max=2048, 실제=300  → 85% 낭비
빈 공간이 있어도 연속하지 않으면 새 요청에 못 씀
```

- 생성 완료 전까지 실제 길이를 모름 → 항상 max 예약
- Internal fragmentation + 요청 간 공간이 연속되지 않아 external fragmentation도 발생

### PagedAttention

OS의 virtual memory에서 착안 (vLLM에서 제안, 2023):

```
물리 page pool (예: page 크기 = 16 tokens)

Page 0  [t1~t16]  ← Request A의 첫 block
Page 1  [t1~t16]  ← Request B의 첫 block
Page 2  [t17~t32] ← Request A의 두 번째 block (물리적으로 비연속 가능)
Page 3  [t17~t32] ← Request C의 두 번째 block
...
```

- 각 요청은 **block table** (논리 block → 물리 page 매핑)을 가짐
- 토큰이 생성될 때만 새 page를 할당 → 미사용 공간 없음
- 물리적으로 연속하지 않아도 됨 → external fragmentation 없음
- 동일한 prefix를 가진 요청 간 page **공유** 가능 (prefix caching의 기반)

낭비는 마지막 page의 미사용 슬롯뿐 (평균 page_size/2 ≈ 7.5 tokens).

> llama.cpp 등 로컬 추론 환경은 대부분 batch=1 → 단편화 자체가 문제가 되지 않음 → PagedAttention을 굳이 구현하지 않는 이유. PagedAttention은 다중 요청을 동시에 처리하는 serving 환경에서 의미 있음.

---

## 3. Prefix Caching

시스템 프롬프트, few-shot 예시처럼 **반복되는 prefix**의 KV를 저장해두고 재사용:

```
요청 1:  [System Prompt 1000 tokens] + [User Query A]
          → prefill 전체 실행, KV cache 저장

요청 2:  [System Prompt 1000 tokens] + [User Query B]
          → prefix KV cache hit → User Query B 부분만 prefill
          → prefill 비용 ~1000/1100 절감
```

동작 방식:
- Prefix 토큰들의 **hash**를 key로 KV page를 캐싱
- 새 요청의 prefix hash가 일치하면 해당 page를 그대로 사용 (copy-on-write)
- 정확히 일치해야 함 — 토큰 1개라도 다르면 miss

효과가 큰 케이스:
- 긴 시스템 프롬프트 (RAG context, 코드베이스 삽입 등)
- 같은 few-shot 예시를 반복하는 배치 요청

---

## 4. GQA / MQA — 아키텍처 레벨 KV 절감

KV head 수를 Q head 수보다 줄여서 KV cache 크기 자체를 줄이는 방식.

```
MHA (Multi-Head Attention):
  Q: H heads   K: H heads   V: H heads
  KV cache = 2 × H heads

MQA (Multi-Query Attention):
  Q: H heads   K: 1 head    V: 1 head
  KV cache = 2 × 1 head  → 1/H 절약
  (품질 손실이 클 수 있음)

GQA (Grouped Query Attention):
  Q: H heads   K: G groups  V: G groups  (G ≪ H)
  KV cache = 2 × G heads  → G/H 절약
  (MHA와 MQA의 중간, 품질 유지하면서 절약)
```

예: Llama-3-8B (H=32 Q heads, G=8 KV heads)
- MHA 대비 KV cache 4× 절약
- 실제 H=32, G=8이면 4개의 Q head가 1개의 K, V head를 공유

채택 현황: Llama-3, Mistral, Qwen 등 현대 모델 대부분이 GQA 사용.

| | MHA | GQA | MQA |
|---|---|---|---|
| KV heads | H | G (H/4 ~ H/8 수준) | 1 |
| KV cache 크기 | 기준 | ~1/4~1/8 | ~1/H |
| 품질 | 기준 | MHA에 근접 | 손실 있음 |

---

## 5. KV Quantization

KV를 낮은 정밀도로 저장해 메모리를 압축:

| 포맷 | 압축률 | 비고 |
|---|---|---|
| FP16 → FP8 | 2× | H100 HW native, 품질 손실 적음 |
| FP16 → INT8 | 2× | per-token 또는 per-channel scale 필요 |
| FP16 → INT4 | 4× | 품질 손실 큼, 실험적 |

### Weight quantization과의 차이

Weight는 정적 → 오프라인 calibration 가능.  
KV는 동적으로 생성되는 activation → outlier가 강하고 분포가 요청마다 다름.

- **Key**가 **Value**보다 quantization에 더 민감 (key의 outlier가 attention score에 직접 영향)
- Per-tensor scale로는 정확도 손실이 크고, per-token 또는 per-channel scale이 필요

---

## 6. CPU Offloading

VRAM이 부족할 때 KV cache를 CPU DRAM으로 내려두는 전략:

```
Layer N 연산 중:
  GPU HBM  [Layer N KV로 attention 연산]
              ↑ 동시에 prefetch (PCIe)
  CPU DRAM [Layer N+1 KV] → GPU로 올라오는 중

Layer N 완료 후:
  GPU HBM  [Layer N KV] → CPU로 offload (다 썼으니 반납)
  GPU HBM  [Layer N+1 KV] → 이미 올라와 있음 → 바로 연산
```

- Layer N attention이 끝나면 Layer N KV는 더 이상 불필요 → CPU로 **offload**
- 그 동안 Layer N+1 KV를 미리 CPU → GPU로 **prefetch**
- 연산과 전송을 **overlap** → PCIe 전송이 critical path에서 숨겨짐

병목: PCIe BW (~32 GB/s, Gen 4 x16) vs HBM (~2 TB/s)
- 레이어당 KV가 크면 prefetch 시간이 연산 시간보다 길어져 stall 발생
- Long context (수만 토큰)에서 불가피하게 쓰이는 선택지

---

## 7. Max Seq Len 초과 시 처리

KV cache가 가득 차면 새 토큰을 위한 공간이 없어짐. 전략은 크게 세 가지:

### 단순 truncation (sliding window)

오래된 토큰의 KV부터 버림. 가장 단순한 방법:

```
max_seq_len = 4096, 현재 4097 토큰 생성 시:
  → 가장 오래된 토큰(t1)의 KV 제거
  → t2~t4097만 attend 가능
```

Mistral의 **Sliding Window Attention**이 이 방식을 아키텍처에 내장. 항상 최근 K 토큰만 attend. 오래된 context는 완전히 소실.

### StreamingLLM

단순 sliding window의 문제: 첫 몇 개 토큰("attention sink")을 버리면 품질이 급격히 떨어짐.

관찰: 초기 토큰들은 내용과 무관하게 attention score가 비정상적으로 높게 몰림 → 이걸 버리면 attention 분포가 무너짐.

```
보존: [t1, t2, t3, t4] (sink tokens) + [최근 W 토큰]
버림: 중간 오래된 토큰들
```

→ sink tokens + sliding window 조합으로 무한 길이 생성 가능, KV cache는 고정 크기 유지.

### KV Eviction (H2O 등)

단순히 오래된 것을 버리는 게 아니라 **중요도 기반으로 evict**:

- 누적 attention score가 낮은 토큰의 KV를 우선 제거
- 최근 토큰 + attention을 많이 받은 토큰("heavy hitter") 보존
- 단순 truncation보다 품질 유지

**"누적"의 의미**: 여러 decode step(토큰 하나씩 생성하는 각 iteration)에 걸쳐 score를 합산:
```
step 1: q_(N+1) @ K_cache^T → 각 과거 토큰에 score
step 2: q_(N+2) @ K_cache^T → 각 과거 토큰에 score
...
누적합이 낮은 토큰 = 생성 과정 내내 거의 참조되지 않은 토큰 → evict 후보
```

**Eviction 단위**: 토큰 하나를 지우면 해당 레이어의 **모든 head에서 동시에** 제거됨:
```
K: [num_heads, seq_len, head_dim]
                ↑ seq_len 차원이 모든 head에서 공유
→ position i 제거 = 모든 head의 position i가 한 번에 제거
```
head별로 독립적으로 다른 토큰을 지우려면 ragged tensor가 필요해 GPU 구현이 비현실적. 단, **레이어 간에는 독립적**으로 eviction 결정 가능 — layer 3은 token 5를 유지하면서 layer 7은 버릴 수 있음.

---

| 전략 | 버리는 기준 | 품질 | 복잡도 |
|---|---|---|---|
| Truncation | 가장 오래된 것 | 낮음 | 단순 |
| StreamingLLM | 중간 토큰 (sink 보존) | 중간 | 낮음 |
| H2O eviction | attention score 낮은 것 | 높음 | 높음 |

공통점: 모두 **오래된/덜 중요한 context를 잃는 trade-off**. 무손실로 무한히 늘리는 방법은 없음.

---

## 8. 최적화 기법 요약

| 기법 | 무엇을 줄이나 | 적용 레이어 | 비용 |
|---|---|---|---|
| **PagedAttention** | 메모리 단편화 | 서빙 시스템 | 구현 복잡도 |
| **Prefix caching** | 반복 prefill 비용 | 서빙 시스템 | 캐시 관리 overhead |
| **GQA** | KV cache 크기 자체 | 모델 아키텍처 | 재학습 필요 |
| **KV quantization** | KV 메모리 사용량 | 커널 | 정확도 일부 손실 |
| **CPU offloading** | VRAM 사용량 | 서빙 시스템 | 레이턴시 증가 |
