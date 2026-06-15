# Concepts

Reusable background notes referenced by paper notes. When the same explanation would help across multiple papers, extract it here and link from each note.

## Index

- [hardware/pim.md](hardware/pim.md) — DRAM-PIM 구조, bank-level MAC, 명령 실행, 데이터 매핑, LLM decode 적용

- [npu.md](npu.md) — Systolic array, weight-stall, 정적 그래프 제약, NPU-1/2/3 특성, 실무 우회 패턴
- [uma.md](uma.md) — Unified Memory Architecture, 캐시 일관성 스펙트럼, 이기종 프로세서 sync, BW 공유 trade-off

- [parallelism/llm-parallelism.md](parallelism/llm-parallelism.md) — Data/Tensor/Pipeline/Sequence/Expert Parallelism, 3D parallelism, ZeRO
- [parallelism/collective-communication.md](parallelism/collective-communication.md) — All-Reduce/Gather/Scatter/All-to-All/P2P, Ring All-Reduce, NVLink vs IB, overlap

## Candidates (논문 두 번째 등장 시 추출)

- W4A16 / weight-only quantization (AWQ, GPTQ 등)
- Prefill vs decoding phases of LLM inference (TTFT, TPOT, KV cache)
- Roofline model / arithmetic intensity
