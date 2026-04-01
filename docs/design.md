# 🧠 Key Design Decisions

## Chunking Strategy: Hierarchical vs Standard + Neighbors

After empirical testing, I found that **standard chunking with ±2 neighbor expansion** outperforms parent-child hierarchical chunking for targeted information retrieval:

- **Parent-child**: Better for complex multi-hop reasoning requiring broad context.
- **Standard + neighbors**: Better for precise fact extraction while maintaining semantic continuity.
- **Result**: System supports both modes; users can choose based on query type.

## LLM-Optional Architecture: Semantic Search is Often Enough

The system supports two operational modes:

1. **Semantic Search Only**: Pure vector similarity retrieval without LLM inference.
2. **Full RAG**: LLM-powered synthesis with optional verification.

- **Empirical finding**: For most research queries, **semantic search alone provides sufficient precision**. Questions like "The lattice parameters of compound XXX" or "The Magnetic ordering of XXXX" are answered directly by retrieving relevant passages — no generation needed.

- **Decision**: The generator is **optional, not required**. Users can:
    - Start with fast semantic search (sub-second, zero LLM cost)
    - Escalate to RAG only when synthesis across multiple sources is needed
    - Inspect retrieved sources directly rather than relying on LLM summarization

This architecture prioritizes **retrieval quality over generation complexity**, recognizing that for information-seeking tasks, finding the right passages matters more than rephrasing them.

---
## Reflection Loop: When Self-Critique Hurts

For queries that do use the generator, the system includes an optional Generator → Critic verification loop. However, **empirical evaluation showed the critic model frequently rejects correct initial responses** for retrieval tasks, adding latency without improving accuracy.

**Root cause**: Critics trained on general reasoning tasks don't align well with retrieval-specific quality metrics. A factually correct answer citing relevant sources can be flagged as "unverified" due to phrasing differences or conservative thresholds.

**Decision**: Reflection loop is configurable but **disabled by default**. For information retrieval, direct source inspection by the user is more reliable than automated verification. The critic remains available for experimentation on synthesis-heavy tasks where self-correction may add value.

## Why Local Inference?

Running LLMs via Ollama (vs API calls) provides:
- **Data privacy**: Research notes never leave your machine
- **Zero marginal cost**: No per-query API fees
- **Offline capability**: Full functionality without internet
- **Model flexibility**: Easy swapping between llama3, gpt-oss:20b, etc.

