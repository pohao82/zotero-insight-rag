# Evaluation: Parsing Quality and Retrieval Impact

## Motivation

In scientific RAG systems, PDF parsing quality directly affects downstream components:

PDF → text → chunking → embeddings → retrieval

Errors introduced early propagate through the entire pipeline.

---

## Lightweight Evaluation Approach

Instead of relying on synthetic benchmarks, this system uses direct inspection of parsed outputs and their impact on retrieval behavior.

### Key idea

Focus on **structural fidelity** rather than abstract metrics.

---

## Cached Markdown for Fast Iteration

Parsed documents are cached as structured Markdown files, enabling:

- Rapid comparison between parsing strategies (`marker-pdf` vs `pypdf`)
- Reuse of parsed content without re-running expensive PDF extraction
- Controlled experiments isolating parsing effects from downstream components

This significantly reduces iteration time during development.

---

## Observations

### 1. Standard Parsing Issues

- Multi-column text becomes interleaved  
- Tables are flattened and lose structure  
- Equations become fragmented or unreadable  

→ Leads to noisy embeddings and poor retrieval alignment  

---

### 2. Layout-Aware Parsing Improvements

- Preserves column structure  
- Maintains table integrity  
- Retains LaTeX equations  

→ Produces cleaner semantic units for embedding  

---

## Impact on Retrieval

- Standard parsing → fragmented chunks → irrelevant retrieval  
- Layout-aware parsing → coherent chunks → correct section retrieval  

Particularly important for:
- tabular data  
- equation-heavy content  
- structured scientific arguments  

---

## Takeaway

Improving parsing quality yields larger gains than increasing model complexity.

This reinforces a key principle:

> Retrieval quality is often the primary bottleneck in RAG systems.
