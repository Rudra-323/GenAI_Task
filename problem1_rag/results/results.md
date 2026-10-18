# RAG evaluation results (k = 5, 22 answerable questions)

## Retrieval

| Metric | No filter | With file_type filter |
|---|---|---|
| Hit rate | 0.9545 | 0.9545 |
| Recall@5 | 0.9545 | 0.9545 |
| MRR | 0.8485 | 0.8485 |
| nDCG@5 | 0.8755 | 0.8755 |
| Precision@5 | 0.2 | 0.2 |
| Context precision | 0.8485 | 0.8485 |

Retrieval latency (embedding + search, local): p50 = 1047.48 ms, p95 = 1231.88 ms

## Chunk size ablation (retrieval only)

| Size | Overlap | Chunks | Hit | MRR | nDCG | Ctx precision |
|---|---|---|---|---|---|---|
| 250 | 25 | 90 | 1.0 | 0.947 | 0.9605 | 0.947 |
| 500 | 50 | 44 | 0.9545 | 0.8485 | 0.8755 | 0.8485 |
| 1000 | 100 | 20 | 0.9545 | 0.7212 | 0.7817 | 0.7227 |

## Answer quality

| Metric | Score |
|---|---|
| Faithfulness (supported claims / all claims) | 1.0 |
| Answer relevance (1-5) | 5.0 |
| Exact match | 0.0 |
| F1 | 0.4107 |
| Contains gold answer | 0.6818 |
| Correct refusals on unanswerable questions | 4 / 4 |
| Wrong refusals on answerable questions | 1 / 22 |

Retrieval vs generation diagnosis: {"retrieved_and_correct": 17, "retrieved_but_wrong (generation problem)": 4, "not_retrieved_and_wrong (retrieval problem)": 1, "not_retrieved_but_correct": 0}

## End-to-end latency and tokens

p50 = 1620.5 ms, p95 = 6178.2 ms, avg tokens/query = 663.4 in + 132.9 out, generation cost = $0.0 per 1000 queries

## Cost (see results/cost_table.md)

```
[
 {
  "vectors": 10000,
  "build_seconds": 2.5,
  "k": 5,
  "p50_ms": 1.54,
  "p95_ms": 2.29,
  "filtered_p50_ms": 10.53,
  "filtered_p95_ms": 15.85
 },
 {
  "vectors": 100000,
  "build_seconds": 44.2,
  "k": 5,
  "p50_ms": 2.2,
  "p95_ms": 3.46,
  "filtered_p50_ms": 122.72,
  "filtered_p95_ms": 168.25
 }
]
```