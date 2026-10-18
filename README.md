# Applied AI / ML Engineering take-home

Two independent projects in one repo:

| Folder | What it is |
|---|---|
| `problem1_rag/` | Cost-efficient RAG service on **ChromaDB** + evaluation harness + cost model |
| `problem2_judge/` | **LLM-as-Judge** pipeline with bias mitigation, bias measurements and judge validation |
| `llm.py` | Tiny shared helper: one OpenAI-compatible client for OpenAI / Gemini / Groq / OpenRouter, robust JSON parsing |

## Setup (5 minutes)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then paste your API keys into .env
```

Models are chosen in `.env` only (no code change): **generator** = Groq `llama-3.3-70b-versatile`,
**judge** = Gemini `gemini-2.5-flash`. They are different model families on purpose (avoids self-enhancement bias).
Secrets are read from the environment; `.env` is git-ignored.

---

# Problem 1 - Cost-efficient RAG

## Store choice: ChromaDB (embedded, persistent)

| Why Chroma | Trade-off accepted |
|---|---|
| Runs in-process, no server, no per-vector bill | Single node, no built-in replication / HA |
| Persistent on disk, HNSW index, cosine distance, metadata filters | HNSW lives in RAM, so RAM (not disk) sets the VM size |
| Lowest code complexity for the same quality of results | Not meant for 10M+ vectors on one box (see Discussion) |

## Pipeline

* **Ingestion** (`ingest.py`): PDF (pypdf), HTML (BeautifulSoup), Markdown/TXT. **Chunk size 500 chars, overlap 50** (configurable via `--chunk-size/--overlap` or env). Paragraph-aware chunker.
* **Idempotent**: chunk id = `sha1(file name | chunk number | text)`; a file's old chunks are deleted before it is upserted. Re-running gives the same count.
* **Embeddings**: `all-MiniLM-L6-v2`, **384 dimensions**, local ONNX (no API cost).
* **Metadata**: `source`, `file_type`, `chunk_index`, `page` (PDF). Filter by `file_type` and/or `source`.
* **Answering** (`rag.py`): top-k (parameter) -> numbered context -> LLM must cite `[n]`. If the best similarity is below `MIN_SIMILARITY` the LLM is **not called** and the system says it does not know; the prompt also tells the LLM to refuse when the context lacks the answer.
* **API** (`app.py`, FastAPI) + per-query JSON log (`logs/queries.jsonl`) with latency, chunk count, token usage.

## Run

```bash
python -m problem1_rag.ingest                          # build the index (run twice: count stays the same)
uvicorn problem1_rag.app:app --port 8000               # start the API  (docs at http://localhost:8000/docs)
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question":"What is the rate limit for the Pro plan?","k":5}'
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question":"How many days of annual leave?","k":3,"file_type":"pdf"}'      # metadata filter

python -m problem1_rag.cost_analysis --bench           # cost table + latency at 10K / 100K vectors
python -m problem1_rag.evaluate --ablation             # all three evaluation layers + chunk-size ablation
```

## Evaluation design

* **Corpus**: 6 documents (MD, HTML, PDF) about a *fictional* company, so the LLM cannot answer from memory and retrieval truly matters. It deliberately has near-duplicate facts (e.g. rate limits in two places).
* **Questions**: `data/questions.json` - 22 answerable + 4 unanswerable.
* **Relevance label**: each answerable question has an `evidence` phrase from the source. A chunk is relevant if it contains it (works for any chunk size).
* **Layer 1 Retrieval**: Hit rate, Recall@k, MRR, nDCG@k, Precision@k, context precision (RAGAS-style average precision). Also with/without the metadata filter.
* **Layer 2 Answer**: faithfulness (supported claims / all claims, LLM judge), answer relevance (1-5, LLM judge), EM / F1 / contains-gold, refusal accuracy on unanswerable questions, and a retrieval-vs-generation diagnosis table.
* **Layer 3 Cost/latency**: p50/p95 retrieval latency, end-to-end latency, tokens, cost model.

## Results

Generated files: `problem1_rag/results/results.json`, `results.md`, `cost_table.md`, `cost.json`.

> Fill in after running (copy from `results.md`):
> Retrieval (k=5): Hit __ | Recall@k __ | MRR __ | nDCG@k __ | Context precision __
> Answer: Faithfulness __ | Relevance __/5 | F1 __ | Correct refusals __/4
> Latency: retrieval p50 __ ms / p95 __ ms | end-to-end p50 __ / p95 __

### Cost comparison (monthly, USD)

Embedding dimension 384, float32. Assumptions are in `cost_analysis.py` (prices checked Oct 2026: Pinecone pricing + docs, AWS us-east-1 on-demand). Self-hosted = one VM sized so the HNSW index fits in RAM, plus disk and snapshots, **no HA replica**. Managed pods = always-on, billed per hour. Serverless = $0.33/GB storage + $16 per million read units (1 RU per GB of index per query) + $4 per million write units, $50 minimum.

| Vectors | Self-hosted Chroma | Managed pods p1 | Managed pods s1 | Serverless, 100K queries/mo | Serverless, 5M queries/mo |
|---|---|---|---|---|---|
| 100K | $32 (t3.medium) | $70 | $70 | $50 | $50 |
| 1M | $63 (t3.large) | $70 | $70 | $50 | $164 |
| 10M | $191 (r6i.xlarge) | $350 | $70 | $50 | $1,644 |

Re-run `python -m problem1_rag.cost_analysis` after changing any assumption.

## Discussion (Problem 1)

> Write this yourself from YOUR results, 6-8 sentences. Questions to answer:
> 1. When would you switch back to managed? (Think about the table: low query volume, ops time, HA, 10M+ vectors, team size.)
> 2. Was retrieval or generation the weak link? (Use the diagnosis counts in `results.json` and the failed question ids.)
> 3. What trade-offs did you accept? (single node, no HA, RAM-bound HNSW, manual backups.)

---

# Problem 2 - LLM-as-Judge

## Judging modes

| Mode | Used for | When it fits |
|---|---|---|
| Pointwise, reference-based | `run` when a case has `expected_output` | You have gold answers; best for correctness |
| Pointwise, reference-free | `run` when there is no reference | Open-ended writing, tone, instruction following |
| Pairwise A-vs-B (both orders) | `compare` | Comparing two prompts/models; humans find relative judgements easier than absolute ones |

## Rubric (`judge.py`)

Five criteria scored 1-5 with written anchors and a rationale each: **correctness (0.35), faithfulness (0.20), completeness (0.20), instruction_following (0.15), tone_safety (0.10)**. The overall score is the weighted mean computed in code (not by the LLM); **PASS if overall >= 4.0**.

## Bias handling

| Bias | Mitigation in code | How it is measured |
|---|---|---|
| Position (A/B order) | every pair judged in both orders; a win needs both to agree, else tie | `validate.py position`: flip rate, error before/after, first-position share |
| Verbosity / length | rubric says padding never raises a score; pairwise prompt says length is not quality | `validate.py bias`: padded-vs-original score gain and pairwise padded-win rate |
| Self-enhancement | judge and generator are configured independently and use different families | `validate.py self`: same-family vs different-family gap, corrected with neutral answers |
| Sycophancy / style | per-criterion rationale must cite evidence; confident-wrong probe set | `validate.py bias`: fooled rate (judge said PASS on a wrong answer) |
| Score clustering | few-shot calibration anchors and "use the full range" rule | `validate.py agreement`: score histogram and spread, naive vs hardened |

The "naive" judge (no anchors, no rules) exists only as the **before** measurement; the hardened judge is the **after**.

## Run

```bash
python -m problem2_judge.judge run      --suite problem2_judge/data/suite.json       # pointwise report
python -m problem2_judge.judge compare  --suite problem2_judge/data/suite.json --configs problem2_judge/data/configs.json
python -m problem2_judge.validate all                                               # every bias + validation check
python -m problem2_judge.validate position --n 8                                    # or one at a time
```

Add `--limit 10` / `--n 4` to any command to use fewer API calls. Suites can be JSON or YAML.

Every judge call (full prompt, raw reply, tokens, parse errors) is appended to `problem2_judge/logs/judge_calls.jsonl`. Malformed JSON is repaired by a tolerant parser, then one retry that shows the model its own bad reply; if that fails the case is marked `ERROR` and excluded from means (and counted).

## Files

* `data/suite.json` - 27 test cases `{input, system_prompt, model_output, expected_output?, criteria?, human_score}`
* `data/probes.json` - correct / confidently-wrong / terse-but-correct answer for 8 questions (padded answers are built in code)
* `data/configs.json` - config A (verbose prompt) vs config B (careful prompt)
* Outputs in `results/`: `suite_report.json`, `compare_report.json`, `validation_report.json`

## Results

> Fill in after running (copy from the JSON files):
> A/B: pass rate A __ vs B __ | mean A __ vs B __ | pairwise wins A __ / B __ / tie __ | flip rate __ | winner __ (p = __)
> Position flip rate __ | verbosity gain naive __ -> hardened __ | fooled rate naive __ -> hardened __
> Agreement with human labels: naive __ (kappa __) -> hardened __ (kappa __) | test-retest flip rate __

## Discussion (Problem 2)

> Write this yourself from YOUR numbers. Questions to answer:
> 1. How biased was the judge before vs after the mitigations? (Use the naive vs hardened numbers.)
> 2. Would you let it gate a release? Under what conditions (agreement, kappa, human spot-check of FAILs, sample size)?
> 3. Limits of this validation: only 27 labelled cases, human labels written by one person, one judge model.
