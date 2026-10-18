# Applied AI / ML Engineering take-home

Two projects in one repo:

| Folder | What it is |
|---|---|
| `problem1_rag/` | Cost-efficient RAG service on **ChromaDB**, evaluation harness, cost model |
| `problem2_judge/` | **LLM-as-Judge** pipeline, bias measurements, judge validation, A/B comparison |
| `llm.py` | Shared helper: one OpenAI-compatible client for OpenAI / Gemini / Groq / OpenRouter, tolerant JSON parsing, rate-limit waiting |
| `PROBLEMS_AND_FIXES.md` | Log of every bug / surprise I hit and how it was fixed |

Lines marked `<<FILL: ...>>` are numbers or text I still have to copy from my own result files.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # Mac/Linux: cp .env.example .env   (then paste your API keys into .env)
```

Models are chosen in `.env` only (no code change):

| Role | Setting I used | Why |
|---|---|---|
| Generator (writes answers) | Groq `openai/gpt-oss-20b` | The Llama model I first tried was not available on my Groq key |
| Judge (grades answers) | Groq `qwen/qwen3.8-27b` | Different model family from the generator (OpenAI vs Alibaba), to avoid self-enhancement bias. Gemini's free tier only allows 20 requests per day, too few |
| Embeddings (P1) | `all-MiniLM-L6-v2`, 384 dimensions, local | $0 per call, small memory |

Secrets are read from environment variables. `.env` is git-ignored; `.env.example` has names only.
Always run scripts from the repo root with `python -m ...`.

---

# Problem 1: Cost-efficient RAG

## Store choice: ChromaDB (embedded, persistent)

| Why | Trade-off I accept |
|---|---|
| Runs inside my app, saves to disk, has HNSW index, cosine distance and metadata filters. No per-vector bill | One machine, no replica / HA |
| I already knew it, so time went into evaluation | The index must fit in RAM, so RAM sizes the VM |
| | Filtered search got slow at 100K vectors in my benchmark (see below) |

I did not benchmark a second store.

## How it works
* **Ingest** (`ingest.py`): PDF (pypdf), HTML (BeautifulSoup), Markdown/TXT. **Chunk size 500 chars, overlap 50** (change with `--chunk-size` / `--overlap` or env).
* **Idempotent**: chunk id = hash of file name + chunk number + text, and a file's old chunks are deleted before it is inserted. Running ingest twice: 44 vectors both times.
* **Metadata**: `source`, `file_type`, `chunk_index`, `page`. Filter by `file_type` and/or `source`.
* **Answer** (`rag.py`): top-k (k is a parameter) -> numbered context -> the LLM must cite `[n]`.
  **No-context handling has two guards:** (1) if the best similarity is below `MIN_SIMILARITY` (0.25) the LLM is not called; (2) the prompt tells the LLM to reply "I don't know based on the provided documents." when the chunks lack the answer.
* **API** (`app.py`, FastAPI) and one log line per query: retrieval ms, LLM ms, chunk count, tokens (`logs/queries.jsonl`).

## Run
```bash
python -m problem1_rag.ingest                       # run twice: same vector count
uvicorn problem1_rag.app:app --port 8000            # then open http://localhost:8000/docs  (POST /query, "Try it out")
python -m problem1_rag.cost_analysis --bench        # cost table + latency at 10K / 100K vectors
python -m problem1_rag.evaluate --ablation          # retrieval + answer + latency evaluation
```
Windows `cmd` does not accept single quotes in curl, so use the `/docs` page, or:
`curl -X POST localhost:8000/query -H "Content-Type: application/json" -d "{\"question\":\"What is the rate limit for the Pro plan?\",\"k\":5}"`

## Evaluation design
* **Corpus**: 6 documents (MD, HTML, PDF) about a *fictional* company, so the LLM cannot answer from memory. It has near-duplicate facts on purpose (e.g. rate limits appear in two documents). 44 chunks.
* **Questions**: `data/questions.json`, 22 answerable + 4 unanswerable.
* **Which chunk is relevant**: each answerable question has an `evidence` phrase from its source document. A chunk is relevant if it contains the phrase. This works for any chunk size, so chunk settings can be compared fairly.
* **Metrics**: Hit rate, Recall@k, MRR, nDCG@k, Precision@k, context precision (RAGAS style). Answers: faithfulness (supported claims / all claims) and relevance (1-5) by an LLM judge from a different family, plus EM, F1, contains-gold, gold word recall, and refusal accuracy.

## Results (k = 5, 22 answerable questions)

| Retrieval metric | Value |
|---|---|
| Hit rate = Recall@5 | 0.955 |
| MRR | 0.849 |
| nDCG@5 | 0.876 |
| Precision@5 | 0.200 |
| Context precision | 0.849 |

Chunk size comparison (retrieval only):

| Size / overlap | Chunks | Hit | MRR | nDCG | Ctx precision |
|---|---|---|---|---|---|
| 250 / 25 | 90 | 1.000 | 0.947 | 0.961 | 0.947 |
| **500 / 50 (default)** | 44 | 0.955 | 0.849 | 0.876 | 0.849 |
| 1000 / 100 | 20 | 0.955 | 0.721 | 0.782 | 0.723 |

Answer layer: `<<FILL from problem1_rag/results/results.md: faithfulness, relevance, EM, F1, contains-gold, diagnosis>>`.
Refusals: 4 of 4 unanswerable questions were refused correctly; 1 answerable question (q19) was wrongly refused.
Latency: `<<FILL: retrieval p50 / p95 and end-to-end p95 from results.md>>` (my laptop was busy during some runs, so latency varied from run to run).

Retrieval latency on synthetic 384-d vectors (k = 5, 200 queries):

| Vectors | p50 | p95 | p50 with metadata filter |
|---|---|---|---|
| 10,000 | 1.5 ms | 2.3 ms | 10.5 ms |
| 100,000 | 2.2 ms | 3.5 ms | 122.7 ms |

### Cost comparison (monthly, USD)

384 dimensions, float32. Assumptions are in `problem1_rag/cost_analysis.py` (prices checked Oct 2026: Pinecone pricing + docs, AWS us-east-1 on-demand). Self-hosted = one VM with enough RAM for the HNSW index, plus disk and snapshots, **no HA replica**. Managed pods = always-on, billed per hour. Serverless = $0.33/GB-month storage + $16 per million read units + $4 per million write units, $50 monthly minimum. Engineer time is not included.

| Vectors | Self-hosted Chroma | Pods p1 | Pods s1 | Serverless, 100K queries/mo | Serverless, 5M queries/mo |
|---|---|---|---|---|---|
| 100K | $32 (t3.medium) | $70 | $70 | $50 | $50 |
| 1M | $63 (t3.large) | $70 | $70 | $50 | $164 |
| 10M | $191 (r6i.xlarge) | $350 | $70 | $50 | $1,644 |

### Findings worth knowing
* The similarity threshold alone cannot separate "no answer" questions: the lowest answerable top-1 similarity was 0.42, but the unanswerable GraphQL question reached 0.56. Only the World Cup question was stopped by the threshold; the other three were refused by the LLM following the prompt. Hence two guards.
* Smaller chunks scored best here (250 beat 500), because each question needs one short fact. The cost is about 2x the vectors and less context per chunk.
* q19 ("Pro rate limit *before* it was raised") was a real retrieval miss: the API-guide chunk outranked the release-notes chunk.
* Against always-on pods, self-hosting is cheaper (most at 10M vectors). Against serverless at low query volume, managed is cheaper because of its $50 floor. At high query volume serverless gets expensive.

## Discussion (Problem 1)

> Write this yourself, 6-8 sentences, from your own results. Answer:
> 1. When would you switch back to managed? (use the cost table, HA, team size, index size)
> 2. Was retrieval or generation the weak link? (use the `diagnosis` counts and the failed question ids)
> 3. Which trade-offs did you accept?
>
> `<<FILL: your own words>>`

---

# Problem 2: LLM-as-Judge

## Judging modes
| Mode | Used for | Why |
|---|---|---|
| Pointwise, reference-based | `run`, when a case has `expected_output` | Best for correctness |
| Pointwise, reference-free | `run`, when there is no reference | Open tasks (tone, format) |
| Pairwise A-vs-B, both orders | `compare` | Judges are better at "which is better" than at absolute scores |

**Rubric** (1-5 each, with anchors and a rationale): correctness 0.35, faithfulness 0.20, completeness 0.20, instruction_following 0.15, tone_safety 0.10. The overall score is the weighted mean computed in code. PASS if overall >= 4.0.

**Robust parsing**: tolerant JSON extraction (fences, `<think>`, trailing commas), then one repair retry, then `ERROR` (counted, never dropped silently). Every judge call (full prompt, raw reply, tokens) is logged to `problem2_judge/logs/judge_calls.jsonl`.

## Bias handling

| Bias | Mitigation in code | How I measured it |
|---|---|---|
| Position | Judge each pair in both orders; a win needs both to agree, else tie | `validate.py position` |
| Verbosity | Rubric and prompt say padding never raises a score | `validate.py bias` (padded probe) |
| Self-enhancement | Judge family differs from generator family | `validate.py self` |
| Sycophancy / style | Evidence-based rationale per criterion; confident-wrong probes | `validate.py bias` |
| Score clustering | Few-shot anchors, "use the full range" rule | `validate.py agreement` |

The "naive" judge (no rules, no anchors) exists only to give the *before* number.

## Run
```bash
python -m problem2_judge.judge run                       # 27-case suite -> results/suite_report.json
python -m problem2_judge.judge compare --limit 10        # prompt v1 vs v2 -> results/compare_report.json
python -m problem2_judge.validate position --n 6
python -m problem2_judge.validate bias --n 4
python -m problem2_judge.validate agreement
python -m problem2_judge.validate retest --n 5 --runs 2
python -m problem2_judge.validate self --n 3
```
Sample sizes are small on purpose: the free Groq tier allows 8K tokens per minute and 200K per day for the judge model.

## Results

**Suite run** (27 cases, pointwise): pass rate 0.63, mean overall 3.84. Mean per criterion: correctness 3.67, faithfulness 3.67, completeness 3.85, instruction following 3.85, tone/safety 4.70. No parse failures.

**A/B comparison** (10 cases, config A = verbose prompt, config B = careful prompt):

| Config | Pass rate | Mean score | Pairwise wins | Win rate |
|---|---|---|---|---|
| A (verbose) | 0.90 | 4.65 | 0 | 0.10 |
| B (careful) | 1.00 | 5.00 | 8 | 0.90 |

2 ties, order-flip rate 0.0, sign test p = 0.0078. **Winner: B** (statistically significant, but only 10 cases and B is at the score ceiling).

**Position bias** (6 questions x 4 pair types, both orders): overall flip rate 4.2% (1 of 24 pairs). Identical answers: 6 of 6 ties. First-position share 0.483. I found **no position bias** in this judge, so the both-orders check is a safeguard that had nothing to fix here.

**Verbosity and confident-wrong probe** (4 questions):

| | Naive judge | Hardened judge |
|---|---|---|
| Padded answer scored higher than the original | 0 of 4 | 0 of 4 |
| Padding score change (ideal <= 0) | -0.21 | -0.60 |
| Confidently wrong answer judged PASS | 0 of 4 | 0 of 4 |
| Terse correct answer judged FAIL | 0 of 4 | 0 of 4 |

Neither judge was fooled. The hardened one penalises padding more, as intended.

**Agreement with human labels** (27 cases):

| | Naive | Hardened |
|---|---|---|
| PASS/FAIL agreement | 88.9% | 88.9% |
| Cohen's kappa | 0.776 | 0.776 |
| Spearman correlation | 0.953 | 0.967 |
| False passes / false fails | 3 / 0 | 3 / 0 |

The hardened prompt did **not** clearly improve agreement here; the base judge was already strong. Scores sit at the extremes (1 and 5), not clustered at 4-5.
The 3 false passes are cases where the facts were right but a format rule was broken. The judge gave those answers 1 for instruction following, but the weighted average (overall 4.4) still passed them. The weakness is in my aggregation rule, not in what the judge saw. A rule "any criterion below 3 means FAIL" would fix all 3 on this suite (27 of 27), but I chose that rule after looking at these same cases, so that number is optimistic and must be tested on new cases.

**Self-enhancement** (3 cases): same-family and different-family judges both gave 5.0 to the generator's own answers. The estimate (+0.33) comes from the neutral answers, so this is **no evidence** of self-enhancement at this sample size.

**Test-retest** (judge temperature 0): `<<FILL from validation_report.json -> test_retest: flip rate, n cases, runs>>`

## Discussion (Problem 2)

> Write this yourself from your own numbers. Answer:
> 1. How biased was the judge before vs after the mitigations? (use the naive vs hardened numbers honestly: where did it change, where not?)
> 2. Would you let it gate a release? Under what conditions?
> 3. What are the limits of this validation? (27 cases, one labeller, one judge model, tiny probe samples)
>
> `<<FILL: your own words>>`

---

## Problems I hit and how I fixed them
The full log with causes is in `PROBLEMS_AND_FIXES.md`. Short version:
* curl with single quotes failed on Windows (422). Used the `/docs` page.
* Groq returned 404 for the Llama model I first chose. I listed the models my key can use and picked `openai/gpt-oss-20b`.
* OpenAI gave 400 (typo in the model name) and then 429 (no credit). I stopped using it.
* Gemini's free tier allowed 20 requests per day, so the judge moved to Qwen on Groq.
* One failed judge call lost a whole evaluation run. I added error handling per call.
* Invisible Unicode characters (non-breaking hyphens and spaces) made correct answers fail the EM/F1 check. I normalised the text.
* Windows read JSON with the wrong encoding. I set `encoding="utf-8"`.
* Groq's token limits made the validation runs slow. I made the waits visible and used small samples.

## Limitations
* Small samples (n = 3 to 6 for bias checks, 27 labelled cases, 10 A/B cases) because of free-tier limits. Treat percentages as indicative.
* Human labels were checked by one person (me). `<<FILL: who labelled the 27 cases>>`
* One judge model. The fictional corpus is small and clean, so real documents would be harder.
* Retrieval latency was measured on a busy laptop; the synthetic benchmark uses random vectors.

## AI assistance
`<<FILL in your own words: which AI tools you used, for what, and what you did yourself>>`
