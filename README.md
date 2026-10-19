# Applied AI / ML Engineering take-home

Two projects in one repo:

| Folder | What it is |
|---|---|
| `problem1_rag/` | Cost-efficient RAG service on **ChromaDB**, evaluation harness, cost model |
| `problem2_judge/` | **LLM-as-Judge** pipeline, bias measurements, judge validation, A/B comparison |
| `llm.py` | Shared helper: one OpenAI-compatible client for OpenAI / Gemini / Groq / OpenRouter, tolerant JSON parsing, rate-limit waiting |
| `PROBLEMS_AND_FIXES.md` | Log of every bug or surprise I hit and how it was fixed |

## Setup

```bash
git clone https://github.com/Rudra-323/GenAI_Task.git
cd GenAI_Task
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
Always run scripts from the repo root with `python -m ...`. Model names and free-tier limits change, so check your own provider console if a model returns 404.

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
* **API** (`app.py`, FastAPI): `POST /query` and `GET /health`. One log line per query: retrieval ms, LLM ms, chunk count, tokens (also saved in `logs/queries.jsonl`).

## Run

```bash
python -m problem1_rag.ingest                       # run twice: same vector count (idempotent)
uvicorn problem1_rag.app:app --port 8000            # start the service
python -m problem1_rag.cost_analysis --bench        # cost table + latency at 10K / 100K vectors
python -m problem1_rag.evaluate --ablation          # retrieval + answer + latency evaluation
```

### Send a query, step by step
1. Open a terminal in the repo root and activate the environment (`.venv\Scripts\activate`).
2. Make sure the index exists (`python -m problem1_rag.ingest`).
3. Start the server: `uvicorn problem1_rag.app:app --port 8000` and keep that terminal open.
4. Open `http://localhost:8000/docs` in a browser, click `POST /query`, click **Try it out**, and send:
   ```json
   {"question": "What is the rate limit for the Pro plan?", "k": 5}
   ```
5. The response has `answer` (with `[n]` citations), `answerable`, `citations`, `retrieved` chunks and `metrics`. The server terminal prints one line with `retrieval`, `llm` and `total` time, chunk count and tokens.
6. Metadata filter example: `{"question": "How many days of annual leave?", "k": 3, "file_type": "pdf"}`.
7. No-answer example: `{"question": "Does the Zentra API support GraphQL queries?", "k": 5}` returns "I don't know based on the provided documents."

Windows `cmd` does not accept single quotes in curl, so use the `/docs` page, or:
`curl -X POST localhost:8000/query -H "Content-Type: application/json" -d "{\"question\":\"What is the rate limit for the Pro plan?\",\"k\":5}"`

### Use your own documents
Loading documents is a command, not an upload endpoint:
```bash
set CHROMA_PATH=my_index                    # optional: a separate index so the project results are not changed
set COLLECTION_NAME=mydocs
python -m problem1_rag.ingest --docs my_docs   # a folder with .pdf, .html, .md or .txt files
uvicorn problem1_rag.app:app --port 8000       # same terminal
```

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

Answer quality:

| Metric | Value |
|---|---|
| Faithfulness (LLM judge) | 1.0 |
| Answer relevance (LLM judge, 1-5) | 5.0 |
| Exact match | 0.0 (gold answers are short phrases, the model writes full sentences) |
| F1 | 0.43 |
| Contains the gold answer | 0.73 |
| Share of gold words found in the answer | 0.93 |
| Unanswerable questions refused correctly | 4 of 4 |
| Answerable questions wrongly refused | 1 of 22 (q19) |

Faithfulness 1.0 and relevance 5.0 are a ceiling result: the answers are short and quote the documents, so this does not mean the judge is perfect.

**Retrieval or generation?** Of the 22 answerable questions, 20 were retrieved and answered correctly, 1 was retrieved but answered wrongly (a generation problem), and 1 was not retrieved (a retrieval problem, q19). 0 were correct without retrieval.

Latency (44 vectors, k = 5, my laptop CPU, 5 repeats per question):

| | p50 | p95 |
|---|---|---|
| Retrieval (embed the question + search) | 237 ms | 345 ms |
| End to end (retrieval + Groq LLM call) | 1489 ms | 6640 ms |

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
* Smaller chunks scored best here (250 beat 500), because each question needs one short fact. The cost is about 2x the vectors and less context per chunk. I kept 500 as the default.
* q19 ("Pro rate limit *before* it was raised") was a real retrieval miss: the API-guide chunk outranked the release-notes chunk.
* Against always-on pods, self-hosting is cheaper (most at 10M vectors). Against serverless at low query volume, managed is cheaper because of its $50 floor. At high query volume serverless gets expensive.
* Chroma's metadata filter was much slower at 100K vectors (122.7 ms vs 2.2 ms p50) in my benchmark.

## Discussion (Problem 1)

> Write this yourself, 6-8 sentences, in your own words. Use the evidence above and your own docx boxes (1.4 design decisions).
> 1. **When would you switch back to managed?** Evidence: the cost table (pods vs serverless, low vs high query volume), no HA replica, index must fit in RAM, filter latency at 100K, engineer time not included.
> 2. **Was retrieval or generation the weak link?** Evidence: 20 correct, 1 generation problem, 1 retrieval problem (q19); faithfulness is at the ceiling; the threshold could not separate GraphQL from real questions.
> 3. **Which trade-offs did you accept?** Evidence: single machine, small corpus, 384-d model, chunk size 500 although 250 scored higher.
>
> `<<WRITE YOURSELF>>`

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
Sample sizes are small on purpose: the free Groq tier allows 8K tokens per minute and 200K per day for the judge model. Waits of several minutes are normal; the script prints them.

## Results

**Suite run** (27 cases, pointwise): pass rate 0.63, mean overall 3.84. Mean per criterion: correctness 3.67, faithfulness 3.67, completeness 3.85, instruction following 3.85, tone/safety 4.70. No parse failures.

**A/B comparison** (10 cases, config A = verbose prompt, config B = careful prompt):

| Config | Pass rate | Mean score | Pairwise wins | Win rate |
|---|---|---|---|---|
| A (verbose) | 0.90 | 4.65 | 0 | 0.10 |
| B (careful) | 1.00 | 5.00 | 8 | 0.90 |

2 ties, order-flip rate 0.0, sign test p = 0.0078. **Winner: B** (statistically significant, but only 10 cases and B is at the score ceiling).
Example: on "Who wrote Pride and Prejudice?" config A (verbose prompt) produced a long answer with invented details and a runaway list of years, scoring 2.2; config B gave one correct sentence and scored 5.0. The low score was deserved, not a length bias.

**Position bias** (6 questions x 4 pair types, both orders): overall flip rate 4.2% (1 of 24 pairs). Identical answers: 6 of 6 ties. First-position share 0.483. I found **no position bias** in this judge, so the both-orders check is a safeguard that had nothing to fix here.

**Verbosity and confident-wrong probe** (4 questions):

| | Naive judge | Hardened judge |
|---|---|---|
| Padded answer scored higher than the original | 0 of 4 | 0 of 4 |
| Padding score change (ideal <= 0) | -0.21 | -0.60 |
| Confidently wrong answer judged PASS | 0 of 4 | 0 of 4 |
| Terse correct answer judged FAIL | 0 of 4 | 0 of 4 |

Neither judge was fooled. The hardened one penalises padding more, as intended.

**Agreement with human labels** (27 cases, labels drafted with AI help, see Limitations):

| | Naive | Hardened |
|---|---|---|
| PASS/FAIL agreement | 88.9% | 88.9% |
| Cohen's kappa | 0.776 | 0.776 |
| Spearman correlation | 0.953 | 0.967 |
| False passes / false fails | 3 / 0 | 3 / 0 |

The hardened prompt did **not** clearly improve agreement here; the base judge was already strong. Scores sit at the extremes (1 and 5), not clustered at 4-5.
The 3 false passes (c08, c15, c27) are cases where the facts were right but a format rule was broken. For all three, the judge gave instruction following = 1 and every other criterion = 5, but the weighted average (overall 4.4) still passed them. The weakness is in my aggregation rule, not in what the judge saw. A rule "any criterion below 3 means FAIL" would fix all 3 on this suite (27 of 27), but I chose that rule after looking at these same cases, so that number is optimistic and must be tested on new cases.

**Self-enhancement** (3 cases): same-family and different-family judges both gave 5.0 to the generator's own answers. The estimate (+0.33) comes from the neutral answers, so this is **no evidence** of self-enhancement at this sample size.

**Test-retest** (judge temperature 0): 5 cases x 2 runs. PASS/FAIL flip rate 0%, and all overall scores were identical in both runs.

## Discussion (Problem 2)

> Write this yourself, in your own words. Use the evidence above and your own docx boxes (2.4 design decisions).
> 1. **How biased was the judge before vs after the mitigations?** Evidence: no position bias (4% flips, 6 of 6 ties); not fooled by padding or confident-wrong answers, with or without the hardened prompt; agreement and kappa identical for naive and hardened; the weak spot was the weighted average, not the judge.
> 2. **Would you let it gate a release? Under what conditions?** Evidence: kappa 0.78 on 27 cases, 3 false passes from format rules, the floor-rule idea (tested only on the same cases), human review of FAILs.
> 3. **What are the limits of this validation?** Evidence: see the Limitations below.
>
> `<<WRITE YOURSELF>>`

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
* The 27 human labels were drafted with AI help (Claude) and I did not independently verify all of them, so agreement is measured against AI-drafted labels.
* One judge model. The fictional corpus is small and clean, so real documents would be harder.
* Documents are loaded with the ingest command; there is no upload endpoint. Loaders are light (pypdf, BeautifulSoup), so scanned PDFs and tables are not handled; Docling would be the next step for those.
* Retrieval latency was measured on my laptop (it varied from run to run); the synthetic benchmark uses random vectors.
* No second vector store was benchmarked, and the managed-database costs are a model from public prices, not a measured bill.

## AI assistance
**Problem 1.** I used Claude (the AI chat assistant) to plan the approach, write the code, fix errors, prepare the sample documents and questions, and analyze the evaluation results. I used GPT-OSS 20B on Groq to generate answers, Qwen3.8 27B on Groq to judge faithfulness and relevance, and all-MiniLM-L6-v2 locally to generate embeddings. My contribution was setting up the environment, configuring my API keys, running all the commands, checking the outputs, collecting screenshots, and creating and updating the GitHub repository. The results reported are from runs on my machine. I also studied the implementation and evaluation results with AI assistance.

**Problem 2.** I used Claude (the AI chat assistant) to design the judging method and rubric, write the judge and validation code and prompts, prepare the test cases, probe questions and draft human scores, and create the architecture diagram. I used GPT-OSS 20B on Groq to generate answers for the A/B comparison and self-enhancement test. Qwen3.8 27B on Groq acted as the judge for every evaluation run. My contribution was running all the commands, handling Groq token limits and waiting periods, checking the results, taking screenshots, and pushing the project to GitHub. The test cases and draft human scores were prepared with AI assistance. I have not independently verified all 27 human labels. I also studied the evaluation process with AI assistance.
