"""Evaluation harness for the RAG service (three layers).

    python -m problem1_rag.evaluate                      # everything (needs API keys for layer 2)
    python -m problem1_rag.evaluate --retrieval-only     # layer 1 only, no API keys needed
    python -m problem1_rag.evaluate --ablation           # + compare chunk sizes (retrieval only)

Layer 1  Retrieval : Hit rate, Recall@k, MRR, nDCG@k, context precision
Layer 2  Answer    : faithfulness + relevance (LLM judge), EM / F1 vs gold answers, refusal accuracy
Layer 3  Cost/latency: p50/p95 retrieval latency, tokens, cost model (from cost_analysis.py)

HOW WE KNOW WHICH CHUNK IS "RELEVANT": every question in data/questions.json has an `evidence`
phrase copied from the source document. A chunk is relevant if it contains that phrase.
This works for any chunk size, so we can compare chunk settings fairly.
"""
import argparse
import json
import math
import re
import string
import tempfile
import time
from pathlib import Path

import numpy as np

from llm import LLM, chat_json
from problem1_rag import ingest, rag

HERE = Path(__file__).parent
RESULTS = HERE / "results"


# ------------------------------------------------------------------ layer 1: retrieval
def norm(text):
    """Lowercase, drop markdown symbols, collapse whitespace (so labels survive formatting)."""
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", text.lower())).strip()


def retrieval_scores(flags, total_relevant, k):
    """flags[i] is True if the chunk at rank i+1 is relevant. total_relevant = relevant chunks in the whole index."""
    hit = 1.0 if any(flags) else 0.0
    recall = sum(flags) / total_relevant if total_relevant else 0.0
    rr = 1.0 / (flags.index(True) + 1) if hit else 0.0                      # reciprocal rank
    dcg = sum(1.0 / math.log2(i + 2) for i, f in enumerate(flags) if f)    # binary relevance
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(total_relevant, k)))
    ndcg = dcg / ideal if ideal else 0.0
    precision_at_k = sum(flags) / k
    hits, precisions = 0, []                                                # context precision (RAGAS-style):
    for rank, f in enumerate(flags, 1):                                     # precision measured at each relevant
        if f:                                                               # rank, then averaged -> rewards putting
            hits += 1                                                       # relevant chunks FIRST
            precisions.append(hits / rank)
    context_precision = sum(precisions) / len(precisions) if precisions else 0.0
    return {"hit": hit, "recall": recall, "rr": rr, "ndcg": ndcg,
            "precision_at_k": precision_at_k, "context_precision": context_precision}


def mean_of(rows, key):
    return round(float(np.mean([r[key] for r in rows])), 4) if rows else None


def run_retrieval(questions, k, collection, use_filter=False, repeats=5):
    """Retrieve for every answerable question. Also times each call `repeats` times."""
    index_texts = [norm(t) for t in collection.get()["documents"]]
    rows, latencies = [], []
    rag.retrieve("warm up the embedding model", k, collection=collection)
    for q in questions:
        if not q["answerable"]:
            continue
        evidence = norm(q["evidence"])
        total_relevant = sum(evidence in t for t in index_texts)
        file_type = q["file_type"] if use_filter else None
        chunks = None
        for _ in range(repeats):
            start = time.perf_counter()
            chunks = rag.retrieve(q["question"], k, file_type=file_type, collection=collection)
            latencies.append((time.perf_counter() - start) * 1000)
        flags = [evidence in norm(c["text"]) for c in chunks]
        row = {"id": q["id"], "total_relevant_in_index": total_relevant,
               "relevant_ranks": [i + 1 for i, f in enumerate(flags) if f],
               **retrieval_scores(flags, total_relevant, k)}
        rows.append(row)
    summary = {
        "k": k, "n_questions": len(rows),
        "hit_rate": mean_of(rows, "hit"), "recall_at_k": mean_of(rows, "recall"), "mrr": mean_of(rows, "rr"),
        "ndcg_at_k": mean_of(rows, "ndcg"), "precision_at_k": mean_of(rows, "precision_at_k"),
        "context_precision": mean_of(rows, "context_precision"),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "questions_without_a_labelled_chunk": [r["id"] for r in rows if r["total_relevant_in_index"] == 0],
    }
    return summary, rows


def run_ablation(questions, k):
    """Re-ingest the corpus with different chunk sizes into temporary indexes and compare retrieval."""
    docs = HERE / "data" / "docs"
    out = []
    for size, overlap in [(250, 25), (500, 50), (1000, 100)]:
        col = rag.get_collection(path=tempfile.mkdtemp(), name="ablation")
        per_file = ingest.ingest_dir(col, docs, size, overlap)
        summary, _ = run_retrieval(questions, k, col, repeats=1)
        out.append({"chunk_size": size, "overlap": overlap, "n_chunks": sum(per_file.values()),
                    **{key: summary[key] for key in ("hit_rate", "recall_at_k", "mrr", "ndcg_at_k", "context_precision")}})
        print(f"  size={size:5d} overlap={overlap:4d} chunks={out[-1]['n_chunks']:4d} "
              f"hit={out[-1]['hit_rate']} mrr={out[-1]['mrr']} ndcg={out[-1]['ndcg_at_k']} ctx_prec={out[-1]['context_precision']}")
    return out


# ------------------------------------------------------------------ layer 2: answers
def squad_norm(text):
    text = re.sub(r"\[\d+\]", " ", text.lower())                       # drop citation markers like [1]
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(w for w in text.split() if w not in ("a", "an", "the"))


def f1_score(prediction, gold):
    p, g = squad_norm(prediction).split(), squad_norm(gold).split()
    common = sum(min(p.count(w), g.count(w)) for w in set(p) & set(g))
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


JUDGE_SYSTEM = """You grade the answer of a retrieval-augmented QA system.
You get the CONTEXT the system saw, the QUESTION and the ANSWER.
Task 1 (faithfulness): list every factual claim in the ANSWER (ignore citation markers like [1]).
        For each claim say whether the CONTEXT supports it (true/false). Do not use outside knowledge.
Task 2 (relevance): score from 1 to 5 how directly the ANSWER addresses the QUESTION
        (5 = fully and directly, 3 = partly, 1 = not at all).
Return ONLY this JSON:
{"claims": [{"claim": "...", "supported": true}], "relevance": 4, "relevance_reason": "one sentence"}"""


def check_judge_output(obj):
    if not isinstance(obj["claims"], list):
        raise ValueError("claims must be a list")
    for c in obj["claims"]:
        bool(c["supported"])
    if not 1 <= int(obj["relevance"]) <= 5:
        raise ValueError("relevance must be 1-5")


def judge_answer(judge, question, chunks, answer_text):
    context = "\n\n".join(f"[{c['n']}] {c['text']}" for c in chunks)
    messages = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER: {answer_text}"}]
    obj, calls = chat_json(judge, messages, check=check_judge_output)
    usage = {"prompt_tokens": sum(c["prompt_tokens"] for c in calls), "completion_tokens": sum(c["completion_tokens"] for c in calls)}
    if obj is None:
        return None, usage
    claims = obj["claims"]
    supported = sum(1 for c in claims if c["supported"] in (True, "true", "True"))
    return {"faithfulness": supported / len(claims) if claims else None,
            "relevance": int(obj["relevance"]), "n_claims": len(claims)}, usage


def run_answers(questions, k, collection, relevant_hit):
    judge = LLM("JUDGE")
    rows, judge_tokens = [], {"prompt_tokens": 0, "completion_tokens": 0}
    for q in questions:
        res = rag.answer(q["question"], k=k, collection=collection)
        row = {"id": q["id"], "question": q["question"], "answerable_question": q["answerable"],
               "answer": res["answer"], "system_answered": res["answerable"], "reason": res["reason"],
               "top_similarity": res["retrieved"][0]["similarity"] if res["retrieved"] else None,
               "n_citations": len(res["citations"]), **res["metrics"]}
        if q["answerable"]:
            row["em"] = float(squad_norm(res["answer"]) == squad_norm(q["gold_answer"]))
            row["f1"] = round(f1_score(res["answer"], q["gold_answer"]), 4)
            row["contains_gold"] = float(squad_norm(q["gold_answer"]) in squad_norm(res["answer"]))
            row["retrieval_hit"] = relevant_hit.get(q["id"])
            if res["answerable"]:
                try:
                    verdict, usage = judge_answer(judge, q["question"], res["retrieved"], res["answer"])
                except Exception as err:  # one failed judge call must not lose the whole run
                    print(f"  WARNING judge call failed for {q['id']}: {type(err).__name__}")
                    verdict, usage = None, {"prompt_tokens": 0, "completion_tokens": 0}
                judge_tokens["prompt_tokens"] += usage["prompt_tokens"]
                judge_tokens["completion_tokens"] += usage["completion_tokens"]
                row["judge"] = verdict
        rows.append(row)
        print(f"  {q['id']} answered={res['answerable']!s:5} | {res['answer'][:90]!r}")
    return rows, judge_tokens


def summarize_answers(rows):
    answerable = [r for r in rows if r["answerable_question"]]
    unanswerable = [r for r in rows if not r["answerable_question"]]
    judged = [r["judge"] for r in answerable if r.get("judge")]
    faith = [j["faithfulness"] for j in judged if j["faithfulness"] is not None]
    # diagnosis: was the failure in retrieval or in generation?
    correct = lambda r: r["contains_gold"] == 1.0 or r["f1"] >= 0.5
    diagnosis = {
        "retrieved_and_correct": sum(1 for r in answerable if r["retrieval_hit"] and correct(r)),
        "retrieved_but_wrong (generation problem)": sum(1 for r in answerable if r["retrieval_hit"] and not correct(r)),
        "not_retrieved_and_wrong (retrieval problem)": sum(1 for r in answerable if not r["retrieval_hit"] and not correct(r)),
        "not_retrieved_but_correct": sum(1 for r in answerable if not r["retrieval_hit"] and correct(r)),
    }
    return {
        "n_answerable": len(answerable), "n_unanswerable": len(unanswerable),
        "faithfulness": round(float(np.mean(faith)), 4) if faith else None,
        "answer_relevance_1to5": round(float(np.mean([j["relevance"] for j in judged])), 3) if judged else None,
        "answer_relevance_0to1": round(float(np.mean([(j["relevance"] - 1) / 4 for j in judged])), 4) if judged else None,
        "judged_answers": len(judged), "judge_failed_to_parse": sum(1 for r in answerable if r["system_answered"] and not r.get("judge")),
        "exact_match": mean_of(answerable, "em"), "f1": mean_of(answerable, "f1"),
        "contains_gold_answer": mean_of(answerable, "contains_gold"),
        "wrongly_refused_answerable": sum(1 for r in answerable if not r["system_answered"]),
        "correctly_refused_unanswerable": sum(1 for r in unanswerable if not r["system_answered"]),
        "diagnosis": diagnosis,
        "top1_similarity_answerable_min": min((r["top_similarity"] for r in answerable if r["top_similarity"] is not None), default=None),
        "top1_similarity_unanswerable_max": max((r["top_similarity"] for r in unanswerable if r["top_similarity"] is not None), default=None),
    }


def summarize_latency_and_tokens(rows, llm_prefix="GEN"):
    answered = [r for r in rows if r["llm_ms"] > 0]
    totals = [r["total_ms"] for r in rows]
    prompt = sum(r["prompt_tokens"] for r in rows)
    completion = sum(r["completion_tokens"] for r in rows)
    cost = LLM(llm_prefix).cost(prompt, completion)
    return {
        "end_to_end_p50_ms": round(float(np.percentile(totals, 50)), 1),
        "end_to_end_p95_ms": round(float(np.percentile(totals, 95)), 1),
        "llm_calls": len(answered),
        "avg_prompt_tokens": round(prompt / max(len(answered), 1), 1),
        "avg_completion_tokens": round(completion / max(len(answered), 1), 1),
        "generation_cost_usd_total": round(cost, 5),
        "generation_cost_usd_per_1000_queries": round(cost / max(len(answered), 1) * 1000, 4),
    }


# ------------------------------------------------------------------ report
def write_markdown(res):
    r, rf = res["retrieval"], res["retrieval_with_file_type_filter"]
    lines = [f"# RAG evaluation results (k = {r['k']}, {r['n_questions']} answerable questions)", "",
             "## Retrieval", "", "| Metric | No filter | With file_type filter |", "|---|---|---|"]
    for label, key in [("Hit rate", "hit_rate"), (f"Recall@{r['k']}", "recall_at_k"), ("MRR", "mrr"),
                       (f"nDCG@{r['k']}", "ndcg_at_k"), (f"Precision@{r['k']}", "precision_at_k"),
                       ("Context precision", "context_precision")]:
        lines.append(f"| {label} | {r[key]} | {rf[key]} |")
    lines += ["", f"Retrieval latency (embedding + search, local): p50 = {r['latency_p50_ms']} ms, p95 = {r['latency_p95_ms']} ms", ""]
    if res.get("ablation"):
        lines += ["## Chunk size ablation (retrieval only)", "", "| Size | Overlap | Chunks | Hit | MRR | nDCG | Ctx precision |", "|---|---|---|---|---|---|---|"]
        for a in res["ablation"]:
            lines.append(f"| {a['chunk_size']} | {a['overlap']} | {a['n_chunks']} | {a['hit_rate']} | {a['mrr']} | {a['ndcg_at_k']} | {a['context_precision']} |")
        lines.append("")
    if res.get("answers"):
        a, lt = res["answers"], res["latency_and_tokens"]
        lines += ["## Answer quality", "", "| Metric | Score |", "|---|---|",
                  f"| Faithfulness (supported claims / all claims) | {a['faithfulness']} |",
                  f"| Answer relevance (1-5) | {a['answer_relevance_1to5']} |",
                  f"| Exact match | {a['exact_match']} |", f"| F1 | {a['f1']} |",
                  f"| Contains gold answer | {a['contains_gold_answer']} |",
                  f"| Correct refusals on unanswerable questions | {a['correctly_refused_unanswerable']} / {a['n_unanswerable']} |",
                  f"| Wrong refusals on answerable questions | {a['wrongly_refused_answerable']} / {a['n_answerable']} |", "",
                  "Retrieval vs generation diagnosis: " + json.dumps(a["diagnosis"]), "",
                  "## End-to-end latency and tokens", "",
                  f"p50 = {lt['end_to_end_p50_ms']} ms, p95 = {lt['end_to_end_p95_ms']} ms, avg tokens/query = "
                  f"{lt['avg_prompt_tokens']} in + {lt['avg_completion_tokens']} out, "
                  f"generation cost = ${lt['generation_cost_usd_per_1000_queries']} per 1000 queries", ""]
    if res.get("cost_and_scale"):
        lines += ["## Cost (see results/cost_table.md)", "", "```", json.dumps(res["cost_and_scale"].get("bench", []), indent=1), "```"]
    (RESULTS / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=rag.DEFAULT_K)
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM layers (no API keys needed)")
    parser.add_argument("--ablation", action="store_true", help="also compare chunk sizes")
    parser.add_argument("--repeats", type=int, default=5, help="timing repeats per question")
    args = parser.parse_args()

    questions = json.loads((HERE / "data" / "questions.json").read_text(encoding="utf-8"))
    collection = rag.get_collection()
    if collection.count() == 0:
        raise SystemExit("The index is empty. Run first:  python -m problem1_rag.ingest")

    print(f"Index: {collection.count()} vectors | k={args.k} | {len(questions)} questions")
    results = {"config": {"k": args.k, "embedding_model": rag.EMBED_MODEL, "embedding_dim": rag.EMBED_DIM,
                          "vectors": collection.count(), "min_similarity": rag.MIN_SIMILARITY}}
    if not args.retrieval_only:
        gen, judge = LLM("GEN"), LLM("JUDGE")
        results["config"]["generator"] = f"{gen.provider}/{gen.model}"
        results["config"]["judge"] = f"{judge.provider}/{judge.model}"

    print("\nLayer 1: retrieval")
    summary, per_question = run_retrieval(questions, args.k, collection, False, args.repeats)
    summary_f, _ = run_retrieval(questions, args.k, collection, True, 1)
    results["retrieval"], results["retrieval_with_file_type_filter"], results["retrieval_per_question"] = summary, summary_f, per_question
    print(json.dumps(summary, indent=1))
    if summary["questions_without_a_labelled_chunk"]:
        print("WARNING: evidence phrase not found in any chunk for", summary["questions_without_a_labelled_chunk"])

    if args.ablation:
        print("\nChunk size ablation")
        results["ablation"] = run_ablation(questions, args.k)

    if not args.retrieval_only:
        print("\nLayer 2: answers")
        hit_by_id = {r["id"]: r["hit"] == 1.0 for r in per_question}
        rows, judge_tokens = run_answers(questions, args.k, collection, hit_by_id)
        results["answers"] = summarize_answers(rows)
        results["answer_rows"] = rows
        results["latency_and_tokens"] = summarize_latency_and_tokens(rows)
        results["judge_tokens"] = judge_tokens
        print(json.dumps({**results["answers"], **results["latency_and_tokens"]}, indent=1))

    cost_file = RESULTS / "cost.json"
    if cost_file.exists():
        results["cost_and_scale"] = json.loads(cost_file.read_text())

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(results)
    print("\nSaved problem1_rag/results/results.json and results.md")


if __name__ == "__main__":
    main()
