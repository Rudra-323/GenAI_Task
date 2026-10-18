"""Monthly cost: self-hosted Chroma on one VM  vs  managed Pinecone, at 100K / 1M / 10M vectors.

    python -m problem1_rag.cost_analysis                    # prints + saves the cost table (no API keys needed)
    python -m problem1_rag.cost_analysis --bench            # also measures query latency at 10K and 100K vectors
    python -m problem1_rag.cost_analysis --bench --bench-sizes 10000 100000 1000000

Every number comes from the ASSUMPTIONS block below. Prices were checked in Oct 2026 on:
  Pinecone pricing page + docs (serverless Standard, pod-based indexes), and public AWS us-east-1 on-demand
  EC2 prices. Prices change - re-check them before you submit and edit the block if needed.
"""
import argparse
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT_JSON = HERE / "results" / "cost.json"
OUT_MD = HERE / "results" / "cost_table.md"

ASSUMPTIONS = {
    "embedding_dim": 384,                  # all-MiniLM-L6-v2
    "bytes_per_float": 4,                  # float32
    "hours_per_month": 730,
    "queries_per_month_low": 100_000,      # "lightly queried" index (~0.04 queries per second)
    "queries_per_month_high": 5_000_000,   # busy index (~2 queries per second)
    # --- self-hosted: Chroma in one VM, no replica / HA (a trade-off we accept) ---
    "hnsw_link_bytes_per_vector": 128,     # graph links, M=16
    "ram_overhead_factor": 1.3,            # headroom for allocator / Python
    "base_ram_gb": 2.0,                    # OS + API + embedding model
    "disk_bytes_text_and_metadata": 1100,  # ~800 chars chunk text + ~300 B metadata per vector
    "ebs_gp3_per_gb_month": 0.08,
    "snapshot_per_gb_month": 0.05,
    "instances": [                         # (name, RAM GiB, $/hour) AWS us-east-1 on-demand Linux
        ("t3.small", 2, 0.0208), ("t3.medium", 4, 0.0416), ("t3.large", 8, 0.0832),
        ("r6i.large", 16, 0.126), ("r6i.xlarge", 32, 0.252), ("r6i.2xlarge", 64, 0.504),
    ],
    # --- managed: Pinecone pod-based (always-on pods, billed per hour whether queried or not) ---
    "pod_price_per_hour": 0.096,           # p1.x1 and s1.x1 starting price
    "p1_vectors_per_pod_768d": 1_000_000,  # performance pod
    "s1_vectors_per_pod_768d": 5_000_000,  # storage pod (slower queries)
    # --- managed: Pinecone serverless, Standard plan ---
    "serverless_storage_per_gb_month": 0.33,
    "serverless_read_per_million_ru": 16.0,   # a query costs 1 RU per GB of index, minimum 0.25 RU
    "serverless_write_per_million_wu": 4.0,
    "serverless_monthly_minimum": 50.0,
    "pinecone_metadata_bytes_per_vector": 500,
    "monthly_fraction_of_vectors_rewritten": 0.10,
    "write_units_per_vector": 2,           # ~1 write unit per KB
}
A = ASSUMPTIONS


def self_hosted(n):
    """Cost of running Chroma on the smallest VM whose RAM fits the HNSW index."""
    ram_needed = n * (A["embedding_dim"] * A["bytes_per_float"] + A["hnsw_link_bytes_per_vector"]) \
        * A["ram_overhead_factor"] / 1e9 + A["base_ram_gb"]
    name, ram, hourly = next((i for i in A["instances"] if i[1] >= ram_needed), A["instances"][-1])
    data_gb = n * (A["embedding_dim"] * A["bytes_per_float"] + A["hnsw_link_bytes_per_vector"]
                   + A["disk_bytes_text_and_metadata"]) / 1e9
    disk_gb = data_gb * 2 + 20  # x2 headroom for index files / WAL, +20 GB OS
    total = hourly * A["hours_per_month"] + disk_gb * A["ebs_gp3_per_gb_month"] + data_gb * A["snapshot_per_gb_month"]
    return {"instance": name, "ram_needed_gb": round(ram_needed, 1), "disk_gb": round(disk_gb, 1), "usd": round(total, 2)}


def pods(n, vectors_per_pod_768d):
    capacity = vectors_per_pod_768d * 768 / A["embedding_dim"]  # smaller vectors -> more fit per pod
    count = math.ceil(n / capacity)
    return {"pods": count, "usd": round(count * A["pod_price_per_hour"] * A["hours_per_month"], 2)}


def serverless(n, queries_per_month):
    record_bytes = A["embedding_dim"] * A["bytes_per_float"] + A["pinecone_metadata_bytes_per_vector"]
    size_gb = n * record_bytes / 1e9
    storage = size_gb * A["serverless_storage_per_gb_month"]
    reads = queries_per_month * max(0.25, size_gb) * A["serverless_read_per_million_ru"] / 1e6
    writes = n * A["monthly_fraction_of_vectors_rewritten"] * A["write_units_per_vector"] \
        * A["serverless_write_per_million_wu"] / 1e6
    usage = storage + reads + writes
    return {"index_gb": round(size_gb, 2), "usage_usd": round(usage, 2),
            "usd": round(max(usage, A["serverless_monthly_minimum"]), 2)}


def build_table():
    rows = []
    for n in (100_000, 1_000_000, 10_000_000):
        rows.append({
            "vectors": n,
            "self_hosted": self_hosted(n),
            "pods_p1": pods(n, A["p1_vectors_per_pod_768d"]),
            "pods_s1": pods(n, A["s1_vectors_per_pod_768d"]),
            "serverless_low_qps": serverless(n, A["queries_per_month_low"]),
            "serverless_high_qps": serverless(n, A["queries_per_month_high"]),
        })
    return rows


def to_markdown(rows):
    names = {100_000: "100K", 1_000_000: "1M", 10_000_000: "10M"}
    lines = [
        "| Vectors | Self-hosted Chroma | Managed pods p1 (always-on) | Managed pods s1 | Managed serverless, 100K queries/mo | Managed serverless, 5M queries/mo | Saving vs p1 pods |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s, p1 = r["self_hosted"], r["pods_p1"]
        saving = (p1["usd"] - s["usd"]) / p1["usd"] * 100
        lines.append(
            f"| {names[r['vectors']]} | ${s['usd']:,.0f} ({s['instance']}) | ${p1['usd']:,.0f} ({p1['pods']} pod) | "
            f"${r['pods_s1']['usd']:,.0f} | ${r['serverless_low_qps']['usd']:,.0f} | "
            f"${r['serverless_high_qps']['usd']:,.0f} | {saving:+.0f}% |")
    return "\n".join(lines)


def bench(sizes, n_queries=200, k=5):
    """Measure Chroma query latency on random 384-d vectors (a harder case than real text embeddings)."""
    import chromadb
    from chromadb.config import Settings
    rng = np.random.default_rng(0)
    dim = A["embedding_dim"]
    results = []
    for n in sizes:
        folder = tempfile.mkdtemp()
        client = chromadb.PersistentClient(path=folder, settings=Settings(anonymized_telemetry=False))
        col = client.create_collection("bench", metadata={"hnsw:space": "cosine"})
        start = time.perf_counter()
        for first in range(0, n, 5000):
            m = min(5000, n - first)
            vecs = rng.standard_normal((m, dim)).astype("float32")
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            col.add(ids=[str(first + i) for i in range(m)], embeddings=vecs.tolist(),
                    metadatas=[{"file_type": "md" if (first + i) % 2 == 0 else "pdf"} for i in range(m)])
        build_s = time.perf_counter() - start

        queries = rng.standard_normal((n_queries, dim)).astype("float32")
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        plain, filtered = [], []
        col.query(query_embeddings=[queries[0].tolist()], n_results=k)  # warm-up
        for q in queries:
            t = time.perf_counter()
            col.query(query_embeddings=[q.tolist()], n_results=k)
            plain.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            col.query(query_embeddings=[q.tolist()], n_results=k, where={"file_type": "md"})
            filtered.append((time.perf_counter() - t) * 1000)
        results.append({"vectors": n, "build_seconds": round(build_s, 1), "k": k,
                        "p50_ms": round(float(np.percentile(plain, 50)), 2),
                        "p95_ms": round(float(np.percentile(plain, 95)), 2),
                        "filtered_p50_ms": round(float(np.percentile(filtered, 50)), 2),
                        "filtered_p95_ms": round(float(np.percentile(filtered, 95)), 2)})
        print(f"  {n:>9,} vectors: p50={results[-1]['p50_ms']} ms  p95={results[-1]['p95_ms']} ms "
              f"(filtered p50={results[-1]['filtered_p50_ms']} ms)  build={build_s:.0f}s")
        del col, client
        shutil.rmtree(folder, ignore_errors=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="also measure latency on synthetic vectors")
    parser.add_argument("--bench-sizes", type=int, nargs="+", default=[10_000, 100_000])
    args = parser.parse_args()

    rows = build_table()
    table = to_markdown(rows)
    print(table)
    for r in rows:
        print(f"  {r['vectors']:>10,}: self-hosted needs ~{r['self_hosted']['ram_needed_gb']} GB RAM "
              f"and ~{r['self_hosted']['disk_gb']} GB disk; serverless index = {r['serverless_low_qps']['index_gb']} GB")

    result = {"assumptions": A, "rows": rows}
    if args.bench:
        print("\nLatency benchmark (random vectors, k=5, 200 queries):")
        result["bench"] = bench(args.bench_sizes)
    elif OUT_JSON.exists():
        result["bench"] = json.loads(OUT_JSON.read_text()).get("bench", [])  # keep an earlier benchmark

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))
    OUT_MD.write_text(table + "\n")
    print(f"\nSaved {OUT_JSON.name} and {OUT_MD.name} in problem1_rag/results/")


if __name__ == "__main__":
    main()
