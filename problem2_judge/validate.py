"""Measure the judge's biases and validate it. Every claim in the README comes from these numbers.

    python -m problem2_judge.validate position     # position bias: flip rate when A/B order is swapped
    python -m problem2_judge.validate bias         # verbosity + confidently-wrong probe (naive vs hardened judge)
    python -m problem2_judge.validate agreement    # agreement/kappa with human labels + score clustering
    python -m problem2_judge.validate retest       # test-retest consistency
    python -m problem2_judge.validate self         # self-enhancement (same-family vs different-family judge)
    python -m problem2_judge.validate all          # everything above

Add --n 4 (or similar) to use fewer items and save API calls. Results go to results/validation_report.json
"""
import argparse
import json
import math
import statistics
from pathlib import Path

from problem2_judge.judge import PASS_THRESHOLD, Judge, generate, load_suite
from llm import LLM

HERE = Path(__file__).parent
REPORT = HERE / "results" / "validation_report.json"

# Filler that adds NO information. Appended to a correct answer to make a "padded" version.
FILLER = (" It is worth noting that this is a topic of considerable interest, and many people find it fascinating to explore "
          "in more depth. There are numerous perspectives one could take, and a great deal has been written about it over the "
          "years. In summary, this is a well-established and widely discussed matter that continues to generate interest.")


# ------------------------------------------------------------------ statistics (no external libraries)
def cohen_kappa(a, b):
    """Agreement between two binary label lists, corrected for chance. 1 = perfect, 0 = chance."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return None if expected == 1 else round((observed - expected) / (1 - expected), 3)


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, i = [0.0] * len(values), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return None if den == 0 else round(num / den, 3)


def mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def save(key, value, judge=None):
    REPORT.parent.mkdir(exist_ok=True)
    data = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    data[key] = value
    if judge:
        data[key]["judge_usage"] = judge.usage()
    REPORT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(data[key], indent=1, ensure_ascii=False))
    print(f"-> saved '{key}' in problem2_judge/results/validation_report.json\n")


def load_probes(n):
    items = json.loads((HERE / "data" / "probes.json").read_text(encoding="utf-8"))["items"][:n]
    for item in items:
        item["padded"] = item["correct"] + FILLER
    return items


# ------------------------------------------------------------------ 1. position bias
def cmd_position(args):
    judge = Judge()
    # (name, answer shown first, answer shown second, outcomes that count as a judge ERROR)
    pair_types = [
        ("identical_answers", "correct", "correct", {"1", "2"}),        # should be a tie
        ("correct_vs_wrong", "correct", "wrong_confident", {"2"}),      # the wrong answer must not win
        ("correct_vs_padded", "correct", "padded", {"2"}),              # padding must not win
        ("correct_vs_terse", "correct", "terse_correct", set()),        # both right: only look at flips
    ]
    stats = {name: {"n": 0, "flips": 0, "hard_flips": 0, "single_order_errors": 0, "single_order_calls": 0,
                    "final_errors": 0, "final_ties": 0} for name, *_ in pair_types}
    letters = {"A": 0, "B": 0, "tie": 0}
    for item in load_probes(args.n):
        for name, first, second, bad in pair_types:
            res = judge.pairwise_both_orders(item, item[first], item[second], item["id"])
            s = stats[name]
            s["n"] += 1
            s["flips"] += res["flipped"]
            s["hard_flips"] += res["hard_flip"]
            for single in (res["order1"], res["order2"]):         # BEFORE the fix: each order on its own
                s["single_order_calls"] += 1
                s["single_order_errors"] += single in bad
            s["final_errors"] += res["final"] in bad              # AFTER the fix: both orders must agree
            s["final_ties"] += res["final"] == "tie"
            for raw in (res["order1_raw"], res["order2_raw"]):
                if raw:
                    letters[raw] += 1
    by_type = {}
    for name, s in stats.items():
        by_type[name] = {"n_pairs": s["n"], "flip_rate": round(s["flips"] / s["n"], 3), "hard_flip_rate": round(s["hard_flips"] / s["n"], 3),
                         "error_rate_single_order_BEFORE": round(s["single_order_errors"] / s["single_order_calls"], 3),
                         "error_rate_both_orders_AFTER": round(s["final_errors"] / s["n"], 3), "final_tie_rate": round(s["final_ties"] / s["n"], 3)}
    total_pairs = sum(s["n"] for s in stats.values())
    decisive = letters["A"] + letters["B"]
    save("position_bias", {
        "overall_flip_rate": round(sum(s["flips"] for s in stats.values()) / total_pairs, 3),
        "share_of_decisive_verdicts_that_picked_first_position": round(letters["A"] / decisive, 3) if decisive else None,
        "note": "flip = the verdict changed when A/B order was swapped. Without bias the first-position share is ~0.5 on equal-quality pairs.",
        "by_pair_type": by_type, "raw_letter_counts": letters}, judge)


# ------------------------------------------------------------------ 2. verbosity + confidently-wrong probe
def cmd_bias(args):
    items = load_probes(args.n)
    out = {}
    judges = {}
    for label, hardened in (("naive_judge_BEFORE", False), ("hardened_judge_AFTER", True)):
        judge = Judge(hardened=hardened)
        judges[label] = judge
        kinds = ["correct", "padded", "wrong_confident", "terse_correct"]
        scores = {k: [] for k in kinds}
        verdicts = {k: [] for k in kinds}
        for item in items:
            for kind in kinds:
                v = judge.score(item, item[kind], f"{item['id']}_{kind}")
                scores[kind].append(v["overall"])
                verdicts[kind].append(v["verdict"])
        pairs = [(c, p) for c, p in zip(scores["correct"], scores["padded"]) if c is not None and p is not None]
        out[label] = {
            "mean_overall_correct": mean(scores["correct"]),
            "mean_overall_padded": mean(scores["padded"]),
            "verbosity_gain (padded - correct, ideal <= 0)": round(statistics.mean(p - c for c, p in pairs), 3) if pairs else None,
            "padded_scored_higher_rate": round(sum(p > c for c, p in pairs) / len(pairs), 3) if pairs else None,
            "mean_overall_wrong_confident (ideal low)": mean(scores["wrong_confident"]),
            "wrong_confident_FOOLED_rate (judge said PASS)": round(verdicts["wrong_confident"].count("PASS") / len(items), 3),
            "mean_overall_terse_correct (ideal high)": mean(scores["terse_correct"]),
            "terse_correct_wrongly_failed_rate": round(verdicts["terse_correct"].count("FAIL") / len(items), 3),
        }
        out[label]["judge_usage"] = judge.usage()
    out["n_items"] = len(items)
    save("verbosity_and_sycophancy_probe", out)


# ------------------------------------------------------------------ 3. agreement with human labels + clustering
def cmd_agreement(args):
    cases = [c for c in load_suite(args.suite) if "human_score" in c][: args.n]
    out = {}
    for label, hardened in (("naive_judge_BEFORE", False), ("hardened_judge_AFTER", True)):
        judge = Judge(hardened=hardened)
        verdicts = [judge.score(c, c["model_output"]) for c in cases]
        pairs = [(c, v) for c, v in zip(cases, verdicts) if v["parse_ok"]]
        human_pass = [int(c["human_score"] >= PASS_THRESHOLD) for c, _ in pairs]
        judge_pass = [int(v["verdict"] == "PASS") for _, v in pairs]
        human_scores = [c["human_score"] for c, _ in pairs]
        judge_scores = [v["overall"] for _, v in pairs]
        correctness = [v["scores"]["correctness"]["score"] for _, v in pairs]
        out[label] = {
            "n_cases": len(pairs),
            "pass_fail_agreement": round(sum(h == j for h, j in zip(human_pass, judge_pass)) / len(pairs), 3),
            "cohens_kappa": cohen_kappa(human_pass, judge_pass),
            "spearman_correlation": spearman(human_scores, judge_scores),
            "mean_abs_error_vs_human": round(statistics.mean(abs(h - j) for h, j in zip(human_scores, judge_scores)), 3),
            "judge_false_pass": sum(1 for h, j in zip(human_pass, judge_pass) if h == 0 and j == 1),
            "judge_false_fail": sum(1 for h, j in zip(human_pass, judge_pass) if h == 1 and j == 0),
            # score clustering: does the judge use the whole 1-5 range or hide in 4-5?
            "correctness_score_histogram": {str(s): correctness.count(s) for s in range(1, 6)},
            "overall_score_stdev": round(statistics.pstdev(judge_scores), 3),
            "share_of_overall_scores_4_or_more": round(sum(s >= 4 for s in judge_scores) / len(judge_scores), 3),
            "human_share_4_or_more": round(sum(s >= 4 for s in human_scores) / len(human_scores), 3),
            "judge_usage": judge.usage()}
    save("agreement_with_human_labels", out)


# ------------------------------------------------------------------ 4. test-retest
def cmd_retest(args):
    cases = [c for c in load_suite(args.suite) if "human_score" in c][: args.n]
    judge = Judge()
    judge.temperature = args.temp
    runs = [[judge.score(c, c["model_output"]) for c in cases] for _ in range(args.runs)]
    flips, ranges = 0, []
    for i in range(len(cases)):
        verdicts = {run[i]["verdict"] for run in runs}
        overall = [run[i]["overall"] for run in runs if run[i]["overall"] is not None]
        flips += len(verdicts) > 1
        if overall:
            ranges.append(max(overall) - min(overall))
    save("test_retest", {"n_cases": len(cases), "runs": args.runs, "judge_temperature": args.temp if args.temp is not None else "from .env",
                         "pass_fail_flip_rate": round(flips / len(cases), 3), "mean_score_range_across_runs": mean(ranges),
                         "identical_overall_scores_rate": round(sum(1 for r in ranges if r == 0) / len(ranges), 3) if ranges else None}, judge)


# ------------------------------------------------------------------ 5. self-enhancement
def cmd_self(args):
    cases = [c for c in load_suite(args.suite) if c.get("expected_output")][: args.n]
    generator = LLM("GEN")
    cross = Judge("JUDGE")                                       # different family from the generator
    same = Judge("SELFJUDGE", fallback_prefix="GEN")             # same model as the generator
    own_cross, own_same, neutral_cross, neutral_same = [], [], [], []
    for c in cases:
        own_answer, _ = generate(generator, c, {})
        own_cross.append(cross.score(c, own_answer)["overall"])
        own_same.append(same.score(c, own_answer)["overall"])
        neutral_cross.append(cross.score(c, c["model_output"])["overall"])   # hand-written answers: neither judge "owns" them
        neutral_same.append(same.score(c, c["model_output"])["overall"])
    gap_own = (mean(own_same) or 0) - (mean(own_cross) or 0)
    gap_neutral = (mean(neutral_same) or 0) - (mean(neutral_cross) or 0)
    save("self_enhancement", {
        "n_cases": len(cases), "generator": f"{generator.provider}/{generator.model}",
        "same_family_judge": f"{same.llm.provider}/{same.llm.model}", "different_family_judge": f"{cross.llm.provider}/{cross.llm.model}",
        "mean_score_given_to_generator_own_answers": {"same_family_judge": mean(own_same), "different_family_judge": mean(own_cross)},
        "mean_score_given_to_neutral_hand_written_answers": {"same_family_judge": mean(neutral_same), "different_family_judge": mean(neutral_cross)},
        "gap_on_own_answers": round(gap_own, 3), "gap_on_neutral_answers": round(gap_neutral, 3),
        "self_enhancement_estimate (gap_own - gap_neutral; > 0 means the same-family judge favours its own family)": round(gap_own - gap_neutral, 3),
        "caveat": "Small sample; indicative only."})


def main():
    parser = argparse.ArgumentParser(description="Judge bias measurements and validation")
    parser.add_argument("command", choices=["position", "bias", "agreement", "retest", "self", "all"])
    parser.add_argument("--suite", default=str(HERE / "data" / "suite.json"))
    parser.add_argument("--n", type=int, default=None, help="use only the first N items (saves API calls)")
    parser.add_argument("--runs", type=int, default=3, help="retest: number of repeated runs")
    parser.add_argument("--temp", type=float, default=None, help="retest: judge temperature (default: from .env)")
    args = parser.parse_args()

    commands = {"position": cmd_position, "bias": cmd_bias, "agreement": cmd_agreement, "retest": cmd_retest, "self": cmd_self}
    if args.command == "all":
        defaults = {"position": 6, "bias": 6, "agreement": None, "retest": 10, "self": 8}
        user_n = args.n
        for name, func in commands.items():
            args.n = user_n or defaults[name]
            func(args)
    else:
        commands[args.command](args)


if __name__ == "__main__":
    main()
