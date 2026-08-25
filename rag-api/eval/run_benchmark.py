import json
import os
import sys
import time
from datetime import datetime
from collections import defaultdict
import numpy as np

# Force UTF-8 encoding on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure app is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.query_service import answer_question


def evaluate_response(response: dict, expected_keywords: list[str]) -> str:
    """Check if answer contains relevant information or correctly refuses out-of-scope queries."""
    ans = response.get("answer", "").lower()
    
    if "error occurred while generating" in ans or "error code: 429" in ans or "rate limit" in ans:
        return "ERROR"
    if not ans:
        return "ERROR"
    
    # If out-of-scope query, check if it gracefully and factually refuses
    if any("could not find" in kw.lower() or "not found" in kw.lower() for kw in expected_keywords):
        refusal_phrases = [
            "could not find", "not found", "no direct answer", "not mentioned", 
            "no relevant", "not contain", "does not mention", "unrelated", 
            "outside the scope", "no information", "not provided"
        ]
        if any(phrase in ans for phrase in refusal_phrases):
            return "PASS"
        return "FAIL"
    
    # For financial queries, check keyword hit rate
    hits = sum(1 for kw in expected_keywords if kw.lower() in ans)
    threshold = max(1, len(expected_keywords) // 2)
    return "PASS" if hits >= threshold else "FAIL"


def run_benchmark(questions_file: str = "eval/benchmark_questions.json"):
    filepath = os.path.join(os.path.dirname(__file__), "benchmark_questions.json")
    if not os.path.exists(filepath):
        filepath = questions_file

    with open(filepath, "r", encoding="utf-8") as f:
        questions = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    raw_results_file = os.path.join(results_dir, f"benchmark_run_{timestamp}.json")

    print(f"\n=======================================================")
    print(f"[BENCHMARK] Running Financial RAG Benchmark ({len(questions)} queries)")
    print(f"=======================================================\n")

    results_by_cat = defaultdict(lambda: {"vector_correct": 0, "hybrid_correct": 0, "total": 0, "total_scorable": 0, "errors": 0})
    detailed_results = []
    vector_latencies = []
    hybrid_latencies = []

    for idx, q in enumerate(questions, 1):
        qid = q["id"]
        cat = q["category"]
        question_text = q["question"]
        expected_keywords = q["expected_answer_keywords"]

        # 1. Run Vector-Only Pass
        t0 = time.time()
        vec_res = answer_question(question=question_text, force_route="vector")
        vec_lat = round((time.time() - t0) * 1000, 2)
        vector_latencies.append(vec_lat)
        vec_passed = evaluate_response(vec_res, expected_keywords)

        # 2. Run Hybrid GraphRAG Pass (Dynamic Routing + Graph Traversal + Vector Context)
        t1 = time.time()
        hyb_res = answer_question(question=question_text)
        hyb_lat = round((time.time() - t1) * 1000, 2)
        hybrid_latencies.append(hyb_lat)
        hyb_passed = evaluate_response(hyb_res, expected_keywords)

        results_by_cat[cat]["total"] += 1
        
        if vec_passed == "ERROR" or hyb_passed == "ERROR":
            results_by_cat[cat]["errors"] += 1
            v_mark = "ERROR" if vec_passed == "ERROR" else "FAIL"
            h_mark = "ERROR" if hyb_passed == "ERROR" else "FAIL"
        else:
            results_by_cat[cat]["total_scorable"] += 1
            if vec_passed == "PASS":
                results_by_cat[cat]["vector_correct"] += 1
            if hyb_passed == "PASS":
                results_by_cat[cat]["hybrid_correct"] += 1
            v_mark = vec_passed
            h_mark = hyb_passed

        print(f"[{cat.upper():<12}] Q{qid:<2}: {question_text[:45]:<45} | Vec: [{v_mark}] ({vec_lat}ms) | Hyb ({hyb_res.get('route')}): [{h_mark}] ({hyb_lat}ms)", flush=True)

        # Sleep to be polite to the API
        time.sleep(1)

        detailed_results.append({
            "id": qid,
            "category": cat,
            "question": question_text,
            "expected_keywords": expected_keywords,
            "vector_eval": {
                "passed": vec_passed,
                "latency_ms": vec_lat,
                "answer_snippet": vec_res.get("answer", "")[:200],
                "sources_count": len(vec_res.get("sources", []))
            },
            "hybrid_eval": {
                "passed": hyb_passed,
                "route_taken": hyb_res.get("route", ""),
                "latency_ms": hyb_lat,
                "answer_snippet": hyb_res.get("answer", "")[:200],
                "sources_count": len(hyb_res.get("sources", []))
            }
        })

    # Summary Calculations
    summary = {}
    category_order = ["single-hop", "two-hop", "three-hop", "aggregation", "out-of-scope"]
    
    total_vec_correct = sum(d["vector_correct"] for d in results_by_cat.values())
    total_hyb_correct = sum(d["hybrid_correct"] for d in results_by_cat.values())
    total_queries = len(questions)
    total_scorable = sum(d["total_scorable"] for d in results_by_cat.values())
    total_errors = sum(d["errors"] for d in results_by_cat.values())

    for cat in category_order:
        if cat in results_by_cat:
            d = results_by_cat[cat]
            summary[cat] = {
                "total": d["total"],
                "total_scorable": d["total_scorable"],
                "errors": d["errors"],
                "vector_correct": d["vector_correct"],
                "vector_accuracy": round((d["vector_correct"] / d["total_scorable"]) * 100, 2) if d["total_scorable"] else 0.0,
                "hybrid_correct": d["hybrid_correct"],
                "hybrid_accuracy": round((d["hybrid_correct"] / d["total_scorable"]) * 100, 2) if d["total_scorable"] else 0.0,
                "delta": round(((d["hybrid_correct"] - d["vector_correct"]) / d["total_scorable"]) * 100, 2) if d["total_scorable"] else 0.0
            }

    latency_stats = {
        "vector": {
            "p50_ms": round(float(np.percentile(vector_latencies, 50)), 2),
            "p95_ms": round(float(np.percentile(vector_latencies, 95)), 2),
            "mean_ms": round(float(np.mean(vector_latencies)), 2)
        },
        "hybrid": {
            "p50_ms": round(float(np.percentile(hybrid_latencies, 50)), 2),
            "p95_ms": round(float(np.percentile(hybrid_latencies, 95)), 2),
            "mean_ms": round(float(np.mean(hybrid_latencies)), 2)
        }
    }

    full_output = {
        "metadata": {
            "timestamp": timestamp,
            "total_queries": total_queries,
            "overall_vector_accuracy": round((total_vec_correct / total_queries) * 100, 2),
            "overall_hybrid_accuracy": round((total_hyb_correct / total_queries) * 100, 2),
            "llm_provider": os.getenv("LLM_PROVIDER", "groq"),
            "llm_model": os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
            "vector_store": "FAISS (all-MiniLM-L6-v2)",
            "graph_store": "Neo4j 5.x"
        },
        "summary_by_category": summary,
        "latency_stats": latency_stats,
        "detailed_results": detailed_results
    }

    # Save to disk
    with open(raw_results_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print("\n" + "=" * 78)
    print("BENCHMARK RESULTS: Vector-Only Baseline vs. Hybrid GraphRAG (50 Questions)")
    print("=" * 78)
    print(f"{'Category':<16} | {'Vector (Correct/Total)':<24} | {'Hybrid (Correct/Total)':<24} | {'Delta':<8}")
    print("-" * 78)

    for cat in category_order:
        if cat in summary:
            s = summary[cat]
            vec_str = f"{s['vector_correct']}/{s['total_scorable']} ({s['vector_accuracy']}%)"
            hyb_str = f"{s['hybrid_correct']}/{s['total_scorable']} ({s['hybrid_accuracy']}%)"
            delta_str = f"+{s['delta']}%" if s['delta'] >= 0 else f"{s['delta']}%"
            print(f"{cat.capitalize():<16} | {vec_str:<24} | {hyb_str:<24} | {delta_str:<8}")

    p50_v = latency_stats['vector']['p50_ms']
    p50_h = latency_stats['hybrid']['p50_ms']
    p50_diff = round(p50_h - p50_v, 1)
    p50_diff_str = f"+{p50_diff} ms" if p50_diff >= 0 else f"{p50_diff} ms"

    p95_v = latency_stats['vector']['p95_ms']
    p95_h = latency_stats['hybrid']['p95_ms']
    p95_diff = round(p95_h - p95_v, 1)
    p95_diff_str = f"+{p95_diff} ms" if p95_diff >= 0 else f"{p95_diff} ms"

    tot_vec_pct = full_output["metadata"]["overall_vector_accuracy"]
    tot_hyb_pct = full_output["metadata"]["overall_hybrid_accuracy"]
    tot_delta = round(tot_hyb_pct - tot_vec_pct, 2)
    tot_delta_str = f"+{tot_delta}%" if tot_delta >= 0 else f"{tot_delta}%"

    print("-" * 78)
    print(f"{'Overall Acc.':<16} | {total_vec_correct}/{total_scorable} ({tot_vec_pct}%) | {total_hyb_correct}/{total_scorable} ({tot_hyb_pct}%) | {tot_delta_str:<8}")
    print(f"{'Latency (p50)':<16} | {p50_v:>18.1f} ms | {p50_h:>18.1f} ms | {p50_diff_str:<8}")
    print(f"{'Latency (p95)':<16} | {p95_v:>18.1f} ms | {p95_h:>18.1f} ms | {p95_diff_str:<8}")
    print("=" * 78)
    if total_errors > 0:
        print(f"\n[WARNING] {total_errors}/{total_queries} questions could not be scored due to rate limiting or infrastructure errors during this run.")
    print(f"\n[OUTPUT] Raw benchmark results written to:\n  {raw_results_file}\n")


if __name__ == "__main__":
    run_benchmark()
