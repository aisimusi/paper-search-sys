"""Additional QASPER metrics, explicit denominators and human support annotations."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import re
import string

from .benchmark import digest, read_json, write_json


def mean(values):
    return sum(values) / len(values) if values else None


def canonical_usage(usage):
    names = {"inputTokens": "input_tokens", "outputTokens": "output_tokens",
             "cacheReadTokens": "cache_read_tokens", "reasoningTokens": "reasoning_tokens",
             "totalTokens": "total_tokens"}
    return [{names.get(k, k): v for k, v in event.items()} for event in usage]


def normalize_answer(answer):
    """QASPER/SQuAD normalization, also used by the additional Exact Match metric."""
    answer = "".join(c for c in answer.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", answer).split())


def evidence_scores(predicted, reference):
    predicted, reference = set(predicted), set(reference)
    if not predicted and not reference:
        return 1.0, 1.0, 1.0
    common = len(predicted & reference)
    precision = common / len(predicted) if predicted else 0.0
    recall = common / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if common else 0.0
    return precision, recall, f1


def prediction_scores(prediction, references):
    if prediction is None:
        return {"exact_match": 0.0, "evidence_precision": 0.0, "evidence_recall": 0.0,
                "evidence_f1": 0.0, "task_success": False}
    matches = [normalize_answer(prediction["answer"]) == normalize_answer(r["answer"]) for r in references]
    # Choose one reference by evidence F1, then recall, then precision, then annotation order.
    scores = [evidence_scores(prediction["evidence"], r["evidence"]) for r in references]
    precision, recall, f1 = max(scores, key=lambda s: (s[2], s[1], s[0]))
    return {"exact_match": float(any(matches)), "evidence_precision": precision,
            "evidence_recall": recall, "evidence_f1": f1,
            "task_success": any(match and set(prediction["evidence"]) == set(r["evidence"])
                                for match, r in zip(matches, references))}


def ranked_scores(context, references, k):
    # Repeated text is one evidence paragraph, as in set-based evidence matching.
    ranked = list(dict.fromkeys(c["text"] for c in context))[:k]
    evidence_sets = [set(r["evidence"]) for r in references if r["evidence"]]
    if not evidence_sets:
        return None
    scores = []
    for evidence in evidence_sets:
        hits = [i for i, text in enumerate(ranked, 1) if text in evidence]
        dcg = sum(1 / math.log2(i + 1) for i in hits)
        ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(evidence)) + 1))
        scores.append({"recall": len(hits) / len(evidence), "mrr": 1 / hits[0] if hits else 0.0,
                       "ndcg": dcg / ideal, "complete_reference": float(len(hits) == len(evidence))})
    # Each ranking metric takes its best reference, then macro-averages over eligible questions.
    return {name: max(s[name] for s in scores) for name in scores[0]}


def quality_metrics(gold, states, rows, ks, has_retrieval=True):
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("retrieval_ks must contain positive integers")
    row_by_id = {row["question_id"]: row for row in rows}
    rankings = {stage: {str(k): [] for k in ks} for stage in ("candidate", "initial", "final")}
    initially_failed, recovered = 0, 0
    per_question = []
    for qid, references in gold.items():
        state = states.get(qid)
        result = prediction_scores(state["prediction"] if state else None, references)
        initial = prediction_scores(state.get("initial_proposal") if state else None, references)
        # Operational failures and missing initial proposals are not answer-revision failures.
        eligible = state is not None and "initial_proposal" in state and not initial["task_success"]
        initially_failed += eligible
        recovered += eligible and result["task_success"]
        row_by_id[qid].update(result, initial_task_success=initial["task_success"] if state and "initial_proposal" in state else None)
        per_question.append(result)
        for stage, key in (("candidate", "candidate_context"), ("initial", "initial_context"), ("final", "context")):
            context = state.get(key) if state else []
            # Historical states do not preserve earlier ranked contexts.
            if context is None:
                continue
            for k in ks:
                score = ranked_scores(context, references, k)
                if score is not None:
                    rankings[stage][str(k)].append(score)
    ranking_summary = {}
    for stage, by_k in rankings.items():
        ranking_summary[stage] = {k: {"questions": len(values), **{
            name: mean([v[name] for v in values]) if has_retrieval else None
            for name in ("recall", "mrr", "ndcg", "complete_reference")}} for k, values in by_k.items()}
    return {"answer_exact_match": mean([r["exact_match"] for r in per_question]),
            "evidence_precision": mean([r["evidence_precision"] for r in per_question]),
            "evidence_recall": mean([r["evidence_recall"] for r in per_question]),
            "task_successes": sum(r["task_success"] for r in per_question),
            "task_success_rate": mean([r["task_success"] for r in per_question]),
            "initial_task_failures": initially_failed, "recovered_tasks": recovered,
            "failure_recovery_rate": recovered / initially_failed if initially_failed else None,
            "retrieval": ranking_summary, "retrieval_applicable": has_retrieval}


def human_support(root, inputs, states):
    packets = []
    for item in inputs:
        state = states.get(item["question_id"])
        if not state or state["prediction"]["answer"] == "Unanswerable":
            continue
        prediction = state["prediction"]
        packet = {"question_id": item["question_id"], "title": item["title"], "question": item["question"],
                  "answer": prediction["answer"], "evidence_ids": prediction["evidence_ids"],
                  "cited_paragraphs": [c for c in state["context"] if c["id"] in prediction["evidence_ids"]]}
        packet["packet_sha256"] = digest(json.dumps(packet, ensure_ascii=False, sort_keys=True).encode())
        packets.append(packet)
    # The editable file binds annotations to exact question/answer/citation packets.
    path = root / "human-support.json"
    document = read_json(path) if path.exists() else {"rubric": "Judge whether ALL answer facts and required qualifiers are supported by the cited paper paragraphs. Use supported, unsupported, or uncertain; leave unreviewed entries pending. Record reviewer and reason.", "annotations": []}
    annotations = document["annotations"]
    expected = {p["question_id"]: p["packet_sha256"] for p in packets}
    ids = [a["question_id"] for a in annotations]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate human annotation question IDs")
    if any(a["question_id"] not in expected or a["packet_sha256"] != expected[a["question_id"]]
           for a in annotations if a["verdict"] != "pending"):
        raise ValueError("Human annotation refers to a different answer or citation packet")
    by_id = {a["question_id"]: a for a in annotations}
    annotations = [by_id[qid] if qid in by_id and by_id[qid]["packet_sha256"] == sha else
                   {"question_id": qid, "packet_sha256": sha, "verdict": "pending", "reviewer": "", "reason": ""}
                   for qid, sha in expected.items()]
    for a in annotations:
        if a["packet_sha256"] != expected[a["question_id"]]:
            raise ValueError("Human annotation refers to a different answer or citation packet")
        if a["verdict"] not in ("pending", "supported", "unsupported", "uncertain"):
            raise ValueError("Unknown human support verdict")
        if a["verdict"] != "pending" and (not a["reviewer"].strip() or not a["reason"].strip()):
            raise ValueError("Human support judgments require reviewer and reason")
    document["annotations"] = annotations
    if not path.exists() or read_json(path) != document:
        write_json(path, document)
    write_json(root / "human-support-packets.json", packets)
    reviewed = [a for a in annotations if a["verdict"] != "pending"]
    supported = sum(a["verdict"] == "supported" for a in reviewed)
    return {"eligible_positive_answers": len(packets), "reviewed": len(reviewed), "supported": supported,
            "uncertain": sum(a["verdict"] == "uncertain" for a in reviewed),
            "coverage": len(reviewed) / len(packets) if packets else None,
            "support_rate": supported / len(reviewed) if reviewed else None,
            "complete": bool(packets) and len(reviewed) == len(packets)}


def percentile95(values):
    """Nearest rank percentile: the ceil(0.95*n)-th observation."""
    return sorted(values)[math.ceil(.95 * len(values)) - 1] if values else None


def priced_cost(records, pricing, backend):
    if pricing is None:
        return None
    currency = pricing.get("currency")
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("pricing.currency is required")
    for key in ("input_per_million", "output_per_million", "cached_input_per_million"):
        value = pricing.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"pricing.{key} must be a nonnegative finite number")
    cost = 0.0
    for record in records:
        if not record["usage"]:
            return None
        for usage in record["usage"]:
            if "input_tokens" not in usage or "output_tokens" not in usage:
                return None
            cached = usage.get("cached_input_tokens", usage.get("cache_read_tokens", 0))
            if any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0
                   for n in (usage["input_tokens"], usage["output_tokens"], cached)):
                raise ValueError("Token counts must be nonnegative finite numbers")
            input_count = usage["input_tokens"] - cached if backend == "codex" else usage["input_tokens"]
            if input_count < 0:
                raise ValueError("Cached tokens exceed total input tokens")
            cost += (input_count * pricing["input_per_million"] + cached * pricing["cached_input_per_million"]
                     + usage["output_tokens"] * pricing["output_per_million"]) / 1_000_000
    return {"amount": cost, "currency": currency, "pricing": pricing}


def performance_metrics(root, batches, question_count, policy):
    paths = sorted((root / "calls").glob("**/record.json"))
    records = [read_json(p) for p in paths]
    online = [r for r in records if r["stage"] != "final-audit"]
    evaluation = [r for r in records if r["stage"] == "final-audit"]
    actual = [r for r in records if "reused_from" not in r]
    grouped = defaultdict(list)
    for path, record in zip(paths, records):
        grouped[path.parent.parent].append(record)
    failed_groups = [values for values in grouped.values() if any(v["status"] != "ok" for v in values)]
    recovered = sum(any(v["status"] == "ok" for v in values) for values in failed_groups)
    # All questions in a synchronous batch complete when that batch finishes.
    latencies = [b["inference_wall_seconds"] for b in batches if "inference_wall_seconds" in b for _ in b["question_ids"]]
    all_latencies = [b["wall_seconds"] for b in batches for _ in b["question_ids"]]
    usage_complete = all(r["usage"] and all("input_tokens" in u and "output_tokens" in u for u in r["usage"]) for r in records)
    pricing = policy.get("pricing")
    backend = policy.get("model", {}).get("backend")
    cost = priced_cost(online, pricing, backend)
    return {"total_wall_seconds": sum(v["wall_seconds"] for v in
                (read_json(p) for p in sorted((root / "executions").glob("*.json")))) or None,
            "batch_wall_seconds_total": sum(b["wall_seconds"] for b in batches),
            "batch_inference_seconds_total": sum(b["inference_wall_seconds"] for b in batches if "inference_wall_seconds" in b),
            "question_latency_mean_seconds": mean(latencies), "question_latency_p95_seconds": percentile95(latencies),
            "evaluation_inclusive_question_latency_mean_seconds": mean(all_latencies),
            "evaluation_inclusive_question_latency_p95_seconds": percentile95(all_latencies),
            "latency_questions": len(latencies), "latency_includes_final_audit": False,
            "latency_measure": "batch inference completion wall time assigned to every question; includes retrieval and retries; excludes final diagnostic audit; resumed cached stages may shorten wall time",
            "online_model_seconds": sum(r["wall_seconds"] for r in online),
            "evaluation_model_seconds": sum(r["wall_seconds"] for r in evaluation),
            "mean_model_attempts": len(records) / question_count if question_count else None,
            "mean_online_model_attempts": len(online) / question_count if question_count else None,
            "failed_call_groups": len(failed_groups), "recovered_call_groups": recovered,
            "call_failure_recovery_rate": recovered / len(failed_groups) if failed_groups else None,
            "usage_complete": usage_complete,
            "online_cost": cost, "evaluation_cost": priced_cost(evaluation, pricing, backend),
            "actual_cost": priced_cost(actual, pricing, backend),
            "mean_task_cost": cost["amount"] / question_count if cost is not None and question_count else None,
            "cost_currency": pricing.get("currency") if pricing else None}
