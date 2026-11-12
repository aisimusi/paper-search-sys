"""Reserve paper-disjoint QASPER questions before agent development."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import random

from .benchmark import (checked_snapshot, digest, exclusion_reason, paragraphs,
                        read_json, tokens, write_json)


def normalized(text):
    return " ".join(tokens(text))


def paper_signature(paper):
    return digest(json.dumps([c["text"] for c in paragraphs(paper)], ensure_ascii=False).encode())


def similar_question(question, seen):
    q = normalized(question)
    return q in seen or (len(q.split()) >= 8 and any(
        len(old.split()) >= 8 and SequenceMatcher(None, q, old).ratio() >= .9 for old in seen))


def select_holdout(papers, exposed, count, seed):
    used_ids, used_titles, used_text, questions = set(), set(), set(), set()
    for gold in exposed:
        for pid, paper in gold.items():
            used_ids.add(pid)
            used_titles.add(normalized(paper.get("title", "")))
            used_text.add(paper_signature(paper))
            questions.update(normalized(q["question"]) for q in paper["qas"])
    candidates = []
    for pid, paper in sorted(papers.items()):
        if (pid in used_ids or normalized(paper.get("title", "")) in used_titles
                or paper_signature(paper) in used_text):
            continue
        chunks = paragraphs(paper)
        candidates.extend((pid, qa) for qa in paper["qas"]
                          if exclusion_reason(qa, chunks) is None)
    random.Random(seed).shuffle(candidates)
    chosen = {}
    for pid, qa in candidates:
        paper = papers[pid]
        title, signature = normalized(paper.get("title", "")), paper_signature(paper)
        if pid in used_ids or title in used_titles or signature in used_text or similar_question(qa["question"], questions):
            continue
        chosen[pid] = dict(paper, qas=[qa])
        used_ids.add(pid)
        used_titles.add(title)
        used_text.add(signature)
        questions.add(normalized(qa["question"]))
        if len(chosen) == count:
            return chosen
    raise ValueError(f"Only {len(chosen)} eligible unseen papers; requested {count}")


def inputs_from_gold(gold):
    return [{"question_id": qa["question_id"], "paper_id": pid, "title": paper["title"],
             "question": qa["question"], "chunks": paragraphs(paper)}
            for pid, paper in gold.items() for qa in paper["qas"]]


def prepare_development(args):
    root = args.study.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Development preparation requires a new empty study directory")
    if args.dev_source is None:
        raise ValueError("Development preparation requires --dev-source")
    manifest, dev, _ = checked_snapshot(args.dev_source.resolve(), "dev")
    write_json(root / "dev/gold.json", dev)
    write_json(root / "dev/inputs.json", inputs_from_gold(dev))
    reservation = {"created_at": datetime.now(timezone.utc).isoformat(),
                   "data_dir": manifest["data_dir"], "dev_source": str(args.dev_source.resolve()),
                   "hashes": {"dev": {name: digest((root / "dev" / f"{name}.json").read_bytes()) for name in ("gold", "inputs")}}}
    write_json(root / "reservation.json", reservation)
    print("Development snapshot prepared; holdout questions will be reserved after freeze.")


def reserve(args):
    from .benchmark_agent import check_frozen
    root = args.study.resolve()
    frozen = check_frozen(root)
    reservation = read_json(root / "reservation.json")
    if "holdout" in reservation["hashes"] or (root / "holdout").exists():
        raise ValueError("Holdout already reserved; use a new study")
    exposed, files = [], []
    for path in sorted(args.exclude_root.resolve().rglob("gold.json")):
        value = read_json(path)
        if isinstance(value, dict) and value and all(isinstance(p, dict) and "qas" in p and "full_text" in p for p in value.values()):
            exposed.append(value)
            files.append({"path": str(path), "sha256": digest(path.read_bytes())})
    exposed.append(read_json(root / "dev/gold.json"))
    data_dir = Path(reservation["data_dir"])
    source = data_dir / "qasper-test-v0.3.json"
    chosen = select_holdout(read_json(source), exposed, args.count, args.seed)
    write_json(root / "holdout/gold.json", chosen)
    write_json(root / "holdout/inputs.json", inputs_from_gold(chosen))
    reservation["hashes"]["holdout"] = {name: digest((root / "holdout" / f"{name}.json").read_bytes()) for name in ("gold", "inputs")}
    reservation.update({"reserved_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed, "count": args.count,
                   "source_sha256": digest(source.read_bytes()), "exposure_snapshots": files,
                   "holdout_question_ids": [q["question_id"] for p in chosen.values() for q in p["qas"]],
                   "holdout_paper_ids": list(chosen), "sampling": "official test; same text-only filter; one question per paper",
                   "exclusion": "prior paper ID, normalized title, fulltext hash; exact normalized questions; >=8-token question similarity >=0.9",
                   "limits": "Fresh to recorded local workflow, not guaranteed absent from model pretraining or undocumented exposure."})
    write_json(root / "reservation.json", reservation)
    write_json(root / "holdout-seal.json", {"reserved_at": reservation["reserved_at"],
               "freeze_sha256": digest((root / "freeze.json").read_bytes()),
               "development_reservation_sha256": frozen["reservation_sha256"],
               "reservation_sha256": digest((root / "reservation.json").read_bytes())})
    check_frozen(root)
    print(f"Reserved {len(chosen)} holdout papers after freeze; excluded {len(files)} recorded snapshots.")


def checked_inputs(root, split):
    reservation = read_json(root / "reservation.json")
    path = root / split / "inputs.json"
    if digest(path.read_bytes()) != reservation["hashes"][split]["inputs"]:
        raise ValueError("Input snapshot changed")
    return read_json(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--dev-source", type=Path)
    parser.add_argument("--prepare-dev", action="store_true")
    parser.add_argument("--exclude-root", type=Path, default=Path("reports"))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026101301)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    if args.prepare_dev:
        prepare_development(args)
    else:
        reserve(args)


if __name__ == "__main__":
    main()
