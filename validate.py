# -*- coding: utf-8 -*-
"""Mechanical validation of a model answer against the numbered sources.

Scope: this checker ONLY verifies JSON well-formedness, the decision ids, the
required fields, and that every cited line range exists inside the source it
names. A "ranges_valid" result says nothing about whether the cited lines
actually entail the claim. Semantic checking is a separate, manual step
(see audit.md).
"""
from pathlib import Path
import json, sys

ALLOWED_IDS = {'P3', 'P5'}
REQUIRED = ("id", "answer", "evidence", "limitations")


def validate(sources_path, output_path):
    sources = {s["id"]: s["text"].split("\n")
               for s in json.loads(Path(sources_path).read_text(encoding="utf8"))}
    raw = Path(output_path).read_text(encoding="utf8")
    errors = []
    try:
        payload = json.loads(raw)
    except Exception as exc:
        return {"status": "invalid_json", "errors": [f"json_parse_error: {exc}"],
                "semantic": "not_evaluated", "excerpts": []}

    rows = payload.get("decisions")
    if not isinstance(rows, list):
        return {"status": "invalid_schema", "errors": ["decisions_not_a_list"],
                "semantic": "not_evaluated", "excerpts": []}

    ids = [r.get("id") for r in rows if isinstance(r, dict)]
    if ids != ["D1", "D2", "D3", "D4"]:
        errors.append("decision_ids:" + ",".join(str(i) for i in ids))

    excerpts = []
    for row in rows:
        if not isinstance(row, dict):
            errors.append("row_not_object")
            continue
        rid = row.get("id", "?")
        for field in REQUIRED:
            if field not in row or row[field] in (None, "", [], {}):
                errors.append(f"{rid}:missing_field:{field}")
        for ev in row.get("evidence", []) or []:
            if not isinstance(ev, dict):
                errors.append(f"{rid}:evidence_not_object")
                continue
            sid = ev.get("source_id")
            a, b = ev.get("start_line"), ev.get("end_line")
            if sid not in ALLOWED_IDS:
                errors.append(f"{rid}:unknown_source_id:{sid}")
                continue
            if sid not in sources:
                errors.append(f"{rid}:source_not_found:{sid}")
                continue
            lines = sources[sid]
            if type(a) is not int or type(b) is not int:
                errors.append(f"{rid}:non_integer_line")
                continue
            if not (1 <= a <= b <= len(lines)):
                errors.append(f"{rid}:invalid_range:{a}-{b}")
                continue
            excerpts.append({"decision": rid, "source_id": sid,
                             "start_line": a, "end_line": b,
                             "text": "\n".join(lines[a - 1:b])})

    return {"status": "range_errors" if errors else "ranges_valid",
            "errors": errors,
            "semantic": "not_evaluated",
            "excerpt_count": len(excerpts),
            "excerpts": excerpts}


if __name__ == "__main__":
    result = validate(sys.argv[1], sys.argv[2])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result["status"] == "ranges_valid" else 1)
