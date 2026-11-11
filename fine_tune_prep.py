#!/usr/bin/env python3
"""fine-tune-prep — get a chat dataset ready for LLM fine-tuning.

Validate, inspect, split and convert JSONL chat datasets (one
{"messages": [...]} per line, OpenAI-style) before you spend money training
on them. Zero dependencies on purpose: the boring part should run locally,
for free, in under a second.
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter

ROLES = ("system", "user", "assistant")
TOKENS_PER_CHAR = 0.25  # ~4 chars per token: a heuristic, not a tokenizer


def example_errors(obj):
    """Every problem with one parsed example, or an empty list."""
    if not isinstance(obj, dict):
        return ["example is not a JSON object"]
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ["missing a non-empty 'messages' list"]
    errs = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            errs.append(f"message {i}: not an object")
            continue
        role = m.get("role")
        if role not in ROLES:
            errs.append(f"message {i}: role {role!r} not in {ROLES}")
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            errs.append(f"message {i}: empty or non-text 'content'")
    return errs


def read_jsonl(path):
    """[(lineno, obj_or_None, parse_error_or_None)] for every non-blank line."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append((lineno, json.loads(line), None))
            except json.JSONDecodeError as e:
                entries.append((lineno, None, f"invalid JSON: {e.msg}"))
    return entries


def cmd_validate(args):
    entries = read_jsonl(args.path)
    bad_lines = 0
    for lineno, obj, parse_err in entries:
        errs = [parse_err] if parse_err else example_errors(obj)
        for e in errs:
            print(f"line {lineno}: {e}")
        bad_lines += bool(errs)
    ok = len(entries) - bad_lines
    print(f"\n{ok}/{len(entries)} examples pass ({bad_lines} bad)")
    return 1 if bad_lines else 0


def cmd_stats(args):
    entries = [(n, o, e) for n, o, e in read_jsonl(args.path) if o is not None and not example_errors(o)]
    roles = Counter()
    chars = 0
    user_texts = Counter()
    for _, obj, _ in entries:
        for m in obj["messages"]:
            roles[m["role"]] += 1
            chars += len(m["content"])
            if m["role"] == "user":
                user_texts[m["content"]] += 1
    est_tokens = int(chars * TOKENS_PER_CHAR)
    print(f"valid examples      {len(entries)}")
    for role in ROLES:
        print(f"{role + ' messages':<19} {roles[role]}")
    print(f"characters          {chars:,}")
    print(f"est. tokens (~c/4)  {est_tokens:,}")
    dups = sum(c - 1 for c in user_texts.values() if c > 1)
    print(f"duplicate prompts   {dups}")
    return 0


def cmd_split(args):
    lines = [l.strip() for l in open(args.source, encoding="utf-8") if l.strip()]
    shuffled = list(lines)
    random.Random(args.seed).shuffle(shuffled)
    cut = round(len(shuffled) * args.ratio)
    for path, part in ((args.train, shuffled[:cut]), (args.val, shuffled[cut:])):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(part) + "\n")
        print(f"{path}: {len(part)} examples")


def cmd_convert(args):
    with open(args.source, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if rows and not rows[0][0].strip().lower() in ("q", "user", "prompt"):
        header = rows[0]
    else:
        header = None
        rows = rows[1:] or rows  # tolerate a missing header
    written = 0
    with open(args.dest, "w", encoding="utf-8") as f:
        for row in rows:
            if not any(cell.strip() for cell in row):
                continue
            q = row[0].strip()
            a = row[1].strip() if len(row) > 1 else ""
            system = row[2].strip() if len(row) > 2 and row[2].strip() else None
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": a})
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            written += 1
    print(f"{args.dest}: {written} examples written")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="prep chat JSONL datasets for fine-tuning")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="report every problem, line by line")
    v.add_argument("path")
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("stats", help="role counts, size, estimated tokens, duplicates")
    s.add_argument("path")
    s.set_defaults(func=cmd_stats)

    sp = sub.add_parser("split", help="deterministic train/val split")
    sp.add_argument("source")
    sp.add_argument("train")
    sp.add_argument("val")
    sp.add_argument("--ratio", type=float, default=0.9, help="train share (default 0.9)")
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(func=cmd_split)

    c = sub.add_parser("convert", help="CSV (prompt,answer[,system]) -> chat JSONL")
    c.add_argument("source")
    c.add_argument("dest")
    c.set_defaults(func=cmd_convert)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
