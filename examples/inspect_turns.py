"""Quick look at the per-turn scores CSV written by wildchat_run.py (stdlib only).

Usage:
    python examples/inspect_turns.py results/<run>/<turns csv or run dir>
        [--config both|previous_assistant] [--language English] [--k 8]

Reads one CSV with a row per turn and prints, for one config: the weight
distribution of USER turns, a length baseline (rank correlation of gain with the
number of chunks), the behaviour of very short turns, and the lowest/highest
weight turns.

The first user turn of a conversation has no anchor, so its weight is 1 by
convention. Those turns are EXCLUDED from the weight statistics and the
low/high lists (they would otherwise fill the "highest weight" list).
"""
import argparse
import csv
import glob
import os
import statistics

REQUIRED = ["config", "role", "n_chunks", "mean_weight", "ig"]


def find_csv(path):
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "*turn*.csv")))
        if len(found) != 1:
            raise SystemExit(f"Expected exactly one *turn*.csv in {path}, found: {found}")
        return found[0]
    return path


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--config", default="both|previous_assistant")
    ap.add_argument("--language", default=None)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    path = find_csv(args.path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("CSV is empty")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise SystemExit(f"Missing columns {missing}. Columns found: {list(rows[0])}")
    configs = sorted({r["config"] for r in rows})
    if args.config not in configs:
        raise SystemExit(f"Unknown config. Available: {configs}")

    # first user turn per (conversation, config): unanchored, weight 1 by convention
    have_ids = "conversation_id" in rows[0] and "turn_index" in rows[0]
    first = {}
    if have_ids:
        for r in rows:
            if r["role"] == "user":
                key = (r["conversation_id"], r["config"])
                t = int(r["turn_index"])
                first[key] = min(t, first.get(key, t))

    def is_first_user(r):
        return have_ids and r["role"] == "user" and int(r["turn_index"]) == first[(r["conversation_id"], r["config"])]

    def sel(role):
        return [r for r in rows if r["config"] == args.config and r["role"] == role
                and (args.language is None or r.get("language") == args.language)
                and int(r["n_chunks"]) > 0]

    print(f"file: {path}\nconfig: {args.config} | language: {args.language or 'all'}")
    if not have_ids:
        print("WARNING: no conversation_id/turn_index columns, so unanchored first user turns cannot be excluded.")

    users_anchored = []
    for role in ("user", "assistant"):
        everything = sel(role)
        if not everything:
            print(f"\n[{role}] no turns with chunks")
            continue
        chosen = [r for r in everything if not is_first_user(r)]
        if role == "user":
            users_anchored = chosen
        g_all = [float(r["ig"]) for r in everything]
        n_all = [float(r["n_chunks"]) for r in everything]
        note = f" ({len(everything) - len(chosen)} first user turns excluded from weight stats)" if role == "user" and have_ids else ""
        print(f"\n[{role}] {len(everything)} turns with chunks{note}")
        if chosen:
            w = [float(r["mean_weight"]) for r in chosen]
            print(f"  mean_weight  min {min(w):.3f} | p10 {pct(w, .1):.3f} | median {pct(w, .5):.3f} "
                  f"| p90 {pct(w, .9):.3f} | max {max(w):.3f}")
            print(f"  share with mean_weight < 0.1: {sum(x < 0.1 for x in w) / len(w):.1%} "
                  f"| < 0.3: {sum(x < 0.3 for x in w) / len(w):.1%}")
        print(f"  Spearman(ig, n_chunks) = {spearman(g_all, n_all):.3f}   (close to 1 => gain mostly tracks length)")
        print(f"  mean ig per chunk = {sum(g_all) / sum(n_all):.3f}")

    if users_anchored and "text_prefix" in rows[0]:
        short = [r for r in users_anchored if 0 < len(r["text_prefix"].strip()) < 30]
        print(f"\n[user, anchored] very short turns (< 30 characters, whole turn visible): {len(short)}")
        if short:
            print(f"  mean weight {statistics.fmean(float(r['mean_weight']) for r in short):.3f} "
                  f"vs all anchored user turns {statistics.fmean(float(r['mean_weight']) for r in users_anchored):.3f}")
        by_w = sorted(users_anchored, key=lambda r: float(r["mean_weight"]))
        for title, part in (("LOWEST weight", by_w[:args.k]), ("HIGHEST weight", by_w[-args.k:][::-1])):
            print(f"\n[user, anchored] {title}")
            for r in part:
                text = r["text_prefix"].replace("\n", " ")[:70] if r.get("text_prefix") else "(no text: run without --no-text)"
                print(f"  w={float(r['mean_weight']):.2f} ig={float(r['ig']):6.2f} chunks={r['n_chunks']:>2}  {text}")


if __name__ == "__main__":
    main()
