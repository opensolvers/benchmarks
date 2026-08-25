#!/usr/bin/env python3
"""Summarize ORT chrome-trace profile JSON from run_real_llm_ort profiling mode.

Splits prefill vs decode using SequentialExecutor::Execute windows (run_tag
is not propagated into node events on ORT 1.29).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def op_type(name: str, args: dict) -> str:
    if isinstance(args, dict):
        if args.get("op_name"):
            return str(args["op_name"])
    base = name.replace("_kernel_time", "")
    for tok in reversed(base.split("/")):
        if tok and tok[0].isupper():
            return re.sub(r"(_\d+)+$", "", tok)
    return base.split("/")[-1] or name


def load_events(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("traceEvents") or data.get("events") or []
    return data


def run_windows(events):
    runs = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("name") == "SequentialExecutor::Execute"
    ]
    if len(runs) >= 2:
        prefill = runs[0]
        decode = runs[-1]
        return {
            "prefill": (prefill["ts"], prefill["ts"] + prefill["dur"]),
            "decode": (decode["ts"], decode["ts"] + decode["dur"]),
        }
    if len(runs) == 1:
        r = runs[0]
        return {"all": (r["ts"], r["ts"] + r["dur"])}
    return {"all": (0, 1 << 62)}


def summarize(events, window=None, top=30):
    by_op = defaultdict(lambda: {"us": 0.0, "count": 0, "nodes": set()})
    total_us = 0.0
    t0, t1 = window if window else (0, 1 << 62)

    for ev in events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        if ev.get("cat") != "Node":
            continue
        ts = ev.get("ts", 0)
        if not (t0 <= ts <= t1):
            continue
        dur = float(ev.get("dur", 0))
        name = ev.get("name", "?")
        args = ev.get("args") or {}
        ot = op_type(name, args)
        by_op[ot]["us"] += dur
        by_op[ot]["count"] += 1
        by_op[ot]["nodes"].add(name)
        total_us += dur

    rows = sorted(by_op.items(), key=lambda x: -x[1]["us"])
    return rows, total_us


def matmul_breakdown(events, window):
    t0, t1 = window
    by_proj = defaultdict(float)
    by_node = []
    for e in events:
        if e.get("ph") != "X" or e.get("cat") != "Node":
            continue
        if not (t0 <= e["ts"] <= t1):
            continue
        args = e.get("args") or {}
        if args.get("op_name") != "MatMulNBits":
            continue
        dur = e["dur"]
        name = e.get("name", "")
        by_node.append((dur, name))
        for proj in (
            "lm_head",
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ):
            if proj in name:
                by_proj[proj] += dur
                break
        else:
            by_proj["other"] += dur
    return by_proj, sorted(by_node, reverse=True)


def print_table(rows, total_us, top):
    print(f"total_kernel_ms={total_us/1000:.1f}")
    print(f"{'op_type':<28} {'ms':>8} {'%':>6} {'count':>6}  sample_node")
    print("-" * 80)
    for ot, st in rows[:top]:
        ms = st["us"] / 1000.0
        pct = 100.0 * st["us"] / total_us if total_us else 0
        sample = next(iter(st["nodes"])).replace("_kernel_time", "")
        if len(sample) > 36:
            sample = "…" + sample[-35:]
        print(f"{ot:<28} {ms:8.1f} {pct:5.1f}% {st['count']:6d}  {sample}")


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <profile.json> [prefill|decode|all] [topN]", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "decode"
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 25

    events = load_events(path)
    wins = run_windows(events)
    window = wins.get(mode, wins.get("all"))

    rows, total_us = summarize(events, window=window, top=top)
    wall = (window[1] - window[0]) / 1000.0
    print(f"file={path.name} mode={mode} wall_ms~={wall:.0f} (SequentialExecutor span)")
    print_table(rows, total_us, top)

    if mode == "decode":
        by_proj, slow = matmul_breakdown(events, window)
        if by_proj:
            print("\nMatMulNBits by projection (decode):")
            for k, v in sorted(by_proj.items(), key=lambda x: -x[1]):
                print(f"  {k:12} {v/1000:8.1f} ms")
            print("\nSlowest MatMulNBits nodes:")
            for dur, name in slow[:6]:
                print(f"  {dur/1000:7.1f} ms  {name.replace('_kernel_time','')[-60:]}")


if __name__ == "__main__":
    main()
