"""
build/convert_timings_sweep.py
-------------------------------
Parses raw timing log files produced by run_m1_eval.bat (or run_sweep.py)
and converts them into a clean CSV ready for plot_results.py.

Each log line is expected to look like:
    D=128 B=4096 Timing (ms): gen 1.234  infer 0.567  weights 0.891  total 2.692
    D=128 B=4096 CPU Timing (ms): gen 12.3  infer 4.5  weights 3.2  total 20.0

Output CSV columns:
    D, B, gpu_gen, gpu_infer, gpu_weights, gpu_total, cpu_gen, cpu_infer, cpu_weights, cpu_total

Usage:
    python build/convert_timings_sweep.py                       # reads gpu_timings.txt + cpu_timings.txt
    python build/convert_timings_sweep.py --gpu=gpu.txt --cpu=cpu.txt --out=timings.csv
    python build/convert_timings_sweep.py --combined=all_timings.txt --out=timings.csv
"""

import argparse
import re
import csv
import sys
from collections import defaultdict


# ── Regex patterns ────────────────────────────────────────────────────────────

# Matches:  D=128 B=4096 Timing (ms): gen 1.234  infer 0.567  weights 0.891  total 2.692
GPU_RE = re.compile(
    r'D=(\d+)\s+B=(\d+)\s+Timing \(ms\):\s+gen\s+([\d.]+)\s+infer\s+([\d.]+)\s+weights\s+([\d.]+)\s+total\s+([\d.]+)'
)

# Matches:  D=128 B=4096 CPU Timing (ms): gen 12.3  infer 4.5  weights 3.2  total 20.0
CPU_RE = re.compile(
    r'D=(\d+)\s+B=(\d+)\s+(?:CPU\s+)?Timing \(ms\):\s+gen\s+([\d.]+)\s+infer\s+([\d.]+)\s+weights\s+([\d.]+)\s+total\s+([\d.]+)'
)

# Also handle the bare format from main.cu printf (no D= B= prefix):
#   Timing (ms): gen 1.234  infer 0.567  weights 0.891  total 2.692
BARE_GPU_RE = re.compile(
    r'Timing \(ms\):\s+gen\s+([\d.]+)\s+infer\s+([\d.]+)\s+weights\s+([\d.]+)\s+total\s+([\d.]+)'
)
BARE_CPU_RE = re.compile(
    r'(?:CPU\s+)?Timing \(ms\):\s+gen\s+([\d.]+)\s+infer\s+([\d.]+)\s+weights\s+([\d.]+)\s+total\s+([\d.]+)'
)

# D/B tag that might precede a bare line: "D=128 B=4096"
DB_TAG_RE = re.compile(r'D=(\d+)\s+B=(\d+)')


# ── Parsers ───────────────────────────────────────────────────────────────────

def _aggregate_runs(values, drop_first):
    if not values:
        return None
    trimmed = values[drop_first:] if drop_first > 0 and len(values) > drop_first else values
    n = float(len(trimmed))
    return {
        'gen': sum(v['gen'] for v in trimmed) / n,
        'infer': sum(v['infer'] for v in trimmed) / n,
        'weights': sum(v['weights'] for v in trimmed) / n,
        'total': sum(v['total'] for v in trimmed) / n,
    }


def parse_file(path, kind='gpu', drop_first=0):
    """
    Parse a log file and return a dict keyed by (D, B) with timing values.
    kind: 'gpu' or 'cpu'
    """
    by_key = defaultdict(list)
    current_D = current_B = None

    try:
        lines = open(path).readlines()
    except FileNotFoundError:
        print(f"[WARN] File not found: {path} — skipping")
        return {}

    for line in lines:
        line = line.strip()

        # Check for prefixed D= B= format
        if kind == 'gpu':
            m = GPU_RE.search(line)
            if m:
                D, B = int(m.group(1)), int(m.group(2))
                by_key[(D, B)].append({
                    'gen': float(m.group(3)), 'infer': float(m.group(4)),
                    'weights': float(m.group(5)), 'total': float(m.group(6))
                })
                continue
        else:
            m = CPU_RE.search(line)
            if m:
                D, B = int(m.group(1)), int(m.group(2))
                by_key[(D, B)].append({
                    'gen': float(m.group(3)), 'infer': float(m.group(4)),
                    'weights': float(m.group(5)), 'total': float(m.group(6))
                })
                continue

        # Check for D= B= tag to associate with next bare line
        tag = DB_TAG_RE.search(line)
        if tag:
            current_D, current_B = int(tag.group(1)), int(tag.group(2))

        # Check for bare format (needs current_D, current_B from previous tag line)
        if kind == 'gpu':
            m = BARE_GPU_RE.search(line)
        else:
            m = BARE_CPU_RE.search(line)

        if m and current_D is not None and current_B is not None:
            by_key[(current_D, current_B)].append({
                'gen': float(m.group(1)), 'infer': float(m.group(2)),
                'weights': float(m.group(3)), 'total': float(m.group(4))
            })

    results = {}
    for key, values in by_key.items():
        agg = _aggregate_runs(values, drop_first)
        if agg is not None:
            results[key] = agg
    return results


def parse_combined(path):
    """Parse a single file that contains both GPU and CPU lines."""
    gpu = {}
    cpu = {}
    current_D = current_B = None

    try:
        lines = open(path).readlines()
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    for line in lines:
        line = line.strip()
        tag = DB_TAG_RE.search(line)
        if tag:
            current_D, current_B = int(tag.group(1)), int(tag.group(2))

        m = GPU_RE.search(line)
        if m:
            D, B = int(m.group(1)), int(m.group(2))
            gpu[(D, B)] = {'gen': float(m.group(3)), 'infer': float(m.group(4)),
                           'weights': float(m.group(5)), 'total': float(m.group(6))}
            continue

        m = CPU_RE.search(line)
        if m:
            D, B = int(m.group(1)), int(m.group(2))
            cpu[(D, B)] = {'gen': float(m.group(3)), 'infer': float(m.group(4)),
                           'weights': float(m.group(5)), 'total': float(m.group(6))}
            continue

        m = BARE_GPU_RE.search(line)
        if m and current_D and 'CPU' not in line:
            gpu[(current_D, current_B)] = {
                'gen': float(m.group(1)), 'infer': float(m.group(2)),
                'weights': float(m.group(3)), 'total': float(m.group(4))}
            continue

        m = BARE_CPU_RE.search(line)
        if m and current_D:
            cpu[(current_D, current_B)] = {
                'gen': float(m.group(1)), 'infer': float(m.group(2)),
                'weights': float(m.group(3)), 'total': float(m.group(4))}

    return gpu, cpu


# ── Writer ────────────────────────────────────────────────────────────────────

def write_csv(gpu, cpu, out_path):
    all_keys = sorted(set(gpu.keys()) | set(cpu.keys()))
    if not all_keys:
        print("[ERROR] No timing entries found. Check your log format.")
        sys.exit(1)

    fieldnames = ['D', 'B',
                  'gpu_gen', 'gpu_infer', 'gpu_weights', 'gpu_total',
                  'cpu_gen', 'cpu_infer', 'cpu_weights', 'cpu_total']

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (D, B) in all_keys:
            g = gpu.get((D, B), {})
            c = cpu.get((D, B), {})
            writer.writerow({
                'D': D, 'B': B,
                'gpu_gen':     g.get('gen',     ''),
                'gpu_infer':   g.get('infer',   ''),
                'gpu_weights': g.get('weights', ''),
                'gpu_total':   g.get('total',   ''),
                'cpu_gen':     c.get('gen',     ''),
                'cpu_infer':   c.get('infer',   ''),
                'cpu_weights': c.get('weights', ''),
                'cpu_total':   c.get('total',   ''),
            })

    print(f"Wrote {len(all_keys)} rows -> {out_path}")
    for (D, B) in all_keys:
        g = gpu.get((D, B), {})
        c = cpu.get((D, B), {})
        print(f"  D={D:4d}  B={B:6d}  gpu={g.get('total','?')} ms  cpu={c.get('total','?')} ms")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert timing logs to CSV")
    parser.add_argument("--gpu",      type=str, default="gpu_timings.txt",
                        help="GPU timing log file")
    parser.add_argument("--cpu",      type=str, default="cpu_timings.txt",
                        help="CPU timing log file")
    parser.add_argument("--combined", type=str, default=None,
                        help="Single file with both GPU and CPU lines (overrides --gpu/--cpu)")
    parser.add_argument("--drop-first", type=int, default=0,
                        help="Drop first N runs per (D,B) before averaging")
    parser.add_argument("--out",      type=str, default="timings.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    if args.combined:
        print(f"Parsing combined log: {args.combined}")
        gpu, cpu = parse_combined(args.combined)
    else:
        print(f"Parsing GPU log: {args.gpu}")
        gpu = parse_file(args.gpu, kind='gpu', drop_first=max(0, args.drop_first))
        print(f"Parsing CPU log: {args.cpu}")
        cpu = parse_file(args.cpu, kind='cpu', drop_first=max(0, args.drop_first))

    print(f"Found {len(gpu)} GPU entries, {len(cpu)} CPU entries")
    write_csv(gpu, cpu, args.out)


if __name__ == "__main__":
    main()
