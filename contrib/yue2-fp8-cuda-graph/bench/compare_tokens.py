"""Compare two YuE2 semantic.pkl files (eager vs CUDA graph, same seed).

Prints: token counts, length of the common prefix (first divergence point), and the
fraction of identical tokens up to the shorter length. Usage:
    python compare_tokens.py eager/semantic.pkl graph/semantic.pkl
"""
import pickle
import sys


def load_tokens(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    toks = getattr(obj, "tokens", obj)
    return [int(t) for t in toks]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    a, b = (load_tokens(p) for p in sys.argv[1:3])
    n = min(len(a), len(b))
    prefix = 0
    while prefix < n and a[prefix] == b[prefix]:
        prefix += 1
    same = sum(1 for i in range(n) if a[i] == b[i])
    print(f"A tokens: {len(a)}")
    print(f"B tokens: {len(b)}")
    print(f"identical prefix: {prefix} tokens ({'bit-identical' if prefix == len(a) == len(b) else 'diverge at token ' + str(prefix)})")
    print(f"position-wise equal tokens over the first {n}: {same} ({100.0 * same / n:.1f} %)")


if __name__ == "__main__":
    main()
