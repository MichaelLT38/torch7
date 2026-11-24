"""PyTorch reference timings for the ops in bench/th_bench.c (optional tool).

    python bench/bench_pytorch.py [threads ...] > pytorch.csv
    luajit bench/compare.lua pytorch.csv th.csv   # speedup < 1: TH is slower

Output format matches th_bench: '#' header lines, then "op,threads,ms".
"""
import statistics
import sys
import time

import torch

REPS = 15


def bench(name, nt, fn):
    fn()  # warm-up
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    print(f"{name},{nt},{statistics.median(times):.4f}", flush=True)


def run(nt):
    torch.set_num_threads(nt)
    torch.manual_seed(0)
    with torch.no_grad():
        n = 10_000_000
        a, b, c = torch.rand(n) * 2 - 1, torch.rand(n) * 2 - 1, torch.empty(n)
        bench("add_10M", nt, lambda: torch.add(a, b, out=c))
        bench("mul_scalar_10M", nt, lambda: torch.mul(a, 2.0, out=c))
        bench("cmul_10M", nt, lambda: torch.mul(a, b, out=c))
        bench("exp_10M", nt, lambda: torch.exp(a, out=c))
        bench("tanh_10M", nt, lambda: torch.tanh(a, out=c))
        bench("sigmoid_10M", nt, lambda: torch.sigmoid(a, out=c))
        bench("sum_all_10M", nt, lambda: a.sum())

        m = 4096
        x, y, z = torch.rand(m, m) * 2 - 1, torch.rand(m, m) * 2 - 1, torch.empty(m, m)
        yt, rs = y.t(), torch.empty(m)
        bench("add_transposed_4096sq", nt, lambda: torch.add(x, yt, out=z))
        bench("copy_transpose_4096sq", nt, lambda: z.copy_(yt))
        bench("sum_dim0_4096sq", nt, lambda: torch.sum(x, 0, out=rs))
        bench("sum_dim1_4096sq", nt, lambda: torch.sum(x, 1, out=rs))

        k = 512
        m1, m2, mo = torch.rand(k, k), torch.rand(k, k), torch.empty(k, k)
        bench("matmul_512", nt, lambda: torch.mm(m1, m2, out=mo))

        s = torch.rand(1_000_000)
        bench("sort_1M", nt, lambda: torch.sort(s))


def main():
    print(f"# bench_pytorch: torch {torch.__version__}, max threads {torch.get_num_threads()}")
    print("op,threads,ms")
    threads = [int(t) for t in sys.argv[1:]] or sorted({1, torch.get_num_threads()})
    for nt in threads:
        run(nt)


if __name__ == "__main__":
    main()
