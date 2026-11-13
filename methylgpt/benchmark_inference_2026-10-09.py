#!/usr/bin/env python3
"""Inference time and memory of the released package (methylgpt, bundle methylgpt-1.1).

Input: the six public example profiles shipped with the package (GSE54848), repeated to the requested number of
profiles; timings do not depend on the values. Measured after one warm-up call: predict() (age as the three-run
mean, sex, tissue and 25 clock outputs), factor_scores() and impute() (factor-model reconstruction).
GPU: wall time with torch.cuda.synchronize and peak allocated memory. CPU: 8 threads.
Output: artifacts/eval/inference_benchmark_2026-10-09.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "package/src"))
from methylgpt import MethylGPT  # noqa: E402

BUNDLE = HERE / "artifacts/release/methylgpt2-1.1"
OUT = HERE / "artifacts/eval/inference_benchmark_2026-10-09.json"


def timed(fn, dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t = time.perf_counter(); fn()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return time.perf_counter() - t


def run(device, n):
    try:
        ex = pd.read_csv(HERE / "package/examples/GSE54848_example_betas.csv", index_col=0)
        m = MethylGPT.from_pretrained(BUNDLE, device=device)
    except OSError as e:
        sys.exit(f"Cannot proceed - required data file not found: {e}")
    X = pd.concat([ex] * int(np.ceil(n / len(ex))), ignore_index=True).iloc[:n]
    dev = m.device
    m.predict(X.iloc[:8]); m.impute(X.iloc[:8]); m.factor_scores(X.iloc[:8])          # warm-up
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    r = {"profiles": n, "predict_s": timed(lambda: m.predict(X), dev), "factor_scores_s": timed(lambda: m.factor_scores(X), dev),
         "impute_s": timed(lambda: m.impute(X), dev)}
    r.update({k.removesuffix("_s") + "_ms_per_profile": 1000 * v / n for k, v in list(r.items()) if k.endswith("_s")})
    if dev.type == "cuda":
        r["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        r["gpu"] = torch.cuda.get_device_name(dev)
    else:
        r["cpu_threads"] = torch.get_num_threads()
    return r


if __name__ == "__main__":
    gpu = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    res = {"bundle": BUNDLE.name, "gpu": run(gpu, 256)}
    torch.set_num_threads(8)
    res["cpu"] = run("cpu", 24)
    res["torch"] = torch.__version__
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
