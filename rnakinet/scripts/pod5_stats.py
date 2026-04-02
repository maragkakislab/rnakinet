import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pod5


def compute_signal_stats(signal: np.ndarray) -> dict:
    signal = signal.astype(float)

    mean = float(np.mean(signal))
    median = float(np.median(signal))
    
    mean_ad = float(np.mean(np.abs(signal - mean)))
    median_ad = float(np.median(np.abs(signal - median)))
    std = float(np.std(signal))

    min_v = float(np.min(signal))
    max_v = float(np.max(signal))
    value_range = float(max_v - min_v)

    length = int(len(signal))

    return {
        "median": median,
        "median_ad": median_ad,
        "mean": mean,
        "mean_ad": mean_ad,
        "std": std,
        "min": min_v,
        "max": max_v,
        "range": value_range,
        "length": length,
    }

def collect_signal_stats(experiment_dir: Path, subsample: float) -> pd.DataFrame:
    experiment_dir = experiment_dir.resolve()
    experiment_name = experiment_dir.name
    pod5_files = sorted(experiment_dir.rglob("*.pod5"))

    rows = []
    for file_idx, pod5_path in enumerate(pod5_files, start=1):
        print(f"Processing {file_idx}/{len(pod5_files)}: {pod5_path.name}")
        with pod5.Reader(pod5_path) as reader:
            for read_idx, read in enumerate(reader.reads()):
                if read_idx % int(1.0 / subsample) == 0:
                    rows.append({"experiment": experiment_name,
                                 "file": pod5_path.name,
                                 "read_id": str(read.read_id),
                                 **compute_signal_stats(read.signal)}
                                )

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Sample reads from all .pod5 files under an experiment folder and write per-read signal stats.")
    p.add_argument("--experiment-dir", required=True, type=Path, help="Path to folder/experiment_name (script will search recursively for *.pod5).")
    p.add_argument("--output", required=True, type=Path, help="Output CSV path")
    p.add_argument("--subsample", required=True, type=float, help="Fraction of reads to sample, distributed evenly across and within files.")
    args = p.parse_args()

    df = collect_signal_stats(args.experiment_dir, args.subsample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()