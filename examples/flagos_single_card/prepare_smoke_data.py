"""Create tiny public synthetic Parquet files for both SFT and GRPO.

Requires already installed pyarrow. Does not download data, models or packages.
"""

import argparse
import os
import tempfile
from pathlib import Path


def build_rows(split, count):
    if split not in {"train", "val"} or count < 1:
        raise ValueError("split must be train/val and count must be positive")
    offset = 0 if split == "train" else 100
    rows = []
    for index in range(count):
        left = offset + index + 1
        right = index % 7 + 2
        question = f"What is {left} + {right}? Reply with only the integer."
        answer = str(left + right)
        rows.append(
            {
                "question": question,
                "answer": answer,
                "prompt": [{"role": "user", "content": question}],
                "data_source": "flagos_smoke_arithmetic",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": index},
            }
        )
    return rows


def write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        pq.write_table(pa.Table.from_pylist(rows), temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/flagos_single_card"))
    parser.add_argument("--train-rows", type=int, default=16)
    parser.add_argument("--val-rows", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing the two data files")
    args = parser.parse_args()
    datasets = {"train": build_rows("train", args.train_rows), "val": build_rows("val", args.val_rows)}
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for split in datasets:
        path = output / f"{split}.parquet"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new directory or --overwrite")
    for split, rows in datasets.items():
        path = output / f"{split}.parquet"
        write_parquet(path, rows)
        print(f"Wrote {len(rows)} synthetic {split} rows: {path}")


if __name__ == "__main__":
    main()
