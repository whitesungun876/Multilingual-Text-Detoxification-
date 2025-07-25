# scripts/data_prep/split_parallel.py

import json
import os
import random
import argparse

def split_json(input_path: str, train_ratio: float, seed: int):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = len(data)
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)

    train_size = int(n * train_ratio)
    train_idx = set(indices[:train_size])
    train_list = [data[i] for i in range(n) if i in train_idx]
    dev_list   = [data[i] for i in range(n) if i not in train_idx]
    return train_list, dev_list

def save_json(data_list, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data_list)} examples to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="divide JSON into Train/Dev "
    )
    parser.add_argument(
        "--lang", choices=["en", "zh", "de"], required=True)
    parser.add_argument(
        "--input_dir", type=str, required=True)
    parser.add_argument(
        "--output_dir", type=str, required=True)
    parser.add_argument(
        "--train_ratio", type=float, default=0.9)
    parser.add_argument(
        "--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = os.path.join(args.input_dir, f"{args.lang}_all.json")
    os.makedirs(args.output_dir, exist_ok=True)
    train_out = os.path.join(args.output_dir, f"{args.lang}_train_split.json")
    dev_out   = os.path.join(args.output_dir, f"{args.lang}_dev_split.json")

    train_list, dev_list = split_json(input_path, args.train_ratio, args.seed)

    save_json(train_list, train_out)
    save_json(dev_list,   dev_out)

if __name__ == "__main__":
    main()
