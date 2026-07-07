import os
import os.path as osp
import pickle
import numpy as np
import json
import argparse


def compute_stats(data):
    all_actions = []
    for scene in data:
        action = scene["action"]
        if isinstance(action, list):
            action = np.concatenate(action)
        all_actions.append(action)
    all_actions = np.concatenate(all_actions, axis=0)

    mean = np.mean(all_actions, axis=0)
    std = np.std(all_actions, axis=0)
    q01 = np.percentile(all_actions, 1, axis=0)
    q99 = np.percentile(all_actions, 99, axis=0)
    return mean, std, q01, q99


def main(pkl_dir, output_json_path=None):
    pkl_files = sorted([f for f in os.listdir(pkl_dir) if f.endswith(".pkl")])
    if not pkl_files:
        print(f"No .pkl files found in {pkl_dir}")
        return

    for pkl_file in pkl_files:
        pkl_path = osp.join(pkl_dir, pkl_file)
        print(f"\nLoading: {pkl_file}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        print(f"  Scenes: {len(data)}")

        mean, std, q01, q99 = compute_stats(data)

        print(f"  Mean:  {mean}")
        print(f"  Std:   {std}")
        print(f"  Q01:   {q01}")
        print(f"  Q99:   {q99}")

        if output_json_path:
            name = osp.splitext(pkl_file)[0]
            out_path = osp.join(output_json_path, f"{name}_stats.json")
            os.makedirs(output_json_path, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "q01": q01.tolist(),
                    "q99": q99.tolist(),
                }, f, indent=2)
            print(f"  Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute action normalization statistics from pickle files.")
    parser.add_argument("--pkl_dir", type=str, default="/mnt/mnt/storage/projects/UD-VLA/processed_data/meta",
                        help="Directory containing .pkl files.")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional path to save stats as JSON.")
    args = parser.parse_args()

    main(args.pkl_dir, args.output_json)
