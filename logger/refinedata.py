import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = "train_data"
OUTPUT_DIR = "train_data_refined"

MIN_TIMESTEPS = 900
MAX_ZERO_RATE = 0.20
EPS_EQ0 = 1e-6
TRUNCATE_STEPS = 230

input_path = Path(INPUT_DIR)
output_path = Path(OUTPUT_DIR)

os.makedirs(output_path, exist_ok=True)

all_files = sorted(list(input_path.rglob("*.npz")))
kept_files_count = 0

for in_file_path in tqdm(all_files, desc="Refining dataset"):
    
    with np.load(in_file_path, allow_pickle=False) as d:
        action_data = d['action']

    num_timesteps = len(action_data)
    
    if num_timesteps <= MIN_TIMESTEPS:
        continue

    v = action_data[:, 0].astype(np.float32)
    zero_rate = float(np.mean(np.abs(v) <= EPS_EQ0))

    if zero_rate >= MAX_ZERO_RATE:
        continue

    full_npz_data = np.load(in_file_path, allow_pickle=True)
    processed_data_dict = {}

    for key, array_data in full_npz_data.items():
        if len(array_data) > TRUNCATE_STEPS:
            processed_data_dict[key] = array_data[:-TRUNCATE_STEPS]
        else:
            processed_data_dict[key] = array_data
    
    out_file_path = output_path / in_file_path.name
    np.savez_compressed(out_file_path, **processed_data_dict)
    kept_files_count += 1

print(f"Complete refine data sets. Kept {kept_files_count} out of {len(all_files)} files in {output_path.resolve()}")