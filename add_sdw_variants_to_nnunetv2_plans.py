#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add 18 U-Net scaling variants (S/D/W) to nnU-Net v2 plans based on 3d_fullres.

Paper alignment:
- We benchmark 18 U-Net variants defined by three orthogonal hyperparameters:
  S ∈ {4, 5, 6}  (resolution stages)
  D ∈ {2, 3}     (conv layers per stage)
  W ∈ {16, 32, 64} (initial channel width)
- Each new configuration is named as: 3d_fullres_S{S}D{D}W{W}

Usage:
    python add_sdw_variants_to_plans.py -d DATASET_ID
Example:
    python add_sdw_variants_to_plans.py -d 5   # => Dataset005_*
"""

import json
import argparse
import os
import sys
import shutil
from copy import deepcopy

def die(msg: str, code: int = 1):
    print(f"Error: {msg}")
    sys.exit(code)

def main():
    parser = argparse.ArgumentParser(
        description='Add 18 S/D/W U-Net variants to nnU-Net v2 plans (based on 3d_fullres).'
    )
    parser.add_argument(
        '-d', '--dataset-id', type=int, required=True,
        help='Dataset ID (e.g., 5 for Dataset005_XXX)'
    )
    args = parser.parse_args()

    # nnUNet v2 expects env var 'nnUNet_preprocessed'
    nnunet_preprocessed = os.environ.get('nnUNet_preprocessed')
    if not nnunet_preprocessed:
        die("nnUNet_preprocessed environment variable not set")

    # Locate dataset folder by prefix "Dataset{ID:03d}_"
    dataset_prefix = f"Dataset{args.dataset_id:03d}"
    dataset_name = next(
        (f for f in os.listdir(nnunet_preprocessed) if f.startswith(dataset_prefix)),
        None
    )
    if not dataset_name:
        die(f"Dataset {args.dataset_id} not found in {nnunet_preprocessed}")

    plans_path = os.path.join(nnunet_preprocessed, dataset_name, 'nnUNetPlans.json')
    if not os.path.exists(plans_path):
        die(f"Plans file not found: {plans_path}")

    with open(plans_path, 'r') as f:
        data = json.load(f)

    # Require 3d_fullres as the base configuration, consistent with the paper
    configs_dict = data.get('configurations', {})
    if '3d_fullres' not in configs_dict:
        die("3d_fullres configuration not found in plans")

    original_config = configs_dict['3d_fullres']

    # ---- Define 18 (S, D, W) variants as in the paper ----
    S_set = [4, 5, 6]     # resolution stages
    D_set = [2, 3]        # conv per stage
    W_set = [16, 32, 64]  # initial width
    triplets = [(S, D, W) for S in S_set for D in D_set for W in W_set]

    print(f"Adding S/D/W variants to {dataset_name} based on 3d_fullres...")
    added = 0

    for S, D, W in triplets:
        short_name = f"S{S}D{D}W{W}"
        config_name = f"3d_fullres_{short_name}"

        if config_name in configs_dict:
            print(f"Skipping {config_name} (already exists)")
            continue

        # Deep copy the base config to preserve all dataset-specific settings
        new_cfg = deepcopy(original_config)

        # Access architecture kwargs (nnU-Net v2 plans schema)
        arch = new_cfg.get("architecture", {})
        arch_kwargs = arch.get("arch_kwargs", None)
        if arch_kwargs is None:
            die("Unexpected plans format: missing architecture.arch_kwargs")

        # ------ Paper-aligned edits ------
        # 1) Number of stages S
        arch_kwargs["n_stages"] = S

        # 2) Width schedule: features_per_stage
        #    Start from W at stage-0, double each stage until cap (512), length=S.
        #    This mirrors the conventional U-Net channel progression and matches
        #    the paper’s controlled width scaling.
        feats = []
        cur = W
        for i in range(S):
            feats.append(min(cur, 512))
            if i < S - 1:
                cur = min(cur * 2, 512)
        arch_kwargs["features_per_stage"] = feats

        # 3) Depth D: convs per encoder/decoder stage
        arch_kwargs["n_conv_per_stage"] = [D] * S
        arch_kwargs["n_conv_per_stage_decoder"] = [D] * (S - 1)

        # 4) Kernel sizes / strides:
        #    Keep dataset-derived schedules from the base 3d_fullres.
        #    If base lengths ≠ S, extend with the last entry (for larger S)
        #    or truncate (for smaller S), preserving anisotropy decisions.
        ks = arch_kwargs.get("kernel_sizes", [])
        st = arch_kwargs.get("strides", [])
        if len(ks) != S or len(st) != S:
            last_k = ks[-1] if ks else [3, 3, 3]
            last_s = st[-1] if st else [2, 2, 2]
            if S > len(ks):
                ks = ks + [last_k] * (S - len(ks))
                st = st + [last_s] * (S - len(st))
            else:
                ks = ks[:S]
                st = st[:S]
            arch_kwargs["kernel_sizes"] = ks
            arch_kwargs["strides"] = st

        # Insert new configuration
        data['configurations'][config_name] = new_cfg
        added += 1
        print(f"Added {config_name}")

    # ---- Save with backup to be safe ----
    if added > 0:
        backup_path = plans_path + ".bak"
        try:
            shutil.copy2(plans_path, backup_path)
            print(f"Backup created: {backup_path}")
        except Exception as e:
            print(f"Warning: failed to create backup: {e}")

        with open(plans_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\nSuccessfully added {added} configurations to {plans_path}")
    else:
        print("No new configurations added (all already exist)")

    print(f"Total configurations now: {len(data['configurations'])}")

if __name__ == "__main__":
    main()
